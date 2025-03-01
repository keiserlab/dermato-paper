from datetime import datetime
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import yaml
import pickle
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import jaccard_score, f1_score
from torchvision.models import densenet121
import torchvision
from captum.attr import GuidedGradCam, NoiseTunnel, visualization as viz
from collections import OrderedDict, defaultdict
import zarr
from PIL import Image
import kornia as K
# from eval import StainDataSet

class StainDataSet(Dataset):
    """

    Initializes an instance of the class.
    Args:
        df (DataFrame): The input DataFrame.

    """
    def __init__(self, df):
        self.df = df
        self.tile_size = 256
        
        # compute the baseline, the ratio of positive to negative samples
        self._compute_baseline()

    def _compute_baseline(self):
        """
        Compute the baseline accuracy
        """
        # sum the number of positive and negative samples   
        num_positives = np.sum(self.df['label']) # sum of the labels = number of positives since label=1
        num_negatives = self.df.shape[0] - num_positives
        # compute the ratio of positive to negative samples
        self.ratio_positives = round(num_positives / self.df.shape[0] * 100, 2)
        self.ratio_negatives = round(num_negatives / self.df.shape[0] * 100, 2)
        print(f'Baseline accuracy: {self.ratio_positives}% positives, {self.ratio_negatives}% negatives')
        

    def __len__(self):
        # return the number of tiles, size of dataframe
        return self.df.shape[0]

    def __getitem__(self, idx):
        """
        
        Args:
            idx (int): Index to iterate through the dataframe
        """
        # get the paths to the H&E and IHC images
        he_section_path = self.df.iloc[idx]['he']
        ihc_section_path = self.df.iloc[idx]['ihc']

        # get the y, x coordinates of the tile
        y, x = self.df.iloc[idx]['y'], self.df.iloc[idx]['x']
        
        # get the label
        label = self.df.iloc[idx]['label']

        # TODO instead of loading the entire section, load only the tile by chunk for speed up
        # temporarily load the /scratch
        if os.path.splitext(he_section_path)[1] == '.zarr':
            he_section = zarr.open(he_section_path, 'r')  #[C,H,W] #
            # crop the tile from the section
            he_tile = he_section[y:y+self.tile_size, x:x+self.tile_size]
            h, w, c = he_tile.shape

            # pad the value 255 to bottom and right sides of the tile if it is smaller than the tile size
            if (h < self.tile_size) or (w < self.tile_size):
                he_tile = np.pad(he_tile, ((0, self.tile_size-h), (0, self.tile_size-w), (0,0)), 'constant', constant_values=(255.))
        
        elif os.path.splitext(he_section_path)[1] == '.jpg':
            he_tile = np.array(Image.open(he_section_path))

        # [H,W,C] -> [C,H,W] torch.uint8 then convert to float32 tensor in range [0,1]
        he_tile = K.image_to_tensor(he_tile, keepdim=True).float() / 255.0
        # label = torch.tensor(label, dtype=torch.float)
        metadata = defaultdict(dict)
        metadata['label'] = torch.tensor(label, dtype=torch.float)
        metadata['coords'] = torch.tensor((y, x), dtype = torch.int32)

        return he_tile, metadata

# --------------------------
# Model Initialization
# --------------------------

def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_state_dict[key.replace('module.', '')] = value
    return new_state_dict

def initialize_model():
    model = densenet121()
    model.classifier = nn.Linear(1024, 1)
    return model

def load_trained_models(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    melanA_model = initialize_model()
    melanA_model.load_state_dict(remove_module_prefix(
        torch.load(config['trained_model_path']['melanA'], map_location='cpu')['state_dict']
    ))
    
    sox10_model = initialize_model()
    sox10_model.load_state_dict(remove_module_prefix(
        torch.load(config['trained_model_path']['sox10'], map_location='cpu')['state_dict']
    ))
    
    return melanA_model.to('cuda:0').eval(), sox10_model.to('cuda:1').eval()

# --------------------------
# Data Transformation
# --------------------------

def load_normalization(norm_paths):
    transforms = {}
    for stain, path in norm_paths.items():
        with open(path, 'rb') as f:
            mean, std = pickle.load(f)
        transforms[stain] = torchvision.transforms.Compose([
            # torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std)
        ])
    return transforms

# --------------------------
# Saliency Generation Core
# --------------------------

def generate_attr_gradcam(model, input_tensor):
    #rewrite this to work batch-wise rather than on a single image
    layers = [name for name, module in model.named_modules() 
             if 'conv' in name and 'dense' in name]
    
    attributions = torch.zeros_like(input_tensor)
    for layer in tqdm(layers, desc = 'Layer progress for a single tile'):
        guided_gc = GuidedGradCam(model, eval(f'model.{layer}'))
        noise_tunnel = NoiseTunnel(guided_gc)
        attr = noise_tunnel.attribute(input_tensor, 
                                    nt_samples=5,
                                    nt_type='smoothgrad')
        attributions += attr
    
    #normalize by number of layers
    attributions /= len(layers)
    #put into image shape
    attr = np.transpose(attributions.squeeze(0).cpu().detach().numpy(), (1,2,0))
    
    return attr

# --------------------------
# Blob Processing Pipeline
# --------------------------

def preprocess_saliency_map(saliency_map):
    #squeeze dimensions to 1 and normalize using captum's normalize fxn
    saliency_normed = viz._normalize_attr(saliency_map, 'absolute_value', outlier_perc = 2.0, reduction_axis=2)
    #threshold the 1-dim attrs
    saliency_uint8 = (saliency_normed * 255).astype(np.uint8)
    _, thresh = cv2.threshold(saliency_uint8, 0, 255, 
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def apply_morphological_operations(binary_map, kernel_size=5, num_iters=2):
    """Enhanced morphological processing with iterative opening/closing"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    processed = binary_map.copy()
    
    for _ in range(num_iters):
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
    
    return processed

def detect_blobs(processed_map, min_area=100):
    contours, _ = cv2.findContours(processed_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= min_area]

def create_blob_masks(contours, shape):
    masks = []
    for cnt in contours:
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        masks.append(mask)
    return masks

# --------------------------
# Agreement Analysis
# --------------------------

def calculate_agreement(masks1, masks2):
    iou_scores, dice_scores = [], []
    for m1 in masks1:
        for m2 in masks2:
            if np.any(np.logical_and(m1, m2)):
                flat1 = (m1 > 0).flatten()
                flat2 = (m2 > 0).flatten()
                iou = jaccard_score(flat1, flat2)
                dice = f1_score(flat1, flat2)
                iou_scores.append(iou)
                dice_scores.append(dice)
    return iou_scores, dice_scores


# --------------------------
# Data loading
# --------------------------

def build_test_data_with_defined_section(saliency_config, stain):
    """
    Build test data with defined section.
    Args:
        splitID (int): The ID of the split.
        eval_config (dict): The evaluation configuration.
    Returns:
        tuple: A tuple containing a list of test data loaders and a list of all sections.
    
    """
    batch_size = saliency_config['batch_size']
    num_workers = saliency_config['num_workers']
    df_path = saliency_config['dataframe_path'][stain]

    list_test_dls = []
    df = pd.read_csv(df_path)
    all_sections = list(df['section_name'].unique())
    for section in all_sections:
        df_sec = df.loc[df['section_name']==section].reset_index(drop=True, inplace=False)
        print(f'{section}: {len(df_sec)} rows')
        
        test_ds = StainDataSet(df=df_sec)
        test_dl = DataLoader(
            test_ds, 
            batch_size=batch_size,
            num_workers=num_workers, 
            shuffle=False, #mutually exclusive with sampler when WRS is on
            pin_memory=True, 
            drop_last=False
            )
        list_test_dls.append(test_dl)

    return list_test_dls, all_sections


# --------------------------
# Main Processing Function
# --------------------------

def generate_saliency_maps(dataloader, melanA_model, sox10_model, norm_transforms,
                          output_dir="results", conf_thresholds=(0.9, 0.9),
                          min_blob_area=100, morph_kernel=5, morph_iters=2, max_tiles=None):
    
    Path(output_dir).mkdir(exist_ok=True)
    blob_records = []
    agreement_metrics = []
    visualization_paths = []

    device_melanA = next(melanA_model.parameters()).device
    device_sox10 = next(sox10_model.parameters()).device

    with torch.no_grad():
        for batch_idx, (tiles, metadata) in enumerate(tqdm(dataloader, desc = 'Batch-level progress')):
            print("Metadata:", metadata)
            if max_tiles and batch_idx >= max_tiles:
                break

            # Model processing
            melanA_input = norm_transforms['melanA'](tiles).to(device_melanA)
            sox10_input = norm_transforms['sox10'](tiles).to(device_sox10)
            
            melanA_preds = torch.sigmoid(melanA_model(melanA_input))
            sox10_preds = torch.sigmoid(sox10_model(sox10_input))
            
            qualified = (melanA_preds.cpu() > conf_thresholds[0]) & (sox10_preds.cpu() > conf_thresholds[1])
            
            # debug = False
            for idx in torch.where(qualified)[0]:
                tile_data = process_qualified_tile(
                    idx, tiles, metadata,
                    melanA_model, sox10_model,
                    melanA_input[idx], sox10_input[idx],
                    min_blob_area, morph_kernel, morph_iters,
                    output_dir, debug = False
                )
                
                blob_records.extend(tile_data['blobs'])
                agreement_metrics.append(tile_data['agreement'])
                save_enhanced_results(blob_records, agreement_metrics, visualization_paths, output_dir)
                # visualization_paths.append(tile_data['visualizations'])
            # Save incremental results
            if batch_idx % 1 == 0:
                # debug = True
                print("batch_idx", batch_idx)
                save_enhanced_results(blob_records, agreement_metrics, visualization_paths, output_dir)
    
    save_enhanced_results(blob_records, agreement_metrics, visualization_paths, output_dir)
    return blob_records, agreement_metrics, visualization_paths

def  process_qualified_tile(idx, tiles, metadata, melanA_model, sox10_model,
                          melanA_input, sox10_input, min_area, kernel_size, num_iters,
                          output_dir, debug = False):
    """Full processing pipeline with morphological ops and agreement analysis"""
    
    # Generate saliency maps
    melanA_attr = generate_attr_gradcam(melanA_model, melanA_input.unsqueeze(0))
    sox10_attr = generate_attr_gradcam(sox10_model, sox10_input.unsqueeze(0))

    # Preprocess and apply morphological operations
    processed = {
        'melanA': apply_morphological_operations(
            preprocess_saliency_map(melanA_attr), 
            kernel_size=kernel_size,
            num_iters=num_iters
        ),
        'sox10': apply_morphological_operations(
            preprocess_saliency_map(sox10_attr),
            kernel_size=kernel_size,
            num_iters=num_iters
        )
    }

    # Detect blobs and create masks
    melanA_blobs = detect_blobs(processed['melanA'], min_area)
    sox10_blobs = detect_blobs(processed['sox10'], min_area)
    
    melanA_masks = create_blob_masks(melanA_blobs, processed['melanA'].shape)
    sox10_masks = create_blob_masks(sox10_blobs, processed['sox10'].shape)
    
    # Calculate and save agreement metrics
    iou_scores, dice_scores = calculate_agreement(melanA_masks, sox10_masks)

    # Save visualizations if debug is True
    # if debug:
        # tile_id = metadata['tile_id'][idx]
        # tile_output_dir = Path(output_dir, f"tile_{tile_id}")
        # tile_output_dir.mkdir(exist_ok=True)
        # visualize_masks(melanA_attr, melanA_masks, tile_output_dir / "melanA_masks.png")
        # visualize_masks(sox10_attr, sox10_masks, tile_output_dir / "sox10_masks.png")
        # plot_histograms(iou_scores, dice_scores, tile_output_dir / "agreement_histograms.png")

    # Create blob records
    blob_records = create_blob_records(
        melanA_blobs, sox10_blobs, metadata, idx, tiles[idx].cpu().numpy()
    )
    
    return {
        'blobs': blob_records,
        'agreement': {'iou': iou_scores, 'dice': dice_scores},
        # 'visualizations': {
        #     'melanA_masks': Path(tile_output_dir, "melanA_masks.png").as_posix(),
        #     'sox10_masks': Path(tile_output_dir, "sox10_masks.png").as_posix(),
        #     'histograms': Path(tile_output_dir, "agreement_histograms.png").as_posix()
        # }
    }

def create_blob_records(melanA_blobs, sox10_blobs, metadata, idx, tile_img):
    records = []
    for stain, blobs in [('melanA', melanA_blobs), ('sox10', sox10_blobs)]:
        for i, blob in enumerate(blobs):
            area = cv2.contourArea(blob)
            x, y, w, h = cv2.boundingRect(blob)
            records.append({
                'tile_id': metadata['tile_id'][idx],
                'stain': stain,
                'blob_id': f"{metadata['tile_id'][idx]}_{stain}_{i}",
                'area': area,
                'centroid': (x + w/2, y + h/2),
                'bbox': (x, y, w, h),
                'coords': metadata['coords'][idx],
                'tile_mean_intensity': tile_img.mean()
            })
    return records

def save_enhanced_results(blob_records, agreement_metrics, visualization_paths, output_dir):
    """Save all results with visualization metadata"""
    blob_df = pd.DataFrame(blob_records)
    agreement_df = pd.DataFrame([{
        **metrics,
        'num_matches': len(metrics['iou'])
    } for metrics in agreement_metrics])
    
    # vis_df = pd.DataFrame(visualization_paths)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    blob_df.to_csv(Path(output_dir) / f"blob_records_{timestamp}.csv")
    agreement_df.to_csv(Path(output_dir) / f"agreement_metrics_{timestamp}.csv")
    # vis_df.to_csv(Path(output_dir) / f"visualization_paths_{timestamp}.csv")


# --------------------------
# Execution
# --------------------------

if __name__ == "__main__":
    # Configuration
    CONFIG_PATH = Path('saliency_config.yaml')
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    
    # Prepare normalization
    NORM_PATHS = {
        'melanA': Path(config['mean_std_path']['melanA']),
        'sox10': Path(config['mean_std_path']['sox10'])
    }
    
    #Define output directory
    output_dir = config['output_dir']

    # Initialize components
    melanA_model, sox10_model = load_trained_models(CONFIG_PATH)
    norm_transforms = load_normalization(NORM_PATHS)
    
    #Init the dataloaders, evaluating only the highest performing models/splits (5 and 5)
    melanA_test_dls, melanA_section_names = build_test_data_with_defined_section(config, "melanA")
    sox10_test_dls, sox10_section_names = build_test_data_with_defined_section(config, "sox10")

    #Concatenate them into two large lists
    test_dls, section_names = melanA_test_dls + sox10_test_dls, melanA_section_names + sox10_section_names
    
    # Run processing
    for test_dl, section_name in tqdm(list(zip(test_dls, section_names)), desc = 'Slide-level progress'):
        generate_saliency_maps(
            dataloader=test_dl,
            melanA_model=melanA_model,
            sox10_model=sox10_model,
            norm_transforms=norm_transforms,
            output_dir=output_dir,
            conf_thresholds=(0.9, 0.9),
            max_tiles=1000  # For stochastic sampling
    )
