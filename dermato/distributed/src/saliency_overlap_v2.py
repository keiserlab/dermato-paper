# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "torch>=2.0.0",
#     "numpy",
#     "opencv-python",
#     "pyyaml",
#     "pandas",
#     "tqdm",
#     "scikit-learn",
#     "torchvision",
#     "torchaudio",
#     "captum",
#     "zarr",
#     "pillow",
#     "kornia",
# ]
# ///

import warnings
warnings.filterwarnings("ignore", message="Setting backward hooks on ReLU activations")

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

# Additional imports for distributed processing
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

# --------------------------
# Dataset Definition
# --------------------------

class StainDataSet(Dataset):
    def __init__(self, df):
        self.df = df
        self.tile_size = 256
        self._compute_baseline()

    def _compute_baseline(self):
        num_positives = np.sum(self.df['label'])
        num_negatives = self.df.shape[0] - num_positives
        self.ratio_positives = round(num_positives / self.df.shape[0] * 100, 2)
        self.ratio_negatives = round(num_negatives / self.df.shape[0] * 100, 2)
        print(f'Baseline accuracy: {self.ratio_positives}% positives, {self.ratio_negatives}% negatives')

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        he_section_path = self.df.iloc[idx]['he']
        ihc_section_path = self.df.iloc[idx]['ihc']
        y, x = self.df.iloc[idx]['y'], self.df.iloc[idx]['x']
        label = self.df.iloc[idx]['label']

        if os.path.splitext(he_section_path)[1] == '.zarr':
            he_section = zarr.open(he_section_path, mode='r')
            he_tile = he_section[y:y+self.tile_size, x:x+self.tile_size]
            h, w, c = he_tile.shape
            if (h < self.tile_size) or (w < self.tile_size):
                he_tile = np.pad(he_tile, ((0, self.tile_size-h), (0, self.tile_size-w), (0,0)), 'constant', constant_values=(255.))
        elif os.path.splitext(he_section_path)[1] == '.jpg':
            he_tile = np.array(Image.open(he_section_path))

        he_tile = K.image_to_tensor(he_tile, keepdim=True).float() / 255.0
        metadata = defaultdict(dict)
        metadata['label'] = torch.tensor(label, dtype=torch.float)
        metadata['coords'] = torch.tensor((y, x), dtype=torch.int32)
        # Optionally, if you have a tile_id field, add it here.
        if 'tile_id' in self.df.columns:
            metadata['tile_id'] = self.df.iloc[idx]['tile_id']
        else:
            metadata['tile_id'] = idx
        return he_tile, metadata

# --------------------------
# Model Initialization and Loading
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
    
    return melanA_model.eval(), sox10_model.eval()

# --------------------------
# Data Transformation
# --------------------------

def load_normalization(norm_paths):
    transforms = {}
    for stain, path in norm_paths.items():
        with open(path, 'rb') as f:
            mean, std = pickle.load(f)
        transforms[stain] = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean, std)
        ])
    return transforms

# --------------------------
# Saliency Generation Core
# --------------------------

def generate_attr_gradcam(model, input_tensor, tile_idx, stain):
    layers = [name for name, module in model.named_modules() 
              if 'conv' in name and 'dense' in name]
    attributions = torch.zeros_like(input_tensor)
    for layer in tqdm(layers, desc=f"Layers for (tile idx {tile_idx}, {stain}) [{model.device}]", position=model.device.index, leave=True):
        guided_gc = GuidedGradCam(model, eval(f'model.{layer}'))
        noise_tunnel = NoiseTunnel(guided_gc)
        attr = noise_tunnel.attribute(input_tensor, nt_samples=5, nt_type='smoothgrad')
        attributions += attr
    attributions /= len(layers)
    attr = np.transpose(attributions.squeeze(0).cpu().detach().numpy(), (1,2,0))
    return attr

# --------------------------
# Blob Processing Pipeline
# --------------------------

def preprocess_saliency_map(saliency_map):
    saliency_normed = viz._normalize_attr(saliency_map, 'absolute_value', outlier_perc=2.0, reduction_axis=2)
    saliency_uint8 = (saliency_normed * 255).astype(np.uint8)
    _, thresh = cv2.threshold(saliency_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def apply_morphological_operations(binary_map, kernel_size=5, num_iters=2):
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
# Data Loading
# --------------------------

def build_test_data_with_defined_section(saliency_config, stain):
    batch_size = saliency_config['batch_size']
    num_workers = saliency_config['num_workers']
    df_path = saliency_config['dataframe_path'][stain]
    list_test_dls = []
    df = pd.read_csv(df_path)
    all_sections = list(df['section_name'].unique())
    for section in all_sections:
        df_sec = df.loc[df['section_name'] == section].reset_index(drop=True)
        print(f'{section}: {len(df_sec)} rows')
        test_ds = StainDataSet(df=df_sec)
        test_dl = DataLoader(
            test_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False
        )
        list_test_dls.append(test_dl)
    return list_test_dls, all_sections

# --------------------------
# Main Saliency Map Generation Function
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
        for batch_idx, (tiles, metadata) in enumerate(tqdm(dataloader, desc='Batch (qualified tiles)', leave=False)):
            if max_tiles and batch_idx >= max_tiles:
                break

            melanA_input = norm_transforms['melanA'](tiles).to(device_melanA)
            sox10_input = norm_transforms['sox10'](tiles).to(device_sox10)
            
            melanA_preds = torch.sigmoid(melanA_model(melanA_input))
            sox10_preds = torch.sigmoid(sox10_model(sox10_input))
            
            qualified = (melanA_preds.cpu() > conf_thresholds[0]) & (sox10_preds.cpu() > conf_thresholds[1])
            
            for idx in torch.where(qualified)[0]:
                tile_data = process_qualified_tile(
                    idx, tiles, metadata,
                    melanA_model, sox10_model,
                    melanA_input[idx], sox10_input[idx],
                    min_blob_area, morph_kernel, morph_iters,
                    output_dir, debug=False
                )
                blob_records.extend(tile_data['blobs'])
                agreement_metrics.append(tile_data['agreement'])
                save_enhanced_results(blob_records, agreement_metrics, visualization_paths, output_dir)
            if batch_idx % 1 == 0:
                print("batch_idx", batch_idx)
                save_enhanced_results(blob_records, agreement_metrics, visualization_paths, output_dir)
    
    save_enhanced_results(blob_records, agreement_metrics, visualization_paths, output_dir)
    return blob_records, agreement_metrics, visualization_paths

def process_qualified_tile(idx, tiles, metadata, melanA_model, sox10_model,
                           melanA_input, sox10_input, min_area, kernel_size, num_iters,
                           output_dir, debug=False):
    
    melanA_attr = generate_attr_gradcam(melanA_model, melanA_input.unsqueeze(0), idx, 'melanA')
    sox10_attr = generate_attr_gradcam(sox10_model, sox10_input.unsqueeze(0), idx, 'sox10')

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

    melanA_blobs = detect_blobs(processed['melanA'], min_area)
    sox10_blobs = detect_blobs(processed['sox10'], min_area)
    
    melanA_masks = create_blob_masks(melanA_blobs, processed['melanA'].shape)
    sox10_masks = create_blob_masks(sox10_blobs, processed['sox10'].shape)
    
    iou_scores, dice_scores = calculate_agreement(melanA_masks, sox10_masks)

    blob_records = create_blob_records(melanA_blobs, sox10_blobs, metadata, idx, tiles[idx].cpu().numpy())
    
    return {
        'blobs': blob_records,
        'agreement': {'iou': iou_scores, 'dice': dice_scores},
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
    blob_df = pd.DataFrame(blob_records)
    agreement_df = pd.DataFrame([{
        **metrics,
        'num_matches': len(metrics['iou'])
    } for metrics in agreement_metrics])
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    blob_df.to_csv(Path(output_dir) / f"blob_records_{timestamp}.csv")
    agreement_df.to_csv(Path(output_dir) / f"agreement_metrics_{timestamp}.csv")

# --------------------------
# Distributed Main Worker Function
# --------------------------

def main_worker(local_rank, config):
    # Set the current GPU device for this process
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Initialize the process group for distributed training
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"Process group: rank {rank} out of {world_size} on device {device}")

    # Load models
    melanA_model, sox10_model = load_trained_models(config['_file_path'])
    melanA_model = melanA_model.to(device)
    sox10_model = sox10_model.to(device)

    # Wrap models with DistributedDataParallel
    from torch.nn.parallel import DistributedDataParallel as DDP
    melanA_model = DDP(melanA_model, device_ids=[local_rank], output_device=local_rank)
    sox10_model = DDP(sox10_model, device_ids=[local_rank], output_device=local_rank)

    # Load normalization transforms
    norm_transforms = load_normalization({
        'melanA': config['mean_std_path']['melanA'],
        'sox10': config['mean_std_path']['sox10']
    })

    # Build test dataloaders for both stains and merge them
    melanA_test_dls, melanA_section_names = build_test_data_with_defined_section(config, "melanA")
    sox10_test_dls, sox10_section_names = build_test_data_with_defined_section(config, "sox10")
    test_dls = melanA_test_dls + sox10_test_dls

    # Update each DataLoader to use a DistributedSampler
    new_test_dls = []
    for test_dl in test_dls:
        sampler = DistributedSampler(test_dl.dataset, num_replicas=world_size, rank=rank)
        new_test_dl = DataLoader(
            test_dl.dataset,
            batch_size=test_dl.batch_size,
            num_workers=test_dl.num_workers,
            pin_memory=True,
            sampler=sampler,
            drop_last=test_dl.drop_last if hasattr(test_dl, 'drop_last') else False,
            shuffle=False
        )
        new_test_dls.append(new_test_dl)

    # Process each DataLoader on this worker
    for test_dl in new_test_dls:
        generate_saliency_maps(
            dataloader=test_dl,
            melanA_model=melanA_model,
            sox10_model=sox10_model,
            norm_transforms=norm_transforms,
            output_dir=config['output_dir'],
            conf_thresholds=(0.9, 0.9),
            max_tiles=1000
        )

    dist.destroy_process_group()

# --------------------------
# Entry Point
# --------------------------

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=int(os.environ["LOCAL_RANK"]))
    args = parser.parse_args()

    # Load configuration from YAML
    CONFIG_PATH = Path('saliency_config.yaml')
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
        config['_file_path'] = CONFIG_PATH

    # Call the main_worker with the local_rank provided by torchrun
    main_worker(args.local_rank, config)