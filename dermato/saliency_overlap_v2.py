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
#     "captum",
#     "zarr",
#     "pillow",
#     "kornia",
# ]
# ///

# WORK IN PROGRESS
# - based on saliency_overlap.py
# - various features (not all tested/draft) incl smarter resumability
# - CUDA OOM (memory) fallbacks are probably overcomplicated and ok to remove
# 
# HOW TO RUN
# 1. install uv if needed (https://docs.astral.sh/uv/getting-started/installation/)
# 2. uv run <scriptname>
# fyi - I recommend batch size = 1280 on the 3090's

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
import concurrent.futures
import argparse
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Set

class StainDataSet(Dataset):
    """
    Dataset class for stain images
    
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
        Get an item from the dataset
        
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
            he_section = zarr.open(he_section_path, mode='r')  #[C,H,W] #
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
        
        # Generate a unique tile_id
        tile_id = f"{os.path.basename(he_section_path)}_{y}_{x}"
        
        metadata = defaultdict(dict)
        metadata['label'] = torch.tensor(label, dtype=torch.float)
        metadata['coords'] = torch.tensor((y, x), dtype=torch.int32)
        metadata['tile_id'] = tile_id

        return he_tile, metadata

# --------------------------
# Model Initialization
# --------------------------

def check_cuda_availability(required_devices=(0, 1)):
    """Check if required CUDA devices are available"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but required for this script")
    
    available_devices = torch.cuda.device_count()
    if available_devices <= max(required_devices):
        raise RuntimeError(f"This script requires at least {max(required_devices) + 1} CUDA devices, "
                          f"but only {available_devices} are available")
    
    # Print out GPU memory information
    for i in range(available_devices):
        free_mem, total_mem = torch.cuda.mem_get_info(i)
        free_mb = free_mem / (1024 * 1024)
        total_mb = total_mem / (1024 * 1024)
        print(f"GPU {i}: {free_mb:.2f}MB free / {total_mb:.2f}MB total")
    
    # Enable memory efficient options
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    print(f"CUDA is available with {available_devices} devices")
    return True

def remove_module_prefix(state_dict):
    """Remove 'module.' prefix from state_dict keys"""
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_state_dict[key.replace('module.', '')] = value
    return new_state_dict

def initialize_model():
    """Initialize DenseNet121 with a binary classification head"""
    model = densenet121()
    model.classifier = nn.Linear(1024, 1)
    return model

def load_trained_models(config_path):
    """Load pre-trained models from config"""
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
    """Load normalization parameters for each stain"""
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

def generate_attr_gradcam(model, input_tensor):
    """Generate GradCAM attributions for a model and input"""
    try:
        layers = [name for name, module in model.named_modules() 
                if 'conv' in name and 'dense' in name]
        
        attributions = torch.zeros_like(input_tensor)
        for layer in layers: #tqdm(layers, desc="Tile"):
            ## Clear cache between layers to reduce memory pressure
            #torch.cuda.empty_cache()
            
            guided_gc = GuidedGradCam(model, eval(f'model.{layer}'))
            noise_tunnel = NoiseTunnel(guided_gc)
            
            # Use original settings for scientific accuracy
            attr = noise_tunnel.attribute(input_tensor, 
                                       nt_samples=5,
                                       nt_type='smoothgrad')
            attributions += attr
            
            # Explicitly delete to help with memory management
            del guided_gc, noise_tunnel, attr
        
        # Normalize by number of layers
        attributions /= len(layers)
        # Convert to image shape (H,W,C)
        attr = np.transpose(attributions.squeeze(0).cpu().detach().numpy(), (1,2,0))
        
        return attr
        
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            # Clear cache and try again with CPU fallback
            torch.cuda.empty_cache()
            print(f"CUDA OOM detected, falling back to CPU for this tile")
            
            # Move model and input to CPU temporarily
            device = next(model.parameters()).device
            cpu_model = model.cpu()
            cpu_input = input_tensor.cpu()
            
            # Process on CPU
            layers = [name for name, module in cpu_model.named_modules() 
                    if 'conv' in name and 'dense' in name]
            
            attributions = torch.zeros_like(cpu_input)
            for layer in layers:
                guided_gc = GuidedGradCam(cpu_model, eval(f'cpu_model.{layer}'))
                noise_tunnel = NoiseTunnel(guided_gc)
                attr = noise_tunnel.attribute(cpu_input, 
                                           nt_samples=5,  # Use original sample count for scientific accuracy
                                           nt_type='smoothgrad')
                attributions += attr
                del guided_gc, noise_tunnel, attr
            
            # Normalize by number of layers
            attributions /= len(layers)
            # Convert to image shape (H,W,C)
            attr = np.transpose(attributions.squeeze(0).detach().numpy(), (1,2,0))
            
            # Move model back to original device
            model.to(device)
            
            return attr
        else:
            # Re-raise the exception if it's not a CUDA OOM error
            raise

# --------------------------
# Blob Processing Pipeline
# --------------------------

def preprocess_saliency_map(saliency_map):
    """Normalize and threshold saliency map"""
    # Squeeze dimensions to 1 and normalize using captum's normalize function
    saliency_normed = viz._normalize_attr(saliency_map, 'absolute_value', outlier_perc=2.0, reduction_axis=2)
    # Threshold the 1-dim attrs
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
    """Detect blobs in processed binary map"""
    contours, _ = cv2.findContours(processed_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= min_area]

def create_blob_masks(contours, shape):
    """Create binary masks for each blob contour"""
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
    """Calculate IoU and Dice scores between masks"""
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
# Visualization
# --------------------------

def visualize_masks(saliency_map, masks, output_path):
    """Visualize saliency map and detected blobs"""
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Saliency map
    saliency_rgb = viz._normalize_attr(saliency_map, 'absolute_value', outlier_perc=2.0, reduction_axis=2)
    ax[0].imshow(saliency_rgb)
    ax[0].set_title('Saliency Map')
    ax[0].axis('off')
    
    # Blob masks
    mask_overlay = np.zeros((*saliency_map.shape[:2], 3), dtype=np.uint8)
    for i, mask in enumerate(masks):
        color = np.random.randint(0, 255, 3)
        for c in range(3):
            mask_overlay[:, :, c] = np.where(mask > 0, color[c], mask_overlay[:, :, c])
    
    ax[1].imshow(mask_overlay)
    ax[1].set_title(f'{len(masks)} Detected Blobs')
    ax[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_histograms(iou_scores, dice_scores, output_path):
    """Plot histograms of IoU and Dice scores"""
    if not iou_scores or not dice_scores:
        # Skip if there are no scores
        return
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    ax[0].hist(iou_scores, bins=20, alpha=0.7)
    ax[0].set_title('IoU Scores')
    ax[0].set_xlabel('IoU')
    ax[0].set_ylabel('Frequency')
    
    ax[1].hist(dice_scores, bins=20, alpha=0.7)
    ax[1].set_title('Dice Scores')
    ax[1].set_xlabel('Dice')
    ax[1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# --------------------------
# Data loading
# --------------------------

def build_test_data_with_defined_section(saliency_config, stain):
    """
    Build test data with defined section.
    Args:
        saliency_config (dict): The saliency configuration.
        stain (str): The stain type (melanA or sox10).
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
            shuffle=False,
            pin_memory=True, 
            drop_last=False
        )
        list_test_dls.append(test_dl)

    return list_test_dls, all_sections

# --------------------------
# Resumability
# --------------------------

def load_processed_tiles(output_dir):
    """Load set of previously processed tile IDs"""
    processed_tiles_file = Path(output_dir) / "processed_tiles.json"
    if processed_tiles_file.exists():
        with open(processed_tiles_file, 'r') as f:
            return set(json.load(f))
    return set()

def save_processed_tiles(output_dir, processed_tiles):
    """Save set of processed tile IDs"""
    processed_tiles_file = Path(output_dir) / "processed_tiles.json"
    with open(processed_tiles_file, 'w') as f:
        json.dump(list(processed_tiles), f)

def load_progress_metadata(output_dir):
    """Load progress metadata"""
    progress_file = Path(output_dir) / "progress_metadata.json"
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {"sections_completed": [], "last_update": None}

def save_progress_metadata(output_dir, progress_data):
    """Save progress metadata"""
    progress_file = Path(output_dir) / "progress_metadata.json"
    progress_data["last_update"] = datetime.now().isoformat()
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f)

# --------------------------
# Parallel Processing
# --------------------------

def process_saliency_batch(batch_data, melanA_model, sox10_model, norm_transforms, 
                          conf_thresholds, min_blob_area, morph_kernel, morph_iters, 
                          enable_visualization=False, max_workers=None):
    """Process a batch of tiles for saliency maps"""
    tiles, metadata = batch_data
    
    # Get device info
    device_melanA = next(melanA_model.parameters()).device
    device_sox10 = next(sox10_model.parameters()).device
    
    # Forward pass through models - truly parallel across both GPUs
    try:
        with torch.no_grad():
            # Process both models in parallel using separate CUDA streams
            melanA_stream = torch.cuda.Stream(device=device_melanA)
            sox10_stream = torch.cuda.Stream(device=device_sox10)
            
            # First prepare the inputs on respective devices
            melanA_input = norm_transforms['melanA'](tiles).to(device_melanA)
            sox10_input = norm_transforms['sox10'](tiles).to(device_sox10)
            
            # Run models in parallel on separate streams
            melanA_preds_cuda = None
            sox10_preds_cuda = None
            
            with torch.cuda.stream(melanA_stream):
                melanA_preds_cuda = torch.sigmoid(melanA_model(melanA_input))
                
            with torch.cuda.stream(sox10_stream):
                sox10_preds_cuda = torch.sigmoid(sox10_model(sox10_input))
                
            # Synchronize both streams to ensure both predictions are done
            torch.cuda.synchronize(device_melanA)
            torch.cuda.synchronize(device_sox10)
            
            # Transfer results to CPU for further processing
            melanA_preds_cpu = melanA_preds_cuda.cpu()
            sox10_preds_cpu = sox10_preds_cuda.cpu()
            
            # Find qualified tiles (above confidence threshold)
            qualified = (melanA_preds_cpu > conf_thresholds[0]) & (sox10_preds_cpu > conf_thresholds[1])
            qualified_indices = torch.where(qualified)[0].tolist()
            
            # Print some stats about qualified tiles
            #print(f"Found {len(qualified_indices)}/{len(tiles)} qualified tiles above threshold\n")
            
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"CUDA OOM during forward pass - trying smaller batch")
            # Clear cache
            torch.cuda.empty_cache()
            
            # Try to process in two halves
            if len(tiles) > 1:
                mid = len(tiles) // 2
                first_half = process_saliency_batch(
                    (tiles[:mid], {k: v[:mid] for k, v in metadata.items()}),
                    melanA_model, sox10_model, norm_transforms,
                    conf_thresholds, min_blob_area, morph_kernel, morph_iters,
                    enable_visualization, max_workers
                )
                
                # Clear cache again before second half
                torch.cuda.empty_cache()
                
                second_half = process_saliency_batch(
                    (tiles[mid:], {k: v[mid:] for k, v in metadata.items()}),
                    melanA_model, sox10_model, norm_transforms,
                    conf_thresholds, min_blob_area, morph_kernel, morph_iters,
                    enable_visualization, max_workers
                )
                
                return first_half + second_half
            else:
                # Can't reduce batch size further
                print(f"WARNING: Cannot reduce batch size further, skipping tile")
                return []
        else:
            # Re-raise if not OOM
            raise
    
    results = []
    
    # Process tile attributions in parallel when possible
    if len(qualified_indices) > 0:
        # For tiles that need both models, process in parallel using both GPUs
        # Use torch.multiprocessing for true GPU parallelism
        import torch.multiprocessing as mp
        
        # Adjust number of workers based on system capabilities and batch size
        if max_workers is None:
            # Use fewer workers for larger qualified batches to reduce memory pressure
            max_workers = max(1, min(4, 16 // (len(qualified_indices) + 1)))
        
        # For tiles that require both models, set up a parallel processing system
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_tasks = []
            
            for idx in qualified_indices:
                # Submit qualified tile processing tasks to thread pool
                future = executor.submit(
                    process_qualified_tile_parallel,
                    idx, tiles, metadata,
                    melanA_model, sox10_model,
                    melanA_input[idx], sox10_input[idx],
                    device_melanA, device_sox10,
                    min_blob_area, morph_kernel, morph_iters,
                    None, enable_visualization
                )
                future_tasks.append((future, idx))
                
                # Sleep briefly to allow for better error propagation
                import time
                time.sleep(0.001)
            
            # Process results as they complete
            for future, idx in tqdm(future_tasks, desc="Batch (qualified tiles)"):
                try:
                    tile_data = future.result()
                    tile_data['tile_id'] = metadata['tile_id'][idx]
                    results.append(tile_data)
                    
                    # Periodically clear cache
                    if len(results) % 5 == 0:
                        torch.cuda.empty_cache()
                        
                except Exception as exc:
                    print(f'Processing of tile {metadata["tile_id"][idx]} generated an exception: {exc}')
                    if "CUDA out of memory" in str(exc):
                        # Force cache clear on OOM
                        torch.cuda.empty_cache()
    
    # Final cache clear
    torch.cuda.empty_cache()
    return results

def process_qualified_tile_parallel(idx, tiles, metadata, melanA_model, sox10_model,
                                   melanA_input, sox10_input, device_melanA, device_sox10,
                                   min_area, kernel_size, num_iters,
                                   output_dir=None, enable_visualization=False):
    """Process a qualified tile with true parallel GPU execution"""
    
    # Run both saliency generations in parallel on their respective devices using streams
    melanA_stream = torch.cuda.Stream(device=device_melanA)
    sox10_stream = torch.cuda.Stream(device=device_sox10)
    
    melanA_attr = None
    sox10_attr = None
    
    # Start processing both in parallel
    melanA_future = concurrent.futures.ThreadPoolExecutor(max_workers=1).submit(
        generate_attr_gradcam, melanA_model, melanA_input.unsqueeze(0)
    )
    
    sox10_future = concurrent.futures.ThreadPoolExecutor(max_workers=1).submit(
        generate_attr_gradcam, sox10_model, sox10_input.unsqueeze(0)
    )
    
    # Wait for both to complete
    try:
        melanA_attr = melanA_future.result()
        sox10_attr = sox10_future.result()
    except Exception as e:
        print(f"Error in parallel saliency generation: {e}")
        raise

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
    
    # Calculate agreement metrics
    iou_scores, dice_scores = calculate_agreement(melanA_masks, sox10_masks)

    # Save visualizations if enabled
    visualization_paths = {}
    if enable_visualization and output_dir:
        tile_id = metadata['tile_id'][idx]
        tile_output_dir = Path(output_dir) / f"vis_tile_{tile_id}"
        tile_output_dir.mkdir(exist_ok=True, parents=True)
        
        melanA_vis_path = tile_output_dir / "melanA_masks.png"
        sox10_vis_path = tile_output_dir / "sox10_masks.png"
        histograms_path = tile_output_dir / "agreement_histograms.png"
        
        visualize_masks(melanA_attr, melanA_masks, melanA_vis_path)
        visualize_masks(sox10_attr, sox10_masks, sox10_vis_path)
        plot_histograms(iou_scores, dice_scores, histograms_path)
        
        visualization_paths = {
            'melanA_masks': str(melanA_vis_path),
            'sox10_masks': str(sox10_vis_path),
            'histograms': str(histograms_path)
        }

    # Create blob records
    blob_records = []
    for stain, blobs in [('melanA', melanA_blobs), ('sox10', sox10_blobs)]:
        for i, blob in enumerate(blobs):
            area = cv2.contourArea(blob)
            x, y, w, h = cv2.boundingRect(blob)
            blob_records.append({
                'tile_id': metadata['tile_id'][idx],
                'stain': stain,
                'blob_id': f"{metadata['tile_id'][idx]}_{stain}_{i}",
                'area': area,
                'centroid': (x + w/2, y + h/2),
                'bbox': (x, y, w, h),
                'coords': metadata['coords'][idx].tolist(),
                'tile_mean_intensity': float(tiles[idx].cpu().mean())
            })
    
    return {
        'blobs': blob_records,
        'agreement': {'iou': iou_scores, 'dice': dice_scores},
        'visualizations': visualization_paths
    }

# --------------------------
# Main Processing Function
# --------------------------

def generate_saliency_maps(dataloader, melanA_model, sox10_model, norm_transforms,
                          output_dir, conf_thresholds=(0.9, 0.9),
                          min_blob_area=100, morph_kernel=5, morph_iters=2, 
                          max_tiles=None, enable_visualization=False):
    """Generate saliency maps for a dataset"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load previously processed tiles and progress metadata
    processed_tiles = load_processed_tiles(output_dir)
    progress_metadata = load_progress_metadata(output_dir)
    
    blob_records = []
    agreement_metrics = []
    visualization_paths = []
    
    # Setup for batch processing
    batch_processor = BatchProcessor(
        melanA_model, sox10_model, norm_transforms,
        conf_thresholds, min_blob_area, morph_kernel, morph_iters,
        enable_visualization, output_dir
    )
    
    # Get section name and total tiles for progress reporting
    section_name = dataloader.dataset.df['section_name'].iloc[0] if 'section_name' in dataloader.dataset.df.columns else "unknown"
    total_tiles = len(dataloader.dataset)
    processed_count = 0
    qualified_count = 0
    
    # Process in batches with enhanced progress tracking
    batch_progress = tqdm(
        dataloader, 
        desc=f"Section: {section_name} ({processed_count}/{total_tiles} tiles, {qualified_count} qualified)",
        unit="batch"
    )
    
    for batch_idx, batch_data in enumerate(batch_progress):
        if max_tiles and batch_idx >= max_tiles:
            break
        
        # Skip already processed tiles
        tiles, metadata = batch_data
        tile_ids = metadata['tile_id']
        
        # Filter out already processed tiles
        indices_to_process = [i for i, tid in enumerate(tile_ids) if tid not in processed_tiles]
        if not indices_to_process:
            continue
            
        # Process only new tiles
        tiles_to_process = tiles[indices_to_process]
        metadata_to_process = {k: [v[i] for i in indices_to_process] for k, v in metadata.items()}
        
        # Process batch and collect results
        batch_results = batch_processor.process_batch(
            (tiles_to_process, metadata_to_process)
        )
        
        # Update records
        for result in batch_results:
            blob_records.extend(result['blobs'])
            agreement_metrics.append(result['agreement'])
            if 'visualizations' in result and result['visualizations']:
                visualization_paths.append(result['visualizations'])
            
            # Mark as processed
            processed_tiles.add(result['tile_id'])
            qualified_count += 1
        
        # Update processed count
        processed_count += len(indices_to_process)
        
        # Update progress bar description
        batch_progress.set_description(
            f"Section: {section_name} ({processed_count}/{total_tiles} tiles, {qualified_count} qualified)"
        )
        
        # Save incremental results
        if batch_idx % 1 == 0 or not batch_results:
            save_results(blob_records, agreement_metrics, visualization_paths, output_dir)
            save_processed_tiles(output_dir, processed_tiles)
            
    # Final save
    save_results(blob_records, agreement_metrics, visualization_paths, output_dir)
    save_processed_tiles(output_dir, processed_tiles)
    
    # Update progress metadata
    if dataloader.dataset.df['section_name'].nunique() == 1:
        section_name = dataloader.dataset.df['section_name'].iloc[0]
        if section_name not in progress_metadata['sections_completed']:
            progress_metadata['sections_completed'].append(section_name)
    
    save_progress_metadata(output_dir, progress_metadata)
    
    return blob_records, agreement_metrics, visualization_paths

class BatchProcessor:
    """Processor for batches of tiles with saliency mapping"""
    
    def __init__(self, melanA_model, sox10_model, norm_transforms,
                conf_thresholds, min_blob_area, morph_kernel, morph_iters,
                enable_visualization, output_dir):
        self.melanA_model = melanA_model
        self.sox10_model = sox10_model
        self.norm_transforms = norm_transforms
        self.conf_thresholds = conf_thresholds
        self.min_blob_area = min_blob_area
        self.morph_kernel = morph_kernel
        self.morph_iters = morph_iters
        self.enable_visualization = enable_visualization
        self.output_dir = output_dir
        self.memory_issues_count = 0
        self.batch_size_reduction = 1
        
        # Set memory management parameters
        self.max_memory_issues = 3  # After this many OOM errors, we'll start reducing batch size
        
        # Ensure models are in eval mode
        self.melanA_model.eval()
        self.sox10_model.eval()
        
    def check_memory_status(self):
        """Check GPU memory status and clear cache if needed"""
        # Clear CUDA cache 
        torch.cuda.empty_cache()
        
        # Get memory info for both GPUs
        free_mem0, total_mem0 = torch.cuda.mem_get_info(0)
        free_mem1, total_mem1 = torch.cuda.mem_get_info(1)
        
        free_pct0 = (free_mem0 / total_mem0) * 100
        free_pct1 = (free_mem1 / total_mem1) * 100
        
        # If either GPU is critically low on memory (less than 5% free)
        if free_pct0 < 5 or free_pct1 < 5:
            print(f"WARNING: Low GPU memory detected. GPU 0: {free_pct0:.1f}% free, GPU 1: {free_pct1:.1f}% free")
            self.memory_issues_count += 1
            
            # If we've had multiple memory issues, reduce effective batch size
            if self.memory_issues_count >= self.max_memory_issues:
                self.batch_size_reduction *= 2
                print(f"Reducing effective batch size by factor of {self.batch_size_reduction} due to memory constraints")
                self.memory_issues_count = 0  # Reset counter
                
            # More aggressive cache clearing
            torch.cuda.empty_cache()
            return False
        return True
        
    def process_batch(self, batch_data):
        """Process a batch of tiles with memory management"""
        # Check memory status and clear cache if needed
        self.check_memory_status()
        
        # If batch size reduction is in effect, split the batch
        if self.batch_size_reduction > 1:
            tiles, metadata = batch_data
            batch_size = tiles.size(0)
            sub_batch_size = max(1, batch_size // self.batch_size_reduction)
            
            # Process in smaller chunks
            all_results = []
            for i in range(0, batch_size, sub_batch_size):
                end_idx = min(i + sub_batch_size, batch_size)
                sub_tiles = tiles[i:end_idx]
                sub_metadata = {k: v[i:end_idx] for k, v in metadata.items()}
                
                # Process the sub-batch
                sub_results = process_saliency_batch(
                    (sub_tiles, sub_metadata),
                    self.melanA_model, self.sox10_model,
                    self.norm_transforms, self.conf_thresholds,
                    self.min_blob_area, self.morph_kernel, self.morph_iters,
                    self.enable_visualization
                )
                all_results.extend(sub_results)
                
                # Clear cache between sub-batches
                torch.cuda.empty_cache()
            
            return all_results
        else:
            # Normal processing if no memory issues
            return process_saliency_batch(
                batch_data,
                self.melanA_model, self.sox10_model,
                self.norm_transforms, self.conf_thresholds,
                self.min_blob_area, self.morph_kernel, self.morph_iters,
                self.enable_visualization
            )

def save_results(blob_records, agreement_metrics, visualization_paths, output_dir):
    """Save results with timestamp"""
    if not blob_records:
        return
        
    blob_df = pd.DataFrame(blob_records)
    
    agreement_df = pd.DataFrame([
        {
            'iou_mean': np.mean(metrics['iou']) if metrics['iou'] else np.nan,
            'iou_std': np.std(metrics['iou']) if metrics['iou'] else np.nan,
            'dice_mean': np.mean(metrics['dice']) if metrics['dice'] else np.nan,
            'dice_std': np.std(metrics['dice']) if metrics['dice'] else np.nan,
            'num_matches': len(metrics['iou'])
        }
        for metrics in agreement_metrics
    ])
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save the full data
    blob_df.to_csv(Path(output_dir) / f"blob_records_{timestamp}.csv", index=False)
    agreement_df.to_csv(Path(output_dir) / f"agreement_metrics_{timestamp}.csv", index=False)
    
    # Also save to a cumulative file
    cumulative_blob_path = Path(output_dir) / "cumulative_blob_records.csv"
    cumulative_agreement_path = Path(output_dir) / "cumulative_agreement_metrics.csv"
    
    # If files exist, append without header, otherwise create with header
    if cumulative_blob_path.exists():
        blob_df.to_csv(cumulative_blob_path, mode='a', header=False, index=False)
    else:
        blob_df.to_csv(cumulative_blob_path, index=False)
        
    if cumulative_agreement_path.exists():
        agreement_df.to_csv(cumulative_agreement_path, mode='a', header=False, index=False)
    else:
        agreement_df.to_csv(cumulative_agreement_path, index=False)
    
    # Save visualization paths if any
    if visualization_paths:
        vis_df = pd.DataFrame(visualization_paths)
        vis_df.to_csv(Path(output_dir) / f"visualization_paths_{timestamp}.csv", index=False)

# --------------------------
# Main Execution
# --------------------------

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate saliency maps for melanA and sox10 models')
    parser.add_argument('--config', type=str, default='saliency_config.yaml', 
                        help='Path to the configuration file')
    parser.add_argument('--output-dir', type=str, help='Output directory (overrides config)')
    parser.add_argument('--max-tiles', type=int, default=None, 
                        help='Maximum number of tiles to process')
    parser.add_argument('--enable-visualization', action='store_true', 
                        help='Enable visualization of saliency maps and blobs')
    parser.add_argument('--conf-thresholds', type=float, nargs=2, default=[0.9, 0.9],
                        help='Confidence thresholds for melanA and sox10 models')
    parser.add_argument('--batch-size', type=int, help='Override batch size from config')
    parser.add_argument('--memory-efficient', action='store_true',
                        help='Enable memory-efficient processing with reduced batch sizes')
    parser.add_argument('--cpu-fallback', action='store_true',
                        help='Allow falling back to CPU if GPU runs out of memory')
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    CONFIG_PATH = Path(args.config)
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    
    # Set PyTorch memory management environment variables
    if args.memory_efficient:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
        
    # Disable PyTorch compilation and related features that cause issues
    os.environ["PYTORCH_INDUCTOR_DISABLE"] = "1"
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    
    # Explicitly disable Triton backend which appears to be causing issues
    os.environ["TORCH_COMPILE_DISABLE_CUDA_TRITON"] = "1"
    
    # Check CUDA availability
    check_cuda_availability(required_devices=(0, 1))
    
    # Output directory
    output_dir = args.output_dir if args.output_dir else config['output_dir']
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    # Override batch size if specified
    if args.batch_size:
        config['batch_size'] = args.batch_size
        print(f"Overriding batch size to {args.batch_size}")
    elif args.memory_efficient:
        # Reduce batch size for memory efficiency
        original_batch_size = config['batch_size']
        config['batch_size'] = max(1, original_batch_size // 4)
        print(f"Memory-efficient mode: Reducing batch size from {original_batch_size} to {config['batch_size']}")
    
    # Log configuration
    print(f"Using config from: {CONFIG_PATH}")
    print(f"Output directory: {output_dir}")
    print(f"Enable visualization: {args.enable_visualization}")
    print(f"Confidence thresholds: {args.conf_thresholds}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Memory-efficient mode: {args.memory_efficient}")
    print(f"CPU fallback enabled: {args.cpu_fallback}")
    
    # Prepare normalization
    NORM_PATHS = {
        'melanA': Path(config['mean_std_path']['melanA']),
        'sox10': Path(config['mean_std_path']['sox10'])
    }
    
    # Initialize components with error handling
    print("Loading models...")
    try:
        melanA_model, sox10_model = load_trained_models(CONFIG_PATH)
        norm_transforms = load_normalization(NORM_PATHS)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("CUDA out of memory during model loading. Try reducing batch size or enable memory-efficient mode.")
            import sys
            sys.exit(1)
        else:
            raise
    
    # Set models to eval mode and optimize for inference
    melanA_model.eval()
    sox10_model.eval()
    
    # Completely disable torch.compile - using eager mode for better compatibility
    print("Using models in standard eager mode (torch.compile disabled)")
    
    # Disable PyTorch inductor and related backends that might cause issues
    os.environ["PYTORCH_INDUCTOR_DISABLE"] = "1"
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    
    # Set optimization flags for standard PyTorch execution
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # Load progress metadata
    progress_metadata = load_progress_metadata(output_dir)
    completed_sections = set(progress_metadata['sections_completed'])
    
    # Initialize dataloaders
    print("Building test dataloaders...")
    melanA_test_dls, melanA_section_names = build_test_data_with_defined_section(config, "melanA")
    sox10_test_dls, sox10_section_names = build_test_data_with_defined_section(config, "sox10")

    # Combine dataloaders and section names
    test_dls, section_names = melanA_test_dls + sox10_test_dls, melanA_section_names + sox10_section_names
    
    # Filter out already completed sections
    to_process = [(dl, name) for dl, name in zip(test_dls, section_names) 
                 if name not in completed_sections]
    
    if not to_process:
        print("All sections have been processed!")
    else:
        print(f"Processing {len(to_process)} remaining sections...")
    
    # Setup signal handling for graceful shutdown
    import signal
    
    def signal_handler(sig, frame):
        print('Caught signal, finishing current batch and exiting...')
        # Save any in-progress work here
        print("Saving progress metadata before exit...")
        save_progress_metadata(output_dir, progress_metadata)
        import sys
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run processing with memory monitoring
    for test_dl, section_name in tqdm(to_process, desc=f"Sections"):
        #print(f"\nProcessing section: {section_name}\n")
        section_output_dir = Path(output_dir) / section_name
        section_output_dir.mkdir(exist_ok=True, parents=True)
        
        try:
            # Clear CUDA cache before each section
            torch.cuda.empty_cache()
            
            generate_saliency_maps(
                dataloader=test_dl,
                melanA_model=melanA_model,
                sox10_model=sox10_model,
                norm_transforms=norm_transforms,
                output_dir=section_output_dir,
                conf_thresholds=tuple(args.conf_thresholds),
                min_blob_area=50,
                max_tiles=args.max_tiles,
                enable_visualization=args.enable_visualization
            )
            
            # Update progress
            if section_name not in progress_metadata['sections_completed']:
                progress_metadata['sections_completed'].append(section_name)
                save_progress_metadata(output_dir, progress_metadata)
                
            #print(f"Completed section: {section_name}")
            
        except Exception as e:
            print(f"Error processing section {section_name}: {e}")
            if "CUDA out of memory" in str(e):
                print("CUDA out of memory - trying to recover...")
                torch.cuda.empty_cache()
                
                if args.memory_efficient:
                    print("Already in memory-efficient mode, skipping to next section")
                    continue
                    
                print("Retrying with memory-efficient settings...")
                # Reduce batch size temporarily for this section
                original_batch_size = config['batch_size']
                config['batch_size'] = max(1, original_batch_size // 8)
                
                # Rebuild dataloader with smaller batch size
                section_df = test_dl.dataset.df
                temp_ds = StainDataSet(df=section_df)
                temp_dl = DataLoader(
                    temp_ds, 
                    batch_size=config['batch_size'],
                    num_workers=1,  # Reduce worker count too
                    shuffle=False,
                    pin_memory=False,  # Disable pin_memory to reduce memory usage
                    drop_last=False
                )
                
                # Try again with reduced settings
                try:
                    generate_saliency_maps(
                        dataloader=temp_dl,
                        melanA_model=melanA_model,
                        sox10_model=sox10_model,
                        norm_transforms=norm_transforms,
                        output_dir=section_output_dir,
                        conf_thresholds=tuple(args.conf_thresholds),
                        min_blob_area=100,
                        max_tiles=args.max_tiles,
                        enable_visualization=False  # Disable visualization to save memory
                    )
                    
                    # Update progress
                    if section_name not in progress_metadata['sections_completed']:
                        progress_metadata['sections_completed'].append(section_name)
                        save_progress_metadata(output_dir, progress_metadata)
                        
                    print(f"Completed section with reduced settings: {section_name}")
                    
                except Exception as e2:
                    print(f"Failed to process section {section_name} even with reduced settings: {e2}")
                    continue
                    
                # Restore original batch size
                config['batch_size'] = original_batch_size
            else:
                # For non-OOM errors, just continue to next section
                continue
    
    print("All processing complete!")