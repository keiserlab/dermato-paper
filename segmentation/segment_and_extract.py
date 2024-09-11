# I/O Handling
from __future__ import print_function
import csv,glob,re
import os,sys

# Data handling
import numpy as np
import math

# Standard tools
from functools import reduce

# Image handling
import pyvips as Vips
import cv2

# Figures + Plotting
import matplotlib.pyplot as plt
import matplotlib.cm as color_mapper
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

# Custom functions
from image_utils import array_vips, show_vips, load_small_vips_image, load_full_vips_image


'''
#############################################################################################################
#                                            DEFAULT VARIABLES                                           
#############################################################################################################
'''

LARGE_DILATION = Vips.Image.new_from_array([[128, 128, 255, 128, 128],
                                            [128, 128, 255, 128, 128],
                                            [255, 255, 255, 255, 255],
                                            [128, 128, 255, 128, 128],
                                            [128, 128, 255, 128, 128]])    


SMALL_DILATION = Vips.Image.new_from_array([[128, 255, 128],
                                            [255, 255, 255],
                                            [128, 255, 128]])


###########################################################################################################################################
#        #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       # 
#                                                              PRIMARY FUNCTIONS                                                           
#    #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #     
###########################################################################################################################################


def get_segment_regions_from_smask(smask):
    asmask = array_vips(smask).squeeze()
    labels = np.unique(asmask)[1:]
    regions = [] 
    for l in labels:        
        y, x = np.where(asmask == l)
        left, top = np.min(x), np.min(y)
        regions.append((left, top, np.max(x) - left, np.max(y) - top))            
    return regions, asmask.shape[0], asmask.shape[1]



def guess_background_label_from_corners(lmask, lcount):
        """Find segment labels of each corner, use the largest label"""
        corners = [(0, 0), (lmask.width-1, 0), (lmask.width-1, lmask.height-1), (0, lmask.height-1)]
        labels = [int(lmask.getpoint(*p)[0]) for p in corners]
        max_label = reduce(lambda a, b: a if lcount[a] > lcount[b] else b, labels)
        return max_label



def thresh_hist_mask_for_preview(imPath, chroma=6.0, min_percent=0.13, max_percent=1.0, rotation=None, level=3, verbose=True):
    """Loads small vesion of image, labels regions based on chroma value, segments image sections, and plots figures"""
    
    # Setup image monikers and load image object
    abspath = os.path.abspath(imPath)
    basename = os.path.basename(abspath)
    wsi_name = basename.split('.')[0]
    imGauss = load_small_vips_image(abspath, rotation=rotation, level=level, gausblur=5.5) # gaus-blur hard-coded
    ##image = load_small_vips_image_bilateral(abspath, rotation=rotation, level=level)

    # Create image mask by filtering LCH color-space
    mask = (imGauss.colourspace('VIPS_INTERPRETATION_LCH')[1] > chroma) & (imGauss.colourspace('VIPS_INTERPRETATION_LCH')[0] < 95)

    # Label connected regions in mask; Get pixel counts to remove background
    lmask = mask.labelregions()    
    lcount = np.bincount(array_vips(lmask).ravel())     
    background_label = guess_background_label_from_corners(lmask, lcount)
    lcount[background_label] = 0. # ignore background
    
    # Create mask from segment labels within min_percent and max_percent size of largest identified segment
    max_count = np.max(lcount)
    relative_min = max_count * min_percent
    relative_max = max_count * max_percent
    if max_percent == 1.0:
        segmented_labels = np.where(lcount > relative_min)[0]
    else:
        segmented_labels = np.where((lcount > relative_min) & (lcount <= relative_max))[0]
    segmented_mask = sum([(lmask[0] == l) * i for i, l in enumerate(segmented_labels, 1)])
    
    # Plot Figures
    if verbose:
        f = plt.figure(figsize=(18, 5))
        window = gridspec.GridSpec(1,2, width_ratios=[2,1])
        seg_window = gridspec.GridSpecFromSubplotSpec(3, 4, subplot_spec=window[0], wspace=0.7)
        hist_window = gridspec.GridSpecFromSubplotSpec(3, 4, subplot_spec=window[1])

        # Get colors of labeled regions for legend creation
        mask_vals = [255 * i+1 for i, l in enumerate(segmented_labels, 1)]
        max_mv = max(mask_vals)
        color_vals = [float(val)/float(max_mv) for val in mask_vals]
        
        # Plot raw image
        ax1 = plt.Subplot(f, seg_window[:2, :2])
        show_vips(imGauss, ax=ax1, show=False)
        ax1.set_title(os.path.basename(imPath))
        f.add_subplot(ax1)
        
        # Plot masked image
        ax2 = plt.Subplot(f, seg_window[:2, 2:])
        show_vips(segmented_mask, ax=ax2, show=False)
        ax2.set_title('{} segments in mask'.format(len(segmented_labels)))
        f.add_subplot(ax2)
        
        # Plot segment legend
        patches = []
        cmap = color_mapper.get_cmap('gist_ncar')
        for i,cv in enumerate(color_vals):
            patches.append(mpatches.Patch(color=cmap(cv), label='{}_seg{}'.format(wsi_name, i+1)))
        ax3 = plt.Subplot(f, seg_window[2, :])
        ax3.legend(handles=patches, loc='upper center', ncol=4, bbox_to_anchor=(.5, 1), fontsize=13)
        ax3.axis('off')
        f.add_subplot(ax3)
        
        # Plot segmentation histogram
        ax4 = plt.Subplot(f, hist_window[:2, :])
        ax4.plot(lcount.tolist())
        ax4.hlines(max_count * 0.1, 0., len(lcount), 
                   color='grey', label='default thresh (10% of max)')
        ax4.hlines(relative_min, 0., len(lcount), 
                   color='green', label='threshold ({}% of max)'.format(int(100*min_percent)))
        ymax = np.max(lcount)
        ax4.vlines(segmented_labels, int(ymax * .9), ymax, color='purple')  
        ax4.legend()
        ax4.set_title('mask histogram')
        f.add_subplot(ax4)
        plt.show()
    return imGauss, lmask, segmented_labels, segmented_mask



def thresh_hist_mask_for_segmentation(imPath, chroma=6.0, min_percent=0.13, max_percent=1.0, rotation=None, verbose=True):
    """Loads full image, labels regions based on chroma value, segments image sections"""
    # Setup image monikers and load image object
    abspath = os.path.abspath(imPath)
    basename = os.path.basename(abspath)
    wsi_name = basename.split('.')[0]
    imGauss = load_full_vips_image(abspath, rotation=rotation, gausblur=5.5)
    ##imGauss = load_full_vips_image_bilateral(abspath, rotation=rotation)
    
    # Create image mask by filtering LCH color-space 
    mask = (imGauss.colourspace('VIPS_INTERPRETATION_LCH')[1] > chroma) & (imGauss.colourspace('VIPS_INTERPRETATION_LCH')[0] < 95)
    del imGauss

    # Label connected regions in mask; Get pixel counts to remove background
    lmask = mask.labelregions()    
    lcount = np.bincount(array_vips(lmask).ravel())     
    background_label = guess_background_label_from_corners(lmask, lcount)
    lcount[background_label] = 0. # ignore background
    
    # Create mask from segment labels within min_percent and max_percent size of largest identified segment
    max_count = np.max(lcount)
    relative_min = max_count * min_percent
    relative_max = max_count * max_percent
    if max_percent == 1.0:
        segmented_labels = np.where(lcount > relative_min)[0]
    else:
        segmented_labels = np.where((lcount > relative_min) & (lcount <= relative_max))[0]
    segmented_mask = sum([(lmask[0] == l) * i for i, l in enumerate(segmented_labels, 1)])

    # Return original image + mask objects
    image = load_full_vips_image(abspath, rotation, gausblur=None)
    return image, lmask, segmented_labels, segmented_mask



def gen_mask_output(heFN, chroma=6.0, minThresh=.13, maxThresh=1.0, rotation=None, level=3, verbose=True):
    """Rreturns segmentation regions and masking objects; If verbose, generates segmentation plots"""
    preview_objects = thresh_hist_mask_for_preview(heFN, chroma=chroma, min_percent=minThresh, max_percent=maxThresh, 
                                                   rotation=rotation, level=level, verbose=verbose)
    vipsIM_small, lmask, seg_labels, smask = preview_objects
    seg_regions, imH, imW = get_segment_regions_from_smask(smask)
    bw_masks, color_masks = preview_segments(vipsIM_small, lmask, seg_labels, seg_regions)
    return seg_regions, bw_masks, color_masks, imW, imH



def retrieve_masked_segment(raw_image, segment_region, lmask, segment_label):
    """Extract bit-mask and color-mask of segment regions as defined by labeled mask"""
    
    # Extract segment region from full image and labeled mask
    left,top,width,height = segment_region
    segment = raw_image.extract_area(left, top, width, height)
    segmentArr = array_vips(segment[0].bandjoin([segment[1], segment[2]]))
    labeled_mask = lmask.extract_area(left, top, width, height)
    
    # Create new mask from labeled-region matching segment-label
    mask1D = labeled_mask == segment_label
    mask3D = mask1D.bandjoin([mask1D, mask1D])
    mask3Darr = array_vips(mask3D)
    mask3Darr[mask3Darr > 0] = 255
    bitmask = np.copy(mask3Darr[:,:,0])
    #print('\t\tmask size:', bitmask.shape)
    #print('\t\tsegment size:', segmentArr.shape)
    
    # Create color mask where regions not in mask3D are turned black
    masked_seg = np.ma.masked_array(segmentArr, mask=np.logical_not(mask3Darr))
    masked_seg = masked_seg.filled(0)
    bitmask2 = cv2.cvtColor(masked_seg, cv2.COLOR_BGR2GRAY)
    bitmask3 = cv2.equalizeHist(bitmask2)
    return bitmask3, masked_seg



def preview_segments(vipsIM_small, lmask, segment_labels, segment_regions):
    """Generate bit-mask and color-masks for alignment preview"""
    bit_masks = []
    color_masks = []
    # Loop through segment labels, extract regions that match segment label from vipsIM_small and lmask
    for i,label in enumerate(segment_labels):
        left,top,width,height = segment_regions[i]
        lseg_mask = lmask.extract_area(left, top, width, height)
        segment = array_vips(vipsIM_small.extract_area(left, top, width, height))
        # Create new mask where labeled mask == label, then bandjoin to create 3D mask
        mask1D = lseg_mask == label # creates mask with normal values  (0-255)
        mask3D = mask1D.bandjoin([mask1D, mask1D])
        mask3Darr = array_vips(mask3D)
        #mask3Darr[mask3Darr > 0] = 255 # may not be necessary, worth testing later
        bit_masks.append(mask3Darr[:,:,0])
        # Mask area in RGB image that does not match label
        segment_mask = np.ma.masked_array(segment, mask=np.logical_not(mask3Darr))
        segment_mask = segment_mask.filled(0)
        color_masks.append(segment_mask)
    return bit_masks, color_masks