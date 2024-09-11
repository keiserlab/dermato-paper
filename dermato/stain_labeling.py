# IImport libraries
import os
import cv2
import zarr
import numpy as np

# image analysis 
from scipy import ndimage
from skimage.filters import threshold_otsu, gaussian
from skimage.color import rgb2hed

#---------------------------------------------------

def replace_black2white(ihc_rgb):
    """
    Convert IHC black pixels to white pixels. 
    The range of black pixel is between (0,0,0) and (180,255,20) in HSV color space.
    """
    filtered_ihc_rgb = ihc_rgb.copy()
    ihc_hsv = cv2.cvtColor(ihc_rgb, cv2.COLOR_RGB2HSV)
    bgMask = np.array(cv2.inRange(ihc_hsv, (0,0,0), (180,255,20))).astype(bool)
    filtered_ihc_rgb[bgMask] = (255, 255, 255)

    return filtered_ihc_rgb

def generate_dab_from_ihc(ihc):
    """
    Generate DAB stain image from IHC image. 
    First, convert RGB to DAB using skimage's rgb2hed function. But the output pixel value is float and vaue. 
    Thus, we normallize the pixel value to 0-255 suing cv2.normalize. 

    Args:
        ihc (np.array): IHC image
    """
    sk_dab = None
    # Convert RGB to DAB using skimage's rgb2hed and normalize using cv2.normalize
    sk_dab = cv2.normalize(rgb2hed(ihc)[:,:,2], sk_dab, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return sk_dab


def segment_out_artifacts(thumbnail_im,
        deconvolve_first=False, stain_unmixing_routine_kwargs=None,
        n_thresholding_steps=2, sigma=0., min_size=500):
    """
    Segment out artifacts from IHC images
    a slight modification of histomicstk.preprocessing.color_deconvolution.get_tissue_mask
    
    Args:
        thumbnail_im (np.array): RGB IHC image
        deconvolve_first (bool): If True, deconvolve the image first
        stain_unmixing_routine_kwargs: Keyword arguments for stain unmixing routine
        n_thresholding_steps (int): Number of thresholding steps
        sigma (float): Sigma for gaussian smoothing
        min_size (int): Minimum size of tissue to be considered as region

    Returns:
        np.array: new IHC image where artifacts are replaced with white pixels
    """
    stain_unmixing_routine_kwargs= {}

    if len(thumbnail_im.shape) == 3:
        # grayscale thumbnail (inverted)
        thumbnail = 255 - cv2.cvtColor(thumbnail_im, cv2.COLOR_BGR2GRAY)
    else:
        thumbnail = thumbnail_im

    # the number of otsu thresholding steps
    for _ in range(n_thresholding_steps):

        # gaussian smoothing of grayscale thumbnail. If we not using it.
        if sigma > 0.0:
            thumbnail = gaussian(
                thumbnail, sigma=sigma,
                output=None, mode='nearest', preserve_range=True)

        # get threshold to keep analysis region
        try:
            thresh = threshold_otsu(thumbnail[thumbnail > 0])
        except ValueError:  # all values are zero
            thresh = 0

        # replace pixels outside analysis region with upper quantile pixels
        thumbnail[thumbnail < thresh] = 0

    # convert to binary
    mask = 0 + (thumbnail > 0)

    # find connected components
    labeled, _ = ndimage.label(mask)

    # only keep
    unique, counts = np.unique(labeled[labeled > 0], return_counts=True)
    discard = np.in1d(labeled, unique[counts < min_size])
    discard = discard.reshape(labeled.shape)
    labeled[discard] = 0

    # largest tissue region
    mask = labeled == unique[np.argmax(counts)]
    
    unique_labeled = np.unique(labeled)
    num_unique_labeled = len(unique_labeled)
        
    thumbnail_im_temp = thumbnail_im.copy()
    mask = labeled != 0
    # set artifact areas to white
    thumbnail_im_temp[mask] = (255,255,255)

    return thumbnail_im_temp

def segment_out_darkgreen(ihc):
    """
    Segment out dark green artifacts from IHC images and replace them with white pixels
    The dark green artifacts are manully identified by color range in HSV color space. 

    Args:
        ihc (np.array): IHC image
    
    Returns:
        ihc: new ihc where dark green artifacts are replaced with white pixels
    
    """
    # convert RGB to HSV
    hsv_ihc = cv2.cvtColor(ihc, cv2.COLOR_RGB2HSV)
    # color range of dark green artifacts in HSV color space
    dark_green_range = ((5, 0, 0), (140, 255, 60)) 
    artifact_mask = np.array(cv2.inRange(hsv_ihc, dark_green_range[0], dark_green_range[1])).astype(bool)
    # set artifact areas to white
    ihc[artifact_mask] = [255, 255, 255]
    return ihc

def count_sox10_stain_blobs(ihc_rgb_tile, dab_tile, stain_color, dab_thred, first_kernel, second_kernel):
    """
    Count the number of SOX10 stain blobs in a single tile.

    Args:
        ihc_rgb_tile (np.array): RGB channel IHC image
        dab_tile (np.array): a single channel DAB stain image
        stain_color (str): color of the stain, red or brown
        dab_thred (int): minimum cutoff value for DAB stain image
        first_kernel (tuple): kernel size for first morphological operation
        second_kernel (tuple): kernel size for second morphological operation
    
    Returns:
        num_sox10_blobs (int): number of SOX10 stain blobs
    """
    ihc_rgb_tile = ihc_rgb_tile.copy()
    # threshold DAB image. 
    binary_dab = dab_tile > dab_thred

    # morphological opening to remove small objects
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, first_kernel) #(3,3) default
    binary_dab_after_morp = cv2.morphologyEx(binary_dab.astype(np.float32), cv2.MORPH_OPEN, rect_kernel) 
    sox10_mask = binary_dab_after_morp.astype(bool)
    # set non-SOX10 area to black
    ihc_rgb_tile[~sox10_mask] = (0,0,0)

    cut_off_false_brown_stain = 140
    if stain_color == "red":
        hsv_ihc = cv2.cvtColor(ihc_rgb_tile, cv2.COLOR_RGB2HSV)
        hue_ihc = hsv_ihc[:,:,0]
        upper_false_brown_mask = hue_ihc < cut_off_false_brown_stain
        ihc_rgb_tile[upper_false_brown_mask] = (0,0,0)

    gray_ihc = cv2.cvtColor(ihc_rgb_tile, cv2.COLOR_RGB2GRAY)
    _, binary_mask = cv2.threshold(gray_ihc, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ihc_rgb_tile[~binary_mask.astype(bool)] = (0,0,0)

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, second_kernel) #(7,7) default
    binary_mask = cv2.morphologyEx(binary_mask.astype(np.float32), cv2.MORPH_OPEN, rect_kernel)
    binary_mask = cv2.morphologyEx(binary_mask.astype(np.float32), cv2.MORPH_CLOSE, rect_kernel)
    ihc_rgb_tile[~binary_mask.astype(bool)] = (0,0,0)

    contours, hierarchy = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt)]
    areas = [cv2.contourArea(cnt) for cnt in contours if cv2.contourArea(cnt)]

    num_sox10_blobs = len(bboxes)

    return num_sox10_blobs

def count_mela_prop_section_from_ihc_dab(ihc_rgb_tile, dab_tile, stain_color, dab_thred, first_kernel, second_kernel):
    """    
    Count the total melanA propotion of area in a single tile.

    Args:
        ihc_rgb_tile (np.array): RGB channel IHC image
        dab_tile (np.array): a single channel DAB stain image
        stain_color (str): color of the stain, red or brown
        dab_thred (int): minimum cutoff value for DAB stain image
        first_kernel (tuple): kernel size for first morphological operation
        second_kernel (tuple): kernel size for second morphological operation
    
    Returns:
        mela_prop (int): proportion of melanA stain in image
    """
    ihc_rgb_tile = ihc_rgb_tile.copy()

    # generate DAB image from IHC image
    # dab_tile = generate_dab_from_ihc(ihc_rgb_tile)

    # threshold DAB image. 
    binary_dab = dab_tile > dab_thred

    # morphological opening to remove small objects
    if first_kernel[0] is not None:
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, first_kernel) #(3,3) default
        binary_dab = cv2.morphologyEx(binary_dab.astype(np.float32), cv2.MORPH_OPEN, rect_kernel) 

    sox10_mask = binary_dab.astype(bool)
    # set non-SOX10 area to black
    ihc_rgb_tile[~sox10_mask] = (0,0,0)

    cut_off_false_brown_stain = 140
    if stain_color == "red":
        hsv_ihc = cv2.cvtColor(ihc_rgb_tile, cv2.COLOR_RGB2HSV)
        hue_ihc = hsv_ihc[:,:,0]
        upper_false_brown_mask = hue_ihc < cut_off_false_brown_stain
        ihc_rgb_tile[upper_false_brown_mask] = (0,0,0)

    gray_ihc = cv2.cvtColor(ihc_rgb_tile, cv2.COLOR_RGB2GRAY)
    _, binary_mask = cv2.threshold(gray_ihc, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ihc_rgb_tile[~binary_mask.astype(bool)] = (0,0,0)

    if second_kernel[0] is not None:
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, second_kernel) #(7,7) default
        binary_mask = cv2.morphologyEx(binary_mask.astype(np.float32), cv2.MORPH_OPEN, rect_kernel)
        # binary_mask = cv2.morphologyEx(binary_mask.astype(np.float32), cv2.MORPH_CLOSE, rect_kernel)
        ihc_rgb_tile[~binary_mask.astype(bool)] = (0,0,0)
    
    #make it an actual binary mask
    binary_mask = (binary_mask.astype(bool) * 1.0)

    return binary_mask.sum() / (binary_mask.shape[0] * binary_mask.shape[1]) * 100, binary_mask.astype(bool)

def count_mela_prop_from_ihc(ihc_rgb_tile, stain_color, dab_thred, first_kernel, second_kernel):
    """    
    Count the total melanA propotion of area in a single tile.

    Args:
        ihc_rgb_tile (np.array): RGB channel IHC image
        dab_tile (np.array): a single channel DAB stain image
        stain_color (str): color of the stain, red or brown
        dab_thred (int): minimum cutoff value for DAB stain image
        first_kernel (tuple): kernel size for first morphological operation
        second_kernel (tuple): kernel size for second morphological operation
    
    Returns:
        mela_prop (int): proportion of melanA stain in image
    """
    ihc_rgb_tile = ihc_rgb_tile.copy()

    # generate DAB image from IHC image
    dab_tile = generate_dab_from_ihc(ihc_rgb_tile)

    # threshold DAB image. 
    binary_dab = dab_tile > dab_thred

    # morphological opening to remove small objects
    if first_kernel[0] is not None:
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, first_kernel) #(3,3) default
        binary_dab = cv2.morphologyEx(binary_dab.astype(np.float32), cv2.MORPH_OPEN, rect_kernel) 

    sox10_mask = binary_dab.astype(bool)
    # set non-SOX10 area to black
    ihc_rgb_tile[~sox10_mask] = (0,0,0)

    cut_off_false_brown_stain = 140
    if stain_color == "red":
        hsv_ihc = cv2.cvtColor(ihc_rgb_tile, cv2.COLOR_RGB2HSV)
        hue_ihc = hsv_ihc[:,:,0]
        upper_false_brown_mask = hue_ihc < cut_off_false_brown_stain
        ihc_rgb_tile[upper_false_brown_mask] = (0,0,0)

    gray_ihc = cv2.cvtColor(ihc_rgb_tile, cv2.COLOR_RGB2GRAY)
    _, binary_mask = cv2.threshold(gray_ihc, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ihc_rgb_tile[~binary_mask.astype(bool)] = (0,0,0)

    if second_kernel[0] is not None:
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, second_kernel) #(7,7) default
        binary_mask = cv2.morphologyEx(binary_mask.astype(np.float32), cv2.MORPH_OPEN, rect_kernel)
        binary_mask = cv2.morphologyEx(binary_mask.astype(np.float32), cv2.MORPH_CLOSE, rect_kernel)
        ihc_rgb_tile[~binary_mask.astype(bool)] = (0,0,0)
    
    #make it an actual binary mask
    binary_mask = (binary_mask.astype(bool) * 1.0)

    return binary_mask.sum() / (binary_mask.shape[0] * binary_mask.shape[1])