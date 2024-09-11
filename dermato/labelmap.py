# Import libraries
import os
import yaml
import argparse
from glob import glob
import time
import math

import numpy as np
import zarr 
import cv2

import multiprocessing as mp
import ctypes as ct
from multiprocessing import sharedctypes as sct

import matplotlib.pyplot as plt
CCMAP = plt.cm.cividis

from stain_labeling import count_sox10_stain_blobs, count_mela_prop_section_from_ihc_dab

#---------------------------------------------------
# Functions for generating discrete labelmaps
def fill_worker(y, x, label):
    """
    Multiprocess worker function for filling pixel color of shared array 
    based on label obtained from IHC color-thresohld
    
    Args:
        y (int): The y-coordinate.
        x (int): The x-coordinate.
        label (int): The IHC label.
    Returns:
        None
    
    """
    
    # Set fill value
    fill = 25
    if label:
        fill = 233
    
    # Get index
    width = IMSHAPE[1]
    y_offset = int(y * width)
    x_offset = x
    idx = y_offset + x_offset

    # Fill shared array
    LABELMAP[idx] = fill
    return

def slice_init():
    """Initializer for multiprocessing pool, allowing child processes to inherit parent processes memory registers"""
    pass

def mpFill_labelmap(indices, nprocs=16):
    """
    Multiprocess manager function for filling in indices of the labelmap (shared array). Function expects 
    input indices to be a list of lists, where each list contains tuples of y,x coordinates and the 
    corresponding IHC label for that coordinate. Each sublists' values are unpacked using starmap_async.
    
    Args:
        indices (list): The list of lists containing tuples of y,x coordinates and their corresponding IHC labels.
        nprocs (int, optional): The number of processes to use. Defaults to 16.
    Returns:
        None
    
    """
    with mp.Pool(initializer=slice_init, processes=nprocs) as pool:
        res = [pool.starmap_async(fill_worker, l) for l in indices if l is not None]
        res = [p.get() for p in res]
    print('')
    print('Finished Contstructing LabelMap')
    return

def calc_window_points(y, x, imShape, window_size=256):
    """
    Calculates window dimensions and slices input image, optionally padding to fit window_size
    
    Args:
        y (int): The y-coordinate.
        x (int): The x-coordinate.
        imShape (tuple): The image shape.
        window_size (int, optional): The window size. Defaults to 256.
    Returns:
        list: The window points.
        list: The padding values.
    
    """
     
    h, w = imShape[0], imShape[1]
    #print('h, w, ch = {}, {}, {}'.format(h, w, ch))
    cp = math.ceil(window_size/2) # center point
    top,bot,left,right = y, h-y, x, w-x 
    nleft,nright,ntop,nbot = cp, window_size - cp, cp, window_size - cp
    pt,pb,pl,pr = np.array([min(0, top-ntop), min(0, bot-nbot), min(0, left-nleft), min(0, right-nright)], dtype=np.int8)
    ystart = max(0, y + (pt - cp))
    ystop = min(h, y + cp)
    xstart = max(0, x + (pl - cp))
    xstop = min(w, x + (cp + pr))
    return [ystart, ystop, xstart, xstop], [abs(int(pt)), abs(int(pb)), abs(int(pl)), abs(int(pr))]

def gen_window_from_buffer(rawArr, imShape, wpoints, pad_vals, dtype=np.uint8, pad=True):
    """
    Return optionally padded window slice from shared-array buffer specified by window points
    
    Args:
        rawArr (sct.RawArray): The shared array.
        imShape (tuple): The image shape.
        wpoints (list): The window points.
        pad_vals (list): The padding values.
        dtype (ctypes, optional): The data type. Defaults to np.uint8.
        pad (bool, optional): Whether to pad the window. Defaults to True.
    Returns:
        numpy.ndarray: The window slice.
    
    """
    # Load window vars
    h, w, ch = imShape
    ystart, ystop, xstart, xstop = wpoints
    pt, pb, pl, pr = pad_vals
    # print(f'pt:{pt}, pb:{pb}, pl:{pl}, pr:{pr}')
    
    # Calc buffer vars
    row_offset = ystart * w * ch
    buffer_size = ((ystop-ystart) * w * ch)
    
    # Slice shared array and pad if necessary
    imSlice = np.frombuffer(rawArr, offset=row_offset, count=buffer_size, dtype=dtype).reshape(ystop-ystart, w, ch)
    window = imSlice[:, xstart:xstop]
    if pad:
        window = np.pad(window, ((pt, pb), (pl, pr), (0,0)), mode='constant')
    return window

def gen_dab_window_from_buffer(rawArr, imShape, wpoints, pad_vals, dtype=np.uint8, pad=True):
    """
    Return optionally padded window slice from shared-array buffer specified by window points
    
    Args:
        rawArr (sct.RawArray): The shared array.
        imShape (tuple): The image shape.
        wpoints (list): The window points.
        pad_vals (list): The padding values.
        dtype (ctypes, optional): The data type. Defaults to np.uint8.
        pad (bool, optional): Whether to pad the window. Defaults to True.
    Returns:
        numpy.ndarray: The window slice.
    
    """
    # Load window vars
    h, w = imShape
    ystart, ystop, xstart, xstop = wpoints
    pt, pb, pl, pr = pad_vals
    
    # Calc buffer vars
    row_offset = ystart * w 
    buffer_size = ((ystop-ystart) * w)
    
    # Slice shared array and pad if necessary
    imSlice = np.frombuffer(rawArr, offset=row_offset, count=buffer_size, dtype=dtype).reshape(ystop-ystart, w)
    window = imSlice[:, xstart:xstop]
    if pad:
        window = np.pad(window, ((pt, pb), (pl, pr)), mode='constant')
    return window

def label_worker(y, section_params, stain, stride=6, window_size=256, dtype=np.uint8):
    """
    Multiprocess worker function for asyncronously extracting tiled regions around each y,x coordinate and 
    generating a positive or negative label for the y,x coordinate based on presense of IHC color within the 
    tiled region. Returns a list of tuples containing y,x coordinates with their corresponding label.
    
    Args:
        y (int): The y-coordinate.
        section_params (dict): The section parameters.
        stain (str): The stain type.
        stride (int, optional): The stride value for creating tiles. Defaults to 6.
        window_size (int, optional): The window size for creating tiles. Defaults to 256.
        dtype (ctypes, optional): The data type. Defaults to np.uint8.
    Returns:
        list: The list of tuples containing y,x coordinates and their corresponding IHC labels.
    
    """
    
    # Staging area
    indices = []
    if not y % stride:
        xrange = range(0, IMSHAPE[1], 1)
    else:
        xrange = range(0, IMSHAPE[1], stride)
    
    stain_color, dab_min_cutoff, first_kernel, second_kernel, threshold = section_params 
    
    # Loop through coordinates and extract 256x256 window surrounding coordinate
    for x in xrange:
        wpoints, pad_vals = calc_window_points(y, x, IMSHAPE, window_size=window_size)
        window = gen_window_from_buffer(SHARED_ARRAY, IMSHAPE, wpoints, pad_vals, pad=True)
        
        dab_wpoints, dab_pad_vals = calc_window_points(y, x, dab_IMSHAPE, window_size=window_size)
        dab_window = gen_dab_window_from_buffer(dab_SHARED_ARRAY, dab_IMSHAPE, dab_wpoints, dab_pad_vals, pad=True)
        
        # If the window is not totally black
        if window.any():
            assert window.shape[:2] == dab_window.shape

            if stain == 'sox10':
                num_sox10_blobs = count_sox10_stain_blobs(window, dab_window, stain_color, dab_min_cutoff, first_kernel, second_kernel)
                label = 1 if num_sox10_blobs > threshold else 0
            else: # stain == 'melana_melpro'
                mela_prop, _ = count_mela_prop_section_from_ihc_dab(window, dab_window, stain_color, dab_min_cutoff, first_kernel, second_kernel)
                label = 1 if mela_prop > threshold else 0
            
            indices.append((y,x,label))
    
    # Return indices 
    if len(indices) == 0:
        # logger.warning('WARNING: indices is NoneType')
        return None
    else:
        return indices



def mpLabel_indices(section_params, stain, stride=6, window_size=256, nprocs=16):
    """
    Multiprocess manager function for asynchronously retrieving y,x coordinates and their corresponding 
    IHC ground-truth labels as determined by an IHC color-thresohld. Returns a list of lists, where each list 
    contains tuples of y,x coordinates and their corresponding IHC labels
    
    Args:
        section_params (dict): The section parameters.
        stain (str): The stain type.
        stride (int, optional): The stride value for creating tiles. Defaults to 6.
        window_size (int, optional): The window size for creating tiles. Defaults to 256.
        nprocs (int, optional): The number of processes to use. Defaults to 16.
    Returns:
        list: The list of lists containing tuples of y,x coordinates and their corresponding IHC labels.
    
    """
    with mp.Pool(initializer=slice_init, processes=nprocs) as pool:
        indices = [pool.apply_async(label_worker, args=(y, section_params, stain, stride, window_size)) for y in range(IMSHAPE[0])]
        indices = [p.get() for p in indices]
    print('\tLabeled (x,y) coordinates via IHC color-threshold')
    print('')
    return indices

def subscript_sct_RawArray(arr, typecode=ct.c_uint8, verbose=True):
    """
    Subscripts shared-ctypes, read-only array from existing numpy array. 
    Shared array has no synchronization lock. Input array should be 1-dimensional (flattened or raveled).
    
    Args: 
        arr (numpy.ndarray): The input array.
        typecode (ctypes): The type of the array.
        verbose (bool, optional): Whether to print verbose output. Defaults to True.
    Returns:
        sct.RawArray: The shared array.
    """
    start_time = time.time()
    raw_arr = sct.RawArray(typecode, arr.size)
    raw_arr[:] = arr
    if verbose:
        print('\tCreated shared ctypes array ({}) in {} seconds'.format(typecode, time.time() - start_time))
    return raw_arr

def subscript_sct_Array(arr, typecode=ct.c_uint8, verbose=True):
    """
    Subscripts shared-ctypes, read-only array from existing numpy array. 
    Shared array has no synchronization lock. Input array should be 1-dimensional (flattened or raveled).
    
    Args: 
        arr (numpy.ndarray): The input array.
        typecode (ctypes): The type of the array.
        verbose (bool, optional): Whether to print verbose output. Defaults to True.
    Returns:
        sct.Array: The shared array.
    
    """
    start_time = time.time()
    raw_arr = sct.Array(typecode, arr.size, lock=True)
    raw_arr[:] = arr
    if verbose:
        print('\tCreated shared ctypes array ({}) in {} seconds'.format(typecode, time.time() - start_time))
    return raw_arr

def labelmap_handler(img, dab_img, outfile, stain, section_params, 
                     stride=6, window_size=256, resize=False,  nprocs=16,
                    verbose=True, seed=20200327):
    """
    Handler function for creating IHC label-maps

    Args:
        img (numpy.ndarray): The input image.
        dab_img (numpy.ndarray): The DAB image.
        outfile (str): The output file path for saving the label-map.
        stain (str): The stain type.
        section_params (dict): The section parameters.
        stride (int, optional): The stride value for creating tiles. Defaults to 6.
        window_size (int, optional): The window size for creating tiles. Defaults to 256.
        resize (bool, optional): Whether to resize the image. Defaults to False.
        nprocs (int, optional): The number of processes to use. Defaults to 16.
        verbose (bool, optional): Whether to print verbose output. Defaults to True.
        seed (int, optional): The seed value. Defaults to 20200327.
    Returns:
        numpy.ndarray: The label-map image.
    
    """

    # Load HSV image and create empty container for hi-res label-map
    print('')
    print('Generating IHC LabelMap')
    print('')
    print('Loading IHC image and creating shared arrays')
    # imHSV = load_image(imPath, HSV=True, resize=resize, verbose=verbose)
    imNull = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    # Load shared arrays
    rawArr = subscript_sct_RawArray(img.ravel(), typecode=ct.c_uint8, verbose=verbose) #this rawArr has 3 ch
    dab_rawArr = subscript_sct_RawArray(dab_img.ravel(), typecode=ct.c_uint8, verbose=verbose) #this rawArr has 0 ch
    lmap = subscript_sct_Array(imNull.ravel(), typecode=ct.c_uint8, verbose=verbose)
    
    # Define global variables for child-processes to inherit
    global SHARED_ARRAY, IMSHAPE, dab_SHARED_ARRAY, dab_IMSHAPE, LABELMAP
    SHARED_ARRAY = rawArr # SHARED_ARRAY has 3 ch
    IMSHAPE = img.shape # IMSHPAE has 3 ch
    dab_SHARED_ARRAY = dab_rawArr # dab_SHARED_ARRAY has 0 ch
    dab_IMSHAPE = dab_img.shape # IMSHPAE has 0 ch
    LABELMAP = lmap

    # Extract relevant indices from which to create tiles, and split
    print('')
    print('Extracting (x,y) coordinates and obtaining IHC labels (nprocs: {} | window-size: {} | stride: {})'
                .format(nprocs, window_size, stride))
    start = time.time()
    indices = mpLabel_indices(section_params=section_params, stain=stain, stride=stride, window_size=window_size, nprocs=nprocs)

    # Fill high-res label-map indices with pos/neg values
    print('Filling LabelMap indices')
    mpFill_labelmap(indices, nprocs=nprocs)
    print('Total extraction and fill time: {}'.format(time.time() - start))
    
    # Convert shared-array to numpy array and save label-map
    imLMAP = np.frombuffer(LABELMAP.get_obj(), dtype=np.uint8).reshape(IMSHAPE[0], IMSHAPE[1])
    np.save(outfile, imLMAP) # save labelmap in npy format
    # cv2.imwrite(outfile.replace('npy', 'jpg'), imLMAP)
    
    # Create blended image of label-map and IHC
    imBGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    imLMAP3D = np.dstack([imLMAP, imLMAP, imLMAP])
    imBlend = cv2.addWeighted(imBGR, 0.25, imLMAP3D, 0.75, 0.0)
    discretelabelmap_outfile = os.path.dirname(outfile) + '/discrete-labelmap.png'
    cv2.imwrite(discretelabelmap_outfile, imBlend) # save discrete labelmap in png format

    # save IHC
    ihc_outfile = os.path.dirname(outfile) + '/IHC.png'
    cv2.imwrite(ihc_outfile, imBGR) # save IHC in png format
    return imLMAP

#---------------------------------------------------
# Functions for generating continuous labelmaps
def square_filler(idx, val, window_size=256):
    """Multiprocess worker that adds brightness/darkness to a square region of the shared array, 
    as determined by x,y coordinate label"""
    
    # Calc shared-array window coordinates based on x,y coords
    y,x = idx
    h, w = IMSHAPE
    wpoints, pad_points = calc_window_points(y, x, IMSHAPE, window_size=window_size)
    ystart, ystop, xstart, xstop = wpoints
    
    # Calc buffer vars
    row_offset = ystart * w * 4 # Offset in bytes
    buffer_size = ((ystop-ystart) * w) # Buffer_size in count/indices

    # Slice shared array and pad if necessary
    imSlice = np.frombuffer(SQUAREMAP.get_obj(), offset=row_offset, count=buffer_size, dtype=np.float32).reshape(ystop-ystart, w)
    window = imSlice[:, xstart:xstop]
    
    # Add fill_vals to shared array window-values
    fill_window = np.full(window.shape, val, dtype=np.float32)
    update = np.add(window, fill_window, dtype=np.float32)
    np.clip(update, 0., 255., update)
    imSlice[:, xstart:xstop] = update
    return True



def mpDispense_fillers(indices, val, window_size=256, nprocs=16):
    """Multiprocess managers dispenses workers to extract window region based on (x,y) coordinate values, 
    and increases/decreases brightness according to val set."""
    with mp.Pool(initializer=slice_init, processes=nprocs) as pool:
        res = [pool.apply_async(square_filler, args=(idx, val, window_size)) for idx in indices]
        res = [p.get() for p in res]
    print('\tFilled {} indices from image with shape {}'.format(len(res), IMSHAPE))
    return



def get_indices(lmap, val):
    indices = np.where(lmap == val)
    indices = list(zip(indices[0], indices[1]))
    return indices

def squaremap_handler(rgb_img, lmap_img, outfn, stride=6, window_size=256, resize=False, nprocs=16, verbose=True, seed=2020331):
    """Handler function for creating ground-truth overlapping tile-labels -- truth-tile Map or SquareMap"""
    
    # Load RGB image, create neutral mask, and convert mask to shared array
    print('')
    print('Generating Truth-tile Map (SquareMap)')
    print('')
    print('Loading IHC image and creating shared array')
    #imBGR = load_image(imPath, resize=resize, verbose=verbose)
    imBGR = rgb_img[...,::-1] #np.transpose(rgb_img, (2,1,0)) #RGB->BGR
    temp = np.copy(imBGR)
    mask_indices = (temp > 0)
    temp[mask_indices] = 127
    print(f'temp: {temp.shape}')
    imMASK = np.copy(temp[:,:,0]).astype(np.float32)
    print(f'imMASK: {imMASK.shape}')
    smap = subscript_sct_Array(imMASK.ravel(), typecode=ct.c_float, verbose=False)

    #  Find positive and negative indices in label-map
    print('')
    print('Loading LabelMap and identifying positive/negative indices')
    # imLMAP = np.load(lmap_path)
    pos_indices = get_indices(lmap_img, val=233) # hard coded
    neg_indices = get_indices(lmap_img, val=25) # hard coded
    print('    {} positive indices'.format(len(pos_indices)))
    print('    {} negative indices'.format(len(neg_indices)))

    # Define global variables for child-processes to inherit
    global IMSHAPE, SQUAREMAP
    IMSHAPE = imMASK.shape
    SQUAREMAP = smap

    # Extract relevant indices from which to create tiles, and split
    print('')
    print('Filling windows as indicated by positive and negative indices'.format(window_size, stride))
    mpDispense_fillers(neg_indices, -0.02, window_size=window_size, nprocs=nprocs)
    print(f'neg indices done')
    mpDispense_fillers(pos_indices, 0.04, window_size=window_size, nprocs=nprocs)
    print(f'post indices done')
    
    # Convert shared-array to numpy array and save label-map
    imSMAP = np.frombuffer(SQUAREMAP.get_obj(), dtype=np.float32).reshape(IMSHAPE).astype(np.uint8)
    SMAP3D = np.dstack([imSMAP, imSMAP, imSMAP])
    # recolor this SMAP3D
    recolor_SMAP3D = CCMAP(SMAP3D[:,:,0])
    recolor_SMAP3D = np.copy(recolor_SMAP3D[:,:,:3] * 255).astype(np.uint8)
    np.save(outfn, recolor_SMAP3D) # save cont_labelmap in npy
    # cv2.imwrite(outfn.replace('npy', 'jpg'), recolor_SMAP3D)
    
    # Create blended image of label-map and H&E
    w1 = 0.42
    w2 = 0.58
    imBlend = cv2.addWeighted(rgb_img, w1, recolor_SMAP3D, w2, 0.0)
    imBlend = cv2.cvtColor(imBlend, cv2.COLOR_RGB2BGR)
    contlabelmap_outfile = os.path.dirname(outfn) + '/cont-labelmap.png'
    cv2.imwrite(contlabelmap_outfile, imBlend) # save blend image in png
    return recolor_SMAP3D



#---------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_config_yaml', type=str, choices=['sox10_labeling.yaml', 'melana_melpro_labeling.yaml'], default='sox10_labeling.yaml')
    parser.add_argument("--sections", nargs='+', default=None)
    parser.add_argument("--stride", type=int, default=6)  
    parser.add_argument("--nprocs", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default='/scratch/mtada/rclone')

    args = parser.parse_args()

    # parse yaml file
    with open(args.label_config_yaml, 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    # dictionary of section names and their corresponding parameters
    section_config = data['sections']

    # get the selected the sections for IHC and DAB
    if args.sections is not None:
        all_sections = args.sections
    # get all the sections for IHC and DAB
    else:
        all_sections = list(data['sections'].keys())
    print(f'All sections: {len(all_sections)}')
    # **hard coded** get the path to the IHC and DAB images
    all_ihc_sections_path = [glob(f'/srv/nas/mk1/users/mtada/paired_aligned_images/*/IHC/{section_name}.zarr')[0] for section_name in all_sections]
    all_dab_sections_path = [glob(f'/srv/nas/mk1/users/mtada/paired_aligned_images/*/DAB/{section_name}.zarr')[0] for section_name in all_sections]

    stain = args.label_config_yaml.split('_')[0]
    print(f'Processing {stain} stain')
    # loop over all the sections
    for ihc_path, dab_path in zip(all_ihc_sections_path, all_dab_sections_path):
        # dataset
        dataset = ihc_path.split('/')[-3]
        # section and wsi name
        section_name = os.path.basename(ihc_path).split('.')[0]

        print(section_name)
        wsi_name = section_name.split('_')[0]
        print(f'Processing {dataset} {wsi_name} {section_name}')
        path2img = os.path.join(args.output_dir, f'{dataset}/{wsi_name}/{section_name}')
        if not os.path.isdir(path2img):
            os.makedirs(path2img, exist_ok=True)

        # get the parameters for the section
        section_params = section_config[section_name]
        stain_color = section_params['stain_color']
        dab_min_cutoff = section_params['dab_thred']
        first_kernel = section_params['first_kernel']
        second_kernel = section_params['second_kernel']
        threshold = section_params['threshold']
        section_params = [stain_color, dab_min_cutoff, first_kernel, second_kernel, threshold]

        # load the IHC and DAB images
        ihc = zarr.load(ihc_path)
        dab = zarr.load(dab_path)
        dis_labelmap_name = f'stain{stain}_{threshold}min_discrete-labelmaps.npy' 
        out_dis_labelmap = os.path.join(path2img, dis_labelmap_name)
        # Generate discrete Label map
        # 1) save the labelmap as a npy file, 2) save the labelmap as a jpeg file, 3) save the labelmap+IHC as a blend image
        labelmap = labelmap_handler(
            img=ihc, dab_img=dab, 
            outfile=out_dis_labelmap, 
            stain=stain,
            section_params=section_params,
            stride=args.stride, window_size=256, nprocs=args.nprocs)
        print(f'    Discrete labelmap {labelmap.shape} saved to {out_dis_labelmap}')

        # save labelmap as npy
        # np_labelmap_outfile = out_labelmap #+ '/discrete.npy'
        # np.save(np_labelmap_outfile, labelmap)

        # Generate continuous Label map
        con_labelmap_name = f'stain{stain}_{threshold}min_continuous-labelmaps.npy' 
        out_cont_labelmap = os.path.join(path2img, con_labelmap_name)
        recolor_SMAP3D = squaremap_handler(
            rgb_img=ihc, lmap_img=labelmap, 
            outfn=out_cont_labelmap, 
            stride=args.stride, window_size=256, nprocs=args.nprocs)
        print(f'    Continyous labelmap {recolor_SMAP3D.shape} saved to {out_cont_labelmap}')

        # save labelmap as npy
        # np_cont_labelmap_outfile = out_cont_labelmap + '/cont.npy'
        # np.save(np_cont_labelmap_outfile, recolor_SMAP3D)

#---------------------------------------------------
if __name__ == '__main__':
    main()