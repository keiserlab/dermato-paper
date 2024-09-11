# I/O Handling
import os,sys
import csv

# Standard tools
import re
from itertools import zip_longest

# Data handling
import numpy as np

# Image handling
import pyvips as Vips
from cv2 import bilateralFilter

# Plotting
from matplotlib import pyplot as plt


# Default Variables
NP_DTYPE_TO_VIPS_FORMAT = {np.dtype('int8'): Vips.BandFormat.CHAR,
                           np.dtype('uint8'): Vips.BandFormat.UCHAR,
                           np.dtype('int16'): Vips.BandFormat.SHORT,
                           np.dtype('uint16'): Vips.BandFormat.USHORT,
                           np.dtype('int32'): Vips.BandFormat.INT,
                           np.dtype('float32'): Vips.BandFormat.FLOAT,
                           np.dtype('float64'): Vips.BandFormat.DOUBLE}

VIPS_FORMAT_TO_NP_DTYPE = {v:k for k, v in NP_DTYPE_TO_VIPS_FORMAT.items()}



###########################################################################################################################################
#        #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       # 
#                                                              PRIMARY FUNCTIONS                                                           
#    #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #     
###########################################################################################################################################


def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    """Sort like a filesystem"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]    



def grouper(iterable, chunksize):
    if (len(iterable) % 2) != 0:
        print('ERROR: Found uneven number of pairs matching file pattern.')
        return -1
    args = [iter(iterable)] * chunksize
    return list(zip_longest(*args, fillvalue=None))



def load_pyramidal_level(image_path, level=2):
    """Finds the number of levels in a pyramidal tiff and loads the last available"""
    if image_path.endswith('.tiff'):
        return Vips.Image.new_from_file(image_path + "[page={}]".format(level))
    elif image_path.endswith('.svs'):
        levels = int(Vips.Image.new_from_file(image_path).get('openslide.level-count'))
        if level > levels:
            print('Level {} does not exist, loading last level')
            return Vips.Image.new_from_file(image_path + "[level={}]".format(levels-1))
        else:
            return Vips.Image.new_from_file(image_path + "[level={}]".format(level))




def image_fields_dict(im_with_fields):
    """Finds descriptive fields in an image file and their values, throws them in a dict"""
    return {k:im_with_fields.get(k) 
            for k in im_with_fields.get_fields() 
            if im_with_fields.get_typeof(k)}



def array_vips(vips_image, verbose=False):
    """Converts vips image to numpy array"""
    dtype = VIPS_FORMAT_TO_NP_DTYPE[vips_image.get_value('format')]
    if verbose:
        print(dtype, vips_image.height, vips_image.width, vips_image.bands)
    return (np.frombuffer(vips_image.write_to_memory(), dtype=dtype) #np.uint8)
             .reshape(vips_image.height, vips_image.width, vips_image.bands))



def convert_array_to_vips(arr):
    if len(arr.shape) == 2:
        bands = 1
        height,width = arr.shape
        arr = arr.reshape(height, width, bands)
    else:
        height, width, bands = arr.shape
    linear = arr.reshape(width * height * bands)
    vips_image = Vips.Image.new_from_memory(linear.data, width, height, bands, Vips.BandFormat.UCHAR)
    return vips_image



def rotate_image(image, rotation):
    if not rotation:
        return image
    rotation = int(rotation)
    if rotation == 90:
        im = image.rot90()
    elif rotation == 180:
        im = image.rot180()
    elif rotation == 270:
        im = image.rot270()
    return im



def load_small_vips_image(image_path, rotation=None, level=3, gausblur=5.5):
    """Open smallest level of pyramidal tiff, blur and return image"""
    try:        
        small_image = load_pyramidal_level(image_path, level).gaussblur(gausblur)[:3]
    except Exception as e:
        print(e)
        try:
            print(image_fields_dict(Vips.Image.new_from_file(image_path)))
        except Exception as e:
            print(e)
        return -1
    if rotation:
        #print('\t\tRotating image by {} degrees'.format(rotation))
        small_image = rotate_image(small_image, rotation)
    return small_image



def load_small_vips_image_bilateral(image_path, rotation=None, level=3):
    """Open smallest level of pyramidal tiff, bilateral filter and return image"""
    imSmall = load_pyramidal_level(image_path, level)
    if rotation:
        imSmall = rotate_image(imSmall, rotation)
    imArray = array_vips(imSmall, verbose=False)
    imBlur = bilateralFilter(imArray[:,:,:3], 5, 99, 99)
    imSmall = convert_array_to_vips(imBlur)
    return imSmall



def load_full_vips_image(image_path, rotation=None, gausblur=None):
    """Return raw (full resolution) image"""
    #print('\tLoading full vips image from: {}'.format(image_path))
    if image_path.endswith('.tiff'):
        try:
            image = Vips.Image.new_from_file(image_path + "[page=0]")
        except Exception as e:
            print(e)
            print('Image causing exception: {}'.format(image_path))
    else:    
        try:
            image = Vips.Image.new_from_file(image_path + "[level=0]")
        except Exception as e:
            print(e)
            print('Image cuasing exception: {}'.format(image_path))
            try:
                print(image_fields_dict(Vips.Image.new_from_file(image_path)))
            except Exception as e:
                print(e)
            return -1
    if gausblur:
        image = image.gaussblur(gausblur)[:3]
    if rotation:
        image = rotate_image(image, rotation)
    return image


def load_full_vips_image_bilateral(image_path, rotation=None):
    """Return raw (full resolution) image"""
    if image_path.endswith('.tiff'):
        try:
            image = Vips.Image.new_from_file(image_path + "[page=0]")
        except Exception as e:
            print(e)
            print('Image causing exception: {}'.format(image_path))
    else:    
        try:
            image = Vips.Image.new_from_file(image_path + "[level=0]")
        except Exception as e:
            print(e)
            print('Image cuasing exception: {}'.format(image_path))
            try:
                print(image_fields_dict(Vips.Image.new_from_file(image_path)))
            except Exception as e:
                print(e)
            return -1
    if rotation:
        image = rotate_image(image, rotation)
    imArray = array_vips(image, verbose=False)
    imBlur = bilateralFilter(imArray[:,:,:3], 5, 99, 99)
    image = convert_array_to_vips(imBlur)
    return image
    

def show_vips(vips_image, ax=plt, show=True, verbose=False):
    """Plot vips image"""
    if not isinstance(vips_image, Vips.Image):
        return -1
    im_np = array_vips(vips_image)
    if verbose:
        print(im_np.shape)
    if vips_image.bands == 1:
        ax.imshow(im_np.squeeze()/np.max(im_np), cmap=plt.get_cmap('gist_ncar'))
    elif vips_image.bands == 2:
        im_np = im_np[:,:,1]
        ax.imshow(im_np/np.max(im_np), cmap=plt.get_cmap('gray'))
    else:
        ax.imshow(im_np)
    if show:
        plt.show()
    


def pad_image_before_loading(image, padsize=200):
    imPadded = np.pad(image, ((padsize, padsize), (padsize, padsize), (0,0)), mode='constant')
    return imPadded



def pad_to_match_images(image1, image2, nearest=10):
    im1_rows, im1_cols, ichannels = image1.shape()
    im2_rows, im2_cols, tchannels = image2.shape()
    row_diff = abs(im2_rows - im1_rows)
    col_diff = abs(im2_cols - im1_cols)



def pad_images_to_match(image1, image2):
    im1_rows, im1_cols = image1.shape[:2]
    im2_rows, im2_cols = image2.shape[:2]
    xdiff = abs(im2_rows - im1_rows)
    ydiff = abs(im2_cols - im1_cols)
    print('Initial image sizes:')
    print('\tImage1:', image1.shape)
    print('\tImage2:', image2.shape)
    print('xdiff: {} ydiff: {}'.format(xdiff, ydiff))
    # Padding if images are 2D
    if (len(image1.shape) == 2) and (len(image2.shape) == 2):
        if (im2_rows > im1_rows) and (im2_cols > im1_cols):
            pad1 = np.pad(image1, ((0, xdiff), (0, ydiff)), mode='constant')
            pad2 = image2
        elif (im2_rows > im1_rows) and (im2_cols < im1_cols):
            pad1 = np.pad(image1, ((0, xdiff), (0, 0)), mode='constant')
            pad2 = np.pad(image2, ((0, 0), (0, ydiff)), mode='constant')
        elif (im2_rows < im1_rows) and (im2_cols > im1_cols):
            pad1 = np.pad(image1, ((0, 0), (0, ydiff)), mode='constant')
            pad2 = np.pad(image2, ((0, xdiff), (0, 0)), mode='constant')
        elif (im2_rows < im1_rows) and (im2_cols < im1_cols):
            pad1 = image1
            pad2 = np.pad(image2, ((0, xdiff), (0, ydiff)), mode='constant')
        return pad1, pad2
    # Padding if images are 3D
    else:
        if (im2_rows > im1_rows) and (im2_cols > im1_cols):
            pad1 = np.pad(image1, ((0, xdiff), (0, ydiff), (0,0)), mode='constant')
            pad2 = image2
        elif (im2_rows > im1_rows) and (im2_cols < im1_cols):
            pad1 = np.pad(image1, ((0, xdiff), (0, 0), (0,0)), mode='constant')
            pad2 = np.pad(image2, ((0, 0), (0, ydiff), (0,0)), mode='constant')
        elif (im2_rows < im1_rows) and (im2_cols > im1_cols):
            pad1 = np.pad(image1, ((0, 0), (0, ydiff), (0,0)), mode='constant')
            pad2 = np.pad(image2, ((0, xdiff), (0, 0), (0,0)), mode='constant')
        elif (im2_rows < im1_rows) and (im2_cols < im1_cols):
            pad1 = image1
            pad2 = np.pad(image2, ((0, xdiff), (0, ydiff), (0,0)), mode='constant')
        return pad1,pad2
    print('SOMETHING WRONG WITH IMAGES TO PAD')
    ##print('padded to match sizes:')
    ##print('\tpad1:', pad1.shape)
    ##print('\tpad2:', pad2.shape)
    return None, None



def load_padded_images(seg_pair, padding=0.025):
    imHE,imIHC = seg_pair
    ncols, nrows, nchannels = imHE.shape
    pad = int(float(ncols)*padding)*2
    padHE = pad_image_before_loading(imHE, padsize=pad)
    padIHC = pad_image_before_loading(imIHC, padsize=pad)
    #HE_bitmask3D, IHC_bitmask3D = create_affine_bitmasks(padHE, padIHC)
    #HE_cmask3D, IHC_cmask3D = create_affine_colormasks(padHE, padIHC)
    #return HE_bitmask3D, IHC_bitmask3D, HE_cmask3D, IHC_cmask3D
    return padHE, padIHC