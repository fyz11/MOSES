#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 22:49:32 2017

@author: felix
@email: felixzhou1@gmail.com

This script and all the code within is licensed under the Ludwig License.

see README.md for details on usage.

"""

# read a single frame from a multi-page .tif file.
def read_PIL_frame(tiffile, frame):

    """
    Use pillow library to read .tif/.TIF files. (single frame)
    
    Input:
    ------
    tiffile: input .tif file to read, can be multipage .tif (string)
    frame: desired frarme number given as C-style 0-indexing (int)
    
    Output:
    -------
    an image as a numpy array either (n_rows x n_cols) for grayscale or (n_rows x n_cols x 3) for RGB
    
    """
    from PIL import Image
    import numpy as np

    img = Image.open(tiffile)

    img.seek(frame)

    return np.array(img)
    

def read_multiimg_PIL(tiffile):
    
    """
    Use pillow library to read .tif/.TIF files. (single frame)
    
    Input:
    ------
    tiffile: input .tif file to read, can be multipage .tif (string)
    frame: desired frarme number given as C-style 0-indexing (int)

    Output:
    -------
    a numpy array that is either:
        (n_frames x n_rows x n_cols) for grayscale or 
        (n_frames x n_rows x n_cols x 3) for RGB

    """

    from PIL import Image
    import numpy as np

    img = Image.open(tiffile)

    imgs = []
    read = True

    frame = 0

    while read:
        try:
            img.seek(frame) # select this as the image
            imgs.append(np.array(img)[None,:,:])
            frame += 1
        except EOFError:
            # Not enough frames in img
            break

    return np.concatenate(imgs, axis=0)
    

def mkdir(directory):
    
    import os 
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    return []

# =============================================================================
# Some utility scripts to help parse experiments
# =============================================================================
    

# aims to file folders containing meantrack outputs. 
def detect_experiments( infolder, exclude=['meantracks', 'optflow'], level1=True):

    """
    level1 - controls whether we detect in the immediate folder or walk the subdirectories. 
    """
    import os
    import numpy as np 
    
    subdirs = []

    if level1:
        for item in os.listdir(infolder):
            fulldirpath = os.path.join(infolder, item)
            if os.path.isdir(fulldirpath):
                subdirs.append(fulldirpath)
    else:
        for dirName, subdirList, fileList in os.walk(infolder):
            for subdir in subdirList:
                if (exclude[0]) not in subdir and (exclude[1] not in subdir):
                    subdirs.append(subdir)

    return np.sort(subdirs)

# aims to return all RG tracks associated. 
def detect_experiment_tracks_RG( infolder, ext='.mat', include='filt2_'):

    """
    how to generalise the script to more general case. ?
    """
    import os
    import numpy as np 

    red = []
    green = []

    infolder = os.path.join(infolder, 'meantracks')

    for dirName, subdirList, fileList in os.walk(infolder):

        for f in fileList:
            if ext in f:
                if 'green' in f:
                    
                    if include is None:
                        index = (f.split('_green')[0])[-3:] # grab the digits.
                        green.append((os.path.join(dirName, f), index))
                    else:
                        if include in f:
                            index = (f.split('_green')[0])[-3:] # grab the digits.
                            green.append((os.path.join(dirName, f), index))

                if 'red' in f:
                    if include is None:
                        index = (f.split('_red')[0])[-3:]
                        red.append((os.path.join(dirName, f), index))
                    else:
                        if include in f:
                            index = (f.split('_red')[0])[-3:]
                            red.append((os.path.join(dirName, f), index))

    # sorting the tuple
    def getKey(item):
        return item[1]

    red = sorted(red, key=getKey)
    green = sorted(green, key=getKey)

    # form new vectors:
    red_files = [item[0] for item in red]
    green_files = [item[0] for item in green]

    return np.array(red_files).ravel(), np.array(green_files).ravel()


def detect_files( folder, ext='.tif'):

    import os 
    import numpy as np
    
    filepaths = []
    fnames = []
    
    for f in os.listdir(folder):
        if ext in f: 
            filepaths.append(os.path.join(folder, f))
            fnames.append(f)
            
    return np.hstack(filepaths), np.hstack(fnames)

def save_multipage_tiff(np_array, savename):
    
    """
    save numpy array of images as a multipage tiff file.... 
    
    Input:
    =====
    np_array: (n_frames, n_rows, n_cols)
    savename: filepath to save the output .tif stack. 
    
    Output:
    =====
    void function
    
    """    
    from tifffile import imsave
    import numpy as np 
    
    if np_array.max() < 1.1:
        imsave(savename, np.uint8(255*np_array))
    else:
        imsave(savename, np.uint8(np_array))
    
    return [] 
    
