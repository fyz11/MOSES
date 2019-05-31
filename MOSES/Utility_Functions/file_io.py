#!/usr/bin/env python2
# -*- coding: utf-8 -*-
##########################################################################
# Created on Thu Jun 29 22:49:32 2017
#
# @author: felix
# @email: felixzhou1@gmail.com
#
# This script and all the code within is licensed under the Ludwig License.
#
# see README.md for details on usage.
###########################################################################

# read a single frame from a multi-page .tif file.
def read_PIL_frame(tiffile, frame):

    """ Use pillow library to read select .tif/.TIF files. (single frame)
    
    Parameters
    ----------
    tiffile : str
        input .tif file to read, can be multipage .tif (string)
    frame : int
        desired frame number given as C-style 0-indexing (int)
    
    Returns
    -------
    img : numpy array
        an image as a numpy array either (n_rows x n_cols) for grayscale or (n_rows x n_cols x 3) for RGB
    
    """
    from PIL import Image
    import numpy as np

    img = Image.open(tiffile)
    img.seek(frame)
    img = np.array(img)

    return img
    
def read_multiimg_PIL(tiffile):
    
    """ Use pillow library to read multipage .tif/.TIF files. 

    Parameters
    ----------
    tiffile : str
        input .tif file to read, can be multipage .tif (string)

    Returns
    -------
    imgs : numpy array
        either (n_frames x n_rows x n_cols) for grayscale or (n_frames x n_rows x n_cols x 3) for RGB

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

    imgs = np.concatenate(imgs, axis=0)

    return imgs

def read_multiimg_stack(tiffile, return_img=True):
    
    """ Use the tifffile.py library through Scikit-Image to read multipage bioformat files such as .tif/.lsm files.

    Parameters
    ----------
    tiffile : str
        input .tif/.lsm file to read, can be multipage .tif (string)
    return_img : bool
        boolean True/False to specify if the image should be read as a numpy array or just the object be returned.

    Returns
    -------
    img_object : Python object
        A read image object containing the attributes: pages, series and micromanager_metadata.
    imgs : numpy array (only if return_img=True)
        an (n_frames x n_slices x n_channels x n_rows x n_cols) image.

    """
    from skimage.external.tifffile import TiffFile

    im_object = TiffFile(Tiffile)

    if return_img:
        imgs = im_object.asarray()
        return im_object, imgs
    else:
        return im_object
    
def mkdir(directory):
    """ check if directory exists and create it through Python if it does not yet.

    Parameters
    ----------
    directory : str
        the directory path (absolute or relative) you wish to create (or check that it exists)

    Returns
    -------
        void function, no return
    """
    import os 
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    return []

# =============================================================================
# Some utility scripts to help parse experiments
# =============================================================================

# aims to file folders containing meantrack outputs. 
def detect_experiments( infolder, exclude=['meantracks', 'optflow'], level1=True):

    """ Locates all subdirectories or subdirectories immediately within infolder

    Parameters
    ----------
    infolder : str
        top-level root folder path
    exclude : list
        list of keywords to skip. Do not include subfolders with names that contain these keywords.
    level1 : bool
        If true searches only in the immediate folder else will walk through all subdirectories within infoldere
    
    Returns
    -------
        nothing, void function 
        
    """
    import os
    import numpy as np 
    
    subdirs = []

    if level1:
        for item in os.listdir(infolder):
            if exclude is not None:
                keep = 0
                for exc in exclude:
                    keep += exc in subdir
                if keep == 0:
                    fulldirpath = os.path.join(infolder, item)
                    if os.path.isdir(fulldirpath):
                        subdirs.append(fulldirpath)
            else:
                fulldirpath = os.path.join(infolder, item)
                if os.path.isdir(fulldirpath):
                    subdirs.append(fulldirpath)
    else:
        for dirName, subdirList, fileList in os.walk(infolder):
            for subdir in subdirList:
                if exclude is not None:
                    keep = 0
                    for exc in exclude:
                        keep += exc in subdir
                    if keep == 0:
                        subdirs.append(subdir)
                else:
                    subdirs.append(subdir)

    subdirs = np.sort(subdirs)
    return subdirs

# aims to return all RG tracks associated. 
def detect_experiment_tracks_RG( infolder, ext='.mat', include='filt2_'):

    """ Locate all paired red and green tracks within infolder/meantracks

    Assumes you have under infolder a meantracks folder
    infolder
        |___ meantracks

    Parameters
    ----------
    infolder : str
        top-level folder path
    ext : str  
        file extension we are looking for e.g. '.mat'
    include : str
        keyword the files in the subfolder will contain for the subfolder to be relevant.

    Returns
    -------
    red_files : numpy array
        numpy of all matched red track filepaths
    green_files : numpy array
        numpy of all matched green track filepaths 

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

    red_files = np.array(red_files).ravel()
    green_files = np.array(green_files).ravel()

    return red_files, green_files

def detect_files( folder, ext='.tif'):
    """ detects files of a given extension within a folder.

    note: does not detect files within subfolders within the folder. One can also use glob module.

    Parameters
    ----------
    folder : str
        path of folder containing desired files
    ext : str (default='.tif')
        name of file extension to search for e.g. '.jpg', '.png'

    Returns
    -------
    filepaths : numpy array
        array of matched full filepaths of files with the desired extension
    fnames : numpy array
        array of matched file names
        
    """
    import os 
    import numpy as np
    
    filepaths = []
    fnames = []
    
    for f in os.listdir(folder):
        if ext in f: 
            filepaths.append(os.path.join(folder, f))
            fnames.append(f)
            
    filepaths = np.hstack(filepaths)
    fnames = np.hstack(fnames)

    return filepaths, fnames

def save_multipage_tiff(np_array, savename):
    
    """ save numpy array of images as a multipage .tiff file 
    
    Parameters
    ----------
    np_array : numpy array
        (n_frames, n_rows, n_cols, n_channels) image 
    savename : str
        filepath to save the output .tif stack. 
    
    Returns
    -------
    void function
    
    """    
    from tifffile import imsave
    import numpy as np 
    
    if np_array.max() < 1.1:
        imsave(savename, np.uint8(255*np_array))
    else:
        imsave(savename, np.uint8(np_array))
    
    return [] 
    
