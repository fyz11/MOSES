# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 23:25:21 2018

@author: felix
"""
import numpy as np

# a normal kymograph projection. 
def kymograph_img(vid, pos=None, axis=1, proj_fn=None):
    """ Given a video create a kymograph with time by collapsing out one of the given spatial dimensions as give by 'axis' using the given projection function, 'proj_fn' or by taking a slice if 'pos' is defined

    Parameters
    ----------
    vid : numpy array
        a (n_frames, n_rows, n_cols) or (n_frames, n_rows, n_cols, 3) or (n_z, n_y, n_x) image
    pos : None or int
        the slice number along the given 'axis' to take image slice instead of using proj_fn
    axis : int 
        the image axis to aggregate information over (in order to collapse the image dimensionality)
    proj_fn : None or numpy function
        function to aggregate image information along one axis, if None, defaults to np.mean
    
    Returns
    -------
    kymograph_img : numpy array
        the projected image with time along the image rows and a spatial dimension along the image columns

    """
    import numpy as np 
    
    if pos is None:    
        if proj_fn is None:
            kymograph_img = np.mean(vid, axis=axis)
            return kymograph_img
        else:
            kymograph_img = proj_fn(vid, axis=axis)
            return kymograph_img
    else:
        if axis == 0:
            kymograph_img = vid[pos]
            return kymograph_img
        if axis == 1:
            kymograph_img = vid[:,pos]
            return kymograph_img
        if axis == 2:
            kymograph_img = vid[:,:,pos]
            return kymograph_img
        
#==============================================================================
# Below are the extensions of kymographs to superpixels.... 
# to do: make it more general to work with histograms (current) + masks!
#==============================================================================
def construct_spatial_time_MOSES_velocity_x(tracks, shape, n_samples=51, axis=1, proj_fn=np.nanmedian):
    """ Compute the velocity kymograph given computed superpixel tracks collapsing out one of the given spatial dimensions as give by 'axis' using the given projection function, 'proj_fn' or by taking a slice if 'pos' is defined

    n_samples is used to bin across the designated spatial axis in order to create an image from the unordered nature of the (x,y) points

    Parameters
    ----------
    tracks : numpy array
        a (n_spixels, n_frames, 2) array of (y,x) coordinates giving the superpixel centroid
    shape : array-like
        a tuple (n_rows, n_cols) giving the dimensions of the original image
    n_samples : int
        the number of values to divide the axis into to aggregate statistics
    axis : int 
        the image axis to aggregate information over (in order to collapse the image dimensionality)
    proj_fn : None or numpy function
        function to aggregate image information along one axis, if None, defaults to np.nanmedian which handles NaN
    
    Returns
    -------
    kymograph_img : numpy array
        the projected image of size (n_frames-1, n_samples-1) with time along the image rows and a spatial dimension along the image columns

    """
    import numpy as np 
    nspixels, nframes, _ = tracks.shape
    velocity_tracks = tracks[:,1:] - tracks[:,:-1]
    velocity_tracks = velocity_tracks.astype(np.float)
#    pos_x = np.mean(tracks, axis=1)[:,1] # take the x positions
    x_sampling = np.linspace(0, shape[1]-1, n_samples+1)
    res_map = np.zeros((nframes-1, len(x_sampling)-1))

#    for x in unique_x:
    for t in range(nframes-1):
        for i in range(len(x_sampling)-1):
            pos_x = tracks[:,t,1]
            select = np.logical_and(pos_x>=x_sampling[i], pos_x<x_sampling[i+1])
            moses = velocity_tracks[select, t , axis]
            nonzero = np.abs(moses) > 1e-8
            # this line is required for dense tracking to avoid biasing the statistics towards 0 causing aliasing artifacts.
            if np.sum(nonzero) > 0:
                av_moses = proj_fn(moses[nonzero], axis=0)
            else:
                av_moses = 0
            if np.sum(select) > 0:
                res_map[t,i] = av_moses
#            else:
#                res_map.append(np.zeros(nframes-1))
    kymograph_img = np.array(res_map)

    return kymograph_img
    
    
def construct_spatial_time_MOSES_plot_x(tracks, moses_spatial_time, shape=None, n_samples=51, filt_outliers=False, min_points=5, axis=1, proj_fn=np.nanmean):
    """ Compute the kymograph of the MOSES mesh strain of individual superpixels given the computed superpixel tracks collapsing out one of the given spatial dimensions as give by 'axis' using the given projection function, 'proj_fn' or by taking a slice if 'pos' is defined

    n_samples is used to bin across the designated spatial axis in order to create an image from the unordered nature of the (x,y) points of the superpixel track positions

    Parameters
    ----------
    tracks : numpy array
        a (n_spixels, n_frames, 2) array of (y,x) coordinates giving the superpixel centroid
    moses_spatial_time : numpy array
        (n_frames x n_superpixels) array specifying the average neighbourhood mesh strain in each frame e.g. as generated by :meth:`MOSES.Motion_Analysis.mesh_statistics_tools.construct_MOSES_mesh`
    shape : array-like
        a tuple (n_rows, n_cols) giving the dimensions of the original image
    n_samples : int
        the number of values to divide the axis into to aggregate statistics
    filt_outliers : bool 
        if True, tests whether each bin at each frame possesses the minimum number of superpixel positions given by min_points to get an estimate.
    min_points : int 
        the minimum number of points in the selectd bin 
    axis : int 
        the image axis to aggregate information over (in order to collapse the image dimensionality)
    proj_fn : None or numpy function
        function to aggregate image information along one axis, if None, defaults to np.nanmedian which handles NaN
    
    Returns
    -------
    kymograph_img : numpy array
        the projected image of size (n_frames-1, n_samples-1) with time along the image rows and a spatial dimension along the image columns

    """
    import numpy as np 
    
    nframes, nspixels = moses_spatial_time.shape
    x_sampling = np.linspace(0, shape[1]-1, n_samples+1)
 
    res_map = np.zeros((nframes, len(x_sampling)-1))

    for t in range(nframes):
        for i in range(len(x_sampling)-1):
            pos_x = tracks[:,t,axis]
            select = np.logical_and(pos_x>=x_sampling[i], pos_x<x_sampling[i+1])
            moses = moses_spatial_time[t, select]
            nonzero = np.abs(moses) > 1e-8

            if np.sum(nonzero) > 0:
                av_moses = proj_fn(moses[nonzero], axis=0)
            else:
                av_moses = 0
                
            if filt_outliers:
                if np.sum(select) > min_points:
                    res_map[t,i] = av_moses
#                elif np.sum(select) > 0 and np.sum(select) <= min_points:
#                    res_map[t,i] = np.nanmedian(res_map[np.max(t-1,0)])
            else:   
                if np.sum(select) > 0:
                    res_map[t,i] = av_moses

    return np.array(res_map)
