# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 23:25:21 2018

@author: felix
"""
import numpy as np

# a normal kymograph projection. 
def kymograph_img(vid, pos=None, axis=1, proj_fn=None):
    import numpy as np 
    
    if pos is None:    
        if proj_fn is None:
            return np.mean(vid, axis=axis)
        else:
            return proj_fn(vid, axis=axis)
    else:
        if axis == 0:
            return vid[pos]
        if axis == 1:
            return vid[:,pos]
        if axis == 2:
            return vid[:,:,pos]
        
        
"""
To Do: Kymograph using masks and erosion to.....
""" 
#==============================================================================
# Below are the extensions of kymographs to superpixels.... 
# to do: make it more general to work with histograms (current) + masks!
#==============================================================================

def construct_spatial_time_MOSES_velocity_x(tracks, shape=None, n_samples=51, axis=1, proj_fn=np.nanmedian):
    
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
    return np.array(res_map)  
    
    
def construct_spatial_time_MOSES_plot_x(tracks, moses_spatial_time, shape=None, n_samples=51, filt_outliers=False, min_points=5, proj_fn=np.nanmean):
    
    import numpy as np 
    
    nframes, nspixels = moses_spatial_time.shape
    x_sampling = np.linspace(0, shape[1]-1, n_samples+1)
 
    res_map = np.zeros((nframes, len(x_sampling)-1))

    for t in range(nframes):
        for i in range(len(x_sampling)-1):
            pos_x = tracks[:,t,1]
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
