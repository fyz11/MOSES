# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 15:31:11 2014

@author: felix
@email: felixzhou1@gmail.com

This script and all the code within is licensed under the Ludwig License.

see README.md for details on usage.

"""

#==============================================================================
# Snippet to suppress deprecation warnings.
#==============================================================================
import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
    
    
#==============================================================================
# Define some helper functions that might also be used as separate library functions
#==============================================================================

def compute_vid_opt_flow(vidstack, params):
    
    """
    Given a grayscale movie as a numpy vidstack, with the specified optical farneback params as params return the stacked optical flow (frame by frame) for the entire video.
    
    Input:
    ------
    vidstack: n_frames x n_rows x n_cols numpy array 
    params: OpenCV opticalflowfarneback parameters (python dict) c.f. http://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html 
    
    Output:
    -------
    (n_frames-1, n_rows x n_cols x 2) numpy array giving the optical flow
    
    """
    
    from Eval_dense_optic_flow import Eval_dense_optic_flow
    import numpy as np 
    
    nframes = vidstack.shape[0]
    flow = [] 
    
    for i in range(nframes-1):
        flow.append(Eval_dense_optic_flow(vidstack[i], vidstack[i+1], params)[None,:])
        
    return np.concatenate(flow, axis=0)
    
    
def compute_regular_spixels(imgsize, n_spixels=1000):
    
    """
    
    This is a function exploiting the SLIC implementation within Scikit-Image in order to quickly tile an image with regular superpixels. It does this by running slic on a blank image the same size as the desired image.
    
    Inputs
    ------
    imgsize: (n_rows x n_cols) tuple
    n_spixels: int specifying number of regular partitions of the image.
    
    Outputs
    -------
    pos: mean positions of superpixels (n_superpixels x 2) numpy array
    mean_width_superpixel: int specifying width of an average superpixel.
    
    """
    # feedback the areas of the spixels 
    from skimage.segmentation import slic 
    import numpy as np
    
    m, n = imgsize
    blank = np.zeros((m,n))
    
    spixels = slic(blank, n_segments=n_spixels)
    Y, X = np.meshgrid(np.arange(n), np.arange(m))
    
    positions_x = []
    positions_y = []
    areas = []
    regions = np.unique(spixels)
    
    for reg in regions:
        positions_x.append(np.mean(X[spixels==reg]))
        positions_y.append(np.mean(Y[spixels==reg]))
        areas.append(np.sum(spixels==reg))
    
    # concatenate the positions inta large vector in the form (y,x)
    positions_x = np.array(positions_x).ravel()
    positions_y = np.array(positions_y).ravel()
        
    pos = np.vstack([positions_x, positions_y]).T
    mean_width_superpixel = np.sqrt(np.mean(areas))

    return pos, mean_width_superpixel
    

# uses vectorization to efficiently track regular superpixels over time, propagating their centroid positions according to the computed optical flow from frame to frame.
def prop_spixels_frame_by_frame( spixel_pos, optflow, spixel_size):

    """
    Propagates each superpixel centroid according to the average flow within its area from frame to frame to generate superpixel tracks.
    
    Inputs:
    -------
    spixel_pos: (n_superpixels x 2) numpy array of superpixel centroids given as (y,x)-image coordinates
    optflow: (n_frames-1, n_rows, n_cols, 2) numpy array of frame-to-frame optical flow i.e. the output to above <compute_vid_opt_flow> function.
    spixel_size: estimated mean superpixel width assuming they are square, output to above <compute_regular_spixels> function.
    
    Output:
    -------
    meantracks: (n_superpixels, n_frames, 2) numpy array
    
    """
    
    import itertools
    import numpy as np
    
    radius = int(np.floor((spixel_size)/2.))

    m, n = optflow.shape[1:-1]
    nframes = optflow.shape[0]
    nspixels = spixel_pos.shape[0]

    y_limits = [0, m]
    x_limits = [0, n]

    meantracks = np.zeros((nspixels, nframes+1, 2), dtype=np.int)
    meantracks[:,0,:] = np.floor(spixel_pos).astype(np.int)
    
    # build the permutation matrix for iteration;
    offset_matrix = [np.array([i,j]).ravel() for i,j in itertools.product(np.arange(-radius,radius+1), repeat=2)]
    offset_matrix = np.vstack(offset_matrix)

    
    # now we update everything in parallel. 
    for i in range(nframes)[:]: 
        
        # 1. compute the mean displacement vectors.
        pos0 = meantracks[:,i,:]
        
        spixel_range = offset_matrix
        expand_pos0 = pos0[:,None,:] + spixel_range[None,:] # n_superpixels x n_points x n_coords
        
        # thresh and mask points that do not fit within the image. 
        expand_pos_0 = (np.floor(expand_pos0)).astype(np.int)
        
#        print expand_pos_0[:,:,0].min(), expand_pos_0[:,:,0].max()
        mask_pos_0_y = np.logical_and(expand_pos_0[:,:,0] >=y_limits[0], expand_pos_0[:,:,0] <=y_limits[1]-1)
        mask_pos_0_x = np.logical_and(expand_pos_0[:,:,1] >=x_limits[0], expand_pos_0[:,:,1] <=x_limits[1]-1)

        final_mask = (np.logical_and(mask_pos_0_y, mask_pos_0_x)).astype(np.int) # this should give 1 where things are valid

        # first apply mask to expanded pos        
        expand_pos_0 *= final_mask[:,:,None]
#        print expand_pos_0.shape    

        final_mask = final_mask.astype(np.float)
        final_mask[final_mask==0] = np.nan # this is so x by number should give nan and we can exploit nansum

        flow_frame = optflow[i].copy()
        flow_frame = flow_frame[:,:,[1,0]]
        flow_pos_0 = flow_frame[expand_pos_0[:,:,0].ravel(), expand_pos_0[:,:,1].ravel(), :] #should give back number of superpixels. 
        flow_pos_0 = flow_pos_0.reshape(expand_pos_0.shape)
#        flow_pos_0 = flow_pos_0[:,:]
                                
        flow_pos_0 *= final_mask[:,:,None]

        # this can still be nan....        
        av_flow_pos = np.nanmean(flow_pos_0, axis=1)
        av_flow_pos[np.isnan(av_flow_pos)] = 0 # no displacement
        
#        print i, av_flow_pos.min(), av_flow_pos.max(), np.mean(av_flow_pos)
        
        # 2. thresh  the last pos by the limits. 
        pos1 = pos0 + av_flow_pos
        pos1 = (np.round(pos1)).astype(np.int)
        pos1[:,0] = np.clip(pos1[:,0], y_limits[0], y_limits[1]-1)
        pos1[:,1] = np.clip(pos1[:,1], x_limits[0], x_limits[1]-1)
        
        # 3. save out . 
        meantracks[:,i+1,:] = pos1.copy()
        
    return meantracks
    
    
def compute_grayscale_vid_superpixel_tracks(vidstack, opt_flow_params, n_spixels=1000):
    
    """
    Combines above functionalities into one easy to use function
    
    Inputs:
    -------
    vidstack: (n_frames x n_rows x n_cols) numpy array representing a gray-scale movie
    opt_flow_params: optical flow parameters for farneback optical flow implementation in opencv.
    n_spixels: number of regular superpixels to decompose images.
    
    Output:
    -------
    meantracks: (n_superpixels, n_frames, 2) numpy array of tracks.

    """
    _, n_rows, n_cols = vidstack.shape
    
    optflow_ = compute_vid_opt_flow(vidstack, opt_flow_params)
    pos0, avDim = compute_regular_spixels((n_rows,n_cols), n_spixels=n_spixels)
    meantracks = prop_spixels_frame_by_frame( pos0, optflow_, avDim)
    
    return meantracks

    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


