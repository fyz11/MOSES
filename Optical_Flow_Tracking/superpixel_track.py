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
# normal optical flow computation using cv2 in 2D. 
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
    
    from .Eval_dense_optic_flow import Eval_dense_optic_flow
    import numpy as np 
    
    nframes = vidstack.shape[0]
    flow = [] 
    
    for i in range(nframes-1):
        flow.append(Eval_dense_optic_flow(vidstack[i], vidstack[i+1], params)[None,:])
        
    return np.concatenate(flow, axis=0)

# optical flow with option of masking out bits where flow will not be evaluated. good for use for registered videos. 
def compute_vid_opt_flow_mask(vidstack, params, mask=None, adaptive=False, ksize=3):
    
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
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_dilation, binary_closing, disk
    from scipy.ndimage.morphology import binary_fill_holes
    
    nframes = vidstack.shape[0]
    flow = [] 
    
    for i in range(nframes-1):
        frame1 = vidstack[i].copy()
        frame2 = vidstack[i+1].copy()
        
        if adaptive:
            # infer areas to compute flow by computing the area of signal. 
            mask = binary_dilation(binary_fill_holes(binary_closing( frame1>= threshold_otsu(frame1), disk(ksize))), disk(ksize))
        
        flow_frame = Eval_dense_optic_flow(frame1, frame2, params)
        flow_frame_x = flow_frame[:,:,0].copy(); flow_frame_x[mask==0] = 0
        flow_frame_y = flow_frame[:,:,1].copy(); flow_frame_y[mask==0] = 0
        
        flow_frame[:,:,0] =  flow_frame_x.copy()
        flow_frame[:,:,1] =  flow_frame_y.copy()
        
        flow.append(flow_frame[None,:])
        
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

    return pos, mean_width_superpixel, spixels


# =============================================================================
#   Utility Functions used for track propagation
# =============================================================================

def count_point_density(poslist, spixelsmap, min_density=1):
    
    import numpy as np 
    
    poslist_ = poslist.astype(np.int)
    uniq_regs = np.unique(spixelsmap)
    
    posregs = spixelsmap[poslist_[:,0], poslist_[:,1]]
    
    posdensity, binspos = np.histogram(posregs, range=[0, uniq_regs[-1]], bins=len(uniq_regs))

    nondenseregs = uniq_regs[posdensity<min_density]
    
    return posdensity, nondenseregs    

    
def dense_tracks2array_tracks(poslist):
    
    import numpy as np
    
    n_frames = len(poslist)
    max_tracks = np.max([len(p) for p in poslist])
    trackarray = np.zeros( (max_tracks, n_frames, 2), dtype=np.int)
    
    # we populate the array backward. 
    for frame in range(n_frames-1, -1, -1):
        pos_frame = poslist[frame]
        n_points_frame = len(pos_frame)
        trackarray[:n_points_frame, frame] = pos_frame

        if n_points_frame < max_tracks and frame < n_frames -1:
            trackarray[n_points_frame:, frame] = trackarray[n_points_frame:, frame+1] # copy across. 
            
    return trackarray 
    

# =============================================================================
#   Propagation functions.
# =============================================================================

# uses vectorization to efficiently track regular superpixels over time, propagating their centroid positions according to the computed optical flow from frame to frame.
def prop_spixels_frame_by_frame_forward( spixel_pos, optflow, spixel_size):

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
    
    
def prop_spixels_frame_by_frame_backward( spixel_pos, optflow, spixel_size):

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

        flow_frame = optflow[nframes-1-i].copy()
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
        pos1 = pos0 - av_flow_pos # move backwards in position, reversing the flow.
        pos1 = (np.round(pos1)).astype(np.int)
        pos1[:,0] = np.clip(pos1[:,0], y_limits[0], y_limits[1]-1)
        pos1[:,1] = np.clip(pos1[:,1], x_limits[0], x_limits[1]-1)
        
        # 3. save out . 
        meantracks[:,i+1,:] = pos1.copy()
        
    return meantracks
    


def prop_spixels_frame_by_frame_forward_dense( spixel_pos, optflow, spixel_size, spixelsmap, mindensity=1):

    """
    Propagates each superpixel centroid according to the average flow within its area from frame to frame to generate superpixel tracks.
    
    Inputs:
    -------
    spixel_pos: (n_superpixels x 2) numpy array of superpixel centroids given as (y,x)-image coordinates
    optflow: (n_frames-1, n_rows, n_cols, 2) numpy array of frame-to-frame optical flow i.e. the output to above <compute_vid_opt_flow> function.
    spixel_size: estimated mean superpixel width assuming they are square, output to above <compute_regular_spixels> function.
    
    
    
    Output:
    -------
    meantracks: (n_points, n_frames, 2) numpy array
    
    """
    
    import itertools
    import numpy as np
    
    radius = int(np.floor((spixel_size)/2.))

    m, n = optflow.shape[1:-1]
    nframes = optflow.shape[0]
    nspixels = spixel_pos.shape[0]

    y_limits = [0, m]
    x_limits = [0, n]

    initial_pos = np.floor(spixel_pos).astype(np.int)

    # construct the save object of tracking.     
    meantracks = [] # initiate as blank
    meantracks.append(initial_pos) # this is the same as the initialisation of the initial pos. 
    
    
    # build the permutation matrix for iteration; (this is the size of the superpixel region.. and applies for all.)
    offset_matrix = [np.array([i,j]).ravel() for i,j in itertools.product(np.arange(-radius,radius+1), repeat=2)]
    offset_matrix = np.vstack(offset_matrix)

    
    # now we update everything in parallel. 
    for i in range(nframes)[:]: 
        
        # 1. compute the mean displacement vectors.
        pos0 = meantracks[i] # retrieve from the first frame. 
        
        spixel_range = offset_matrix
        expand_pos0 = pos0[:,None,:] + spixel_range[None,:] # n_pos x n_points x n_coords
        
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
        flow_frame = flow_frame[:,:,[1,0]] # switch x,y to (y,x) consistent with the image coordinates. 
        flow_pos_0 = flow_frame[expand_pos_0[:,:,0].ravel(), expand_pos_0[:,:,1].ravel(), :] #should give back number of superpixels. (retrieving the coordinates around the position.)
        flow_pos_0 = flow_pos_0.reshape(expand_pos_0.shape)
                                
        flow_pos_0 *= final_mask[:,:,None]

        # this can still be nan....        
        av_flow_pos = np.nanmedian(flow_pos_0, axis=1)
        av_flow_pos[np.isnan(av_flow_pos)] = 0 # no displacement
        
#        print i, av_flow_pos.min(), av_flow_pos.max(), np.mean(av_flow_pos)
        
        # 2. thresh  the last pos by the limits. 
        pos1 = pos0 + av_flow_pos
        pos1 = (np.round(pos1)).astype(np.int)
        pos1[:,0] = np.clip(pos1[:,0], y_limits[0], y_limits[1]-1)
        pos1[:,1] = np.clip(pos1[:,1], x_limits[0], x_limits[1]-1)
        
        
        # 3. check the density. 
        _, nondense_pos = count_point_density(pos1, spixelsmap, min_density=mindensity)
        
        if len(nondense_pos) > 0: 
            # if its not empty then. expand pos1. 
            add_pos = initial_pos[nondense_pos]
            pos1 = np.vstack([pos1, add_pos])
        
        # 4. update the tracking
        meantracks.append(pos1)
        
    # return the list of track positions. 
    meantracks = dense_tracks2array_tracks(meantracks)
    
    return meantracks
    
    
def prop_spixels_frame_by_frame_backward_dense( spixel_pos, optflow, spixel_size, spixelsmap, mindensity=1):

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

    initial_pos = np.floor(spixel_pos).astype(np.int)

    # construct the save object of tracking.     
    meantracks = [] # initiate as blank
    meantracks.append(initial_pos) # this is the same as the initialisation of the initial pos. 
    
    # build the permutation matrix for iteration;
    offset_matrix = [np.array([i,j]).ravel() for i,j in itertools.product(np.arange(-radius,radius+1), repeat=2)]
    offset_matrix = np.vstack(offset_matrix)

    
    # now we update everything in parallel. 
    for i in range(nframes)[:]: 
        
        # 1. compute the mean displacement vectors.
        pos0 = meantracks[i]
        
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

        final_mask = final_mask.astype(np.float)
        final_mask[final_mask==0] = np.nan # this is so x by number should give nan and we can exploit nansum

        flow_frame = optflow[nframes-1-i].copy()
        flow_frame = flow_frame[:,:,[1,0]]
        flow_pos_0 = flow_frame[expand_pos_0[:,:,0].ravel(), expand_pos_0[:,:,1].ravel(), :] #should give back number of superpixels. 
        flow_pos_0 = flow_pos_0.reshape(expand_pos_0.shape)
#        flow_pos_0 = flow_pos_0[:,:]
                                
        flow_pos_0 *= final_mask[:,:,None]

        # this can still be nan....        
        av_flow_pos = np.nanmedian(flow_pos_0, axis=1)
        av_flow_pos[np.isnan(av_flow_pos)] = 0 # no displacement
        
#        print i, av_flow_pos.min(), av_flow_pos.max(), np.mean(av_flow_pos)
        
        # 2. thresh  the last pos by the limits. 
        pos1 = pos0 - av_flow_pos # move backwards in position, reversing the flow.
        pos1 = (np.round(pos1)).astype(np.int)
        pos1[:,0] = np.clip(pos1[:,0], y_limits[0], y_limits[1]-1)
        pos1[:,1] = np.clip(pos1[:,1], x_limits[0], x_limits[1]-1)
        
        # 3. check the density. 
        _, nondense_pos = count_point_density(pos1, spixelsmap, min_density=mindensity)
        
        if len(nondense_pos) > 0: 
            # if its not empty then. expand pos1. 
            add_pos = initial_pos[nondense_pos]
            pos1 = np.vstack([pos1, add_pos])
        
        # 4. update the tracking
        meantracks.append(pos1)
        
    # return the list of track positions. 
    meantracks = dense_tracks2array_tracks(meantracks) # o true .. we have to match the two somewhat... 
        
    return meantracks


# =============================================================================
#   These are the boilerplate scripts in order to efficiently batch process the entire video. 
# =============================================================================
    
# forward tracks only for the channel
def compute_grayscale_vid_superpixel_tracks(vidstack, opt_flow_params, n_spixels=1000, mask=None, adaptive_mask=False, ksize_mask=3, direction='F', dense=False, mindensity=1):
    
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
    
    if mask is None:
        optflow_ = compute_vid_opt_flow(vidstack, opt_flow_params)    
    else:
        optflow_ = compute_vid_opt_flow_mask(vidstack, opt_flow_params, mask=mask, adaptive_mask=adaptive_mask, ksize_mask=ksize_mask)
        
    pos0, avDim, spixels = compute_regular_spixels((n_rows,n_cols), n_spixels=n_spixels) # this can be made faster without using SLIC!.
    
    if direction == 'F':
        if dense:
            meantracks = prop_spixels_frame_by_frame_forward_dense( pos0, optflow_, avDim, spixels, mindensity=mindensity)
        else:
            meantracks = prop_spixels_frame_by_frame_forward( pos0, optflow_, avDim)
    
    if direction == 'B':
        if dense:
            meantracks = prop_spixels_frame_by_frame_backward_dense( pos0, optflow_, avDim, spixels, mindensity=mindensity)
        else:
            meantracks = prop_spixels_frame_by_frame_backward(pos0, optflow_, avDim)
        
    return optflow_, meantracks


# forward + backward computation with optional mask. 
def compute_grayscale_vid_superpixel_tracks_FB(vidstack, opt_flow_params, n_spixels=1000, mask=None, adaptive_mask=False, ksize_mask=3, dense=False, mindensity=1):
    
    """
    Combines above functionalities into one easy to use function to compute the tracks forward and backward. 
    
    Inputs:
    -------
    vidstack: (n_frames x n_rows x n_cols) numpy array representing a gray-scale movie
    opt_flow_params: optical flow parameters for farneback optical flow implementation in opencv.
    n_spixels: number of regular superpixels to decompose images.
    
    Output:
    -------
    meantracks_F: (n_superpixels, n_frames, 2) numpy array of forward tracks.
    meantracks_B: (n_superpixels, n_frames, 2) numpy array of backward tracks. 

    """
    _, n_rows, n_cols = vidstack.shape
    
    if mask is None:
        optflow_ = compute_vid_opt_flow(vidstack, opt_flow_params)    
    else:
        optflow_ = compute_vid_opt_flow_mask(vidstack, opt_flow_params, mask=mask, adaptive=adaptive_mask, ksize=ksize_mask)
    
    pos0, avDim, spixels = compute_regular_spixels((n_rows,n_cols), n_spixels=n_spixels)
    
    print ('computing forward tracks ...')
    if dense:
        meantracks_F = prop_spixels_frame_by_frame_forward_dense( pos0, optflow_, avDim, spixels, mindensity=mindensity)
    else:
        meantracks_F = prop_spixels_frame_by_frame_forward( pos0, optflow_, avDim)
    
    print ('computing backward tracks ...')
    if dense:
        meantracks_B = prop_spixels_frame_by_frame_backward_dense( pos0, optflow_, avDim, spixels, mindensity=mindensity)
    else:
        meantracks_B = prop_spixels_frame_by_frame_backward( pos0, optflow_, avDim)
    
    return meantracks_F, meantracks_B
  
    
