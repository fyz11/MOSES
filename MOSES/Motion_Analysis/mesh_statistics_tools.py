#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 01:40:16 2017

@author: felix
@email: felixzhou1@gmail.com

This script and all the code within is licensed under the Ludwig License.

see README.md for details on usage.

"""

from ..Visualisation_Tools.mesh_visualisation import visualise_mesh
import numpy as np 

#==============================================================================
#  Compute MOSES mesh and the motion stability index.
#==============================================================================
def construct_MOSES_mesh(tracks, dist_thresh, spixel_size):
    """ constructs the MOSES mesh, linking neighbours defined by the initial positions of the superpixels.
    
    Parameters
    ----------
    tracks : numpy array
        (n_superpixels, n_frames, 2), numpy array of superpixel tracks.
    dist_thresh : float
        radial distance cut-off as a multiplier of the average superpixel width.
    spixel_size : float
        average superpixel width.
    
    Returns
    -------
    MOSES_mesh_strain_time : numpy array
        (n_frames x n_superpixels), specifying the average neighbourhood mesh strain of each superpixel in each frame.
    nearest_neighbors_refined : list of numpy arrays
        adjacency graph of how superpixels are connected (only found from initial points therefore is defined with the one timepoint at frame 0)
    
    """
    
    from sklearn.metrics.pairwise import pairwise_distances
    import numpy as np 
    
    # for first frame determine the partners. 
    first_frame = tracks[:,0,:]    
    n_regions = first_frame.shape[0] 
    n_frames = tracks.shape[1]
    dist_thresh = dist_thresh*spixel_size

    dist_regions = pairwise_distances(first_frame, metric='euclidean')
    dist_regions[dist_regions>dist_thresh] = np.inf #set to ridiculously high   
    # sort the regions 
    nearest_neighbors = np.argsort(dist_regions, axis=1)
    
    # iterate and for each one, remove those that have inf. 
    nearest_neighbors_refined = []
    
    for i in range(n_regions):
            
        good_index = []
        for j in range(1, n_regions):

            if dist_regions[i, nearest_neighbors[i][j]] != np.inf:
                good_index.append(nearest_neighbors[i,j])
            else:
                break
            
        nearest_neighbors_refined.append(np.array(good_index).ravel())
        
    # now for each superpixel we can compute the average distance to its neighbours. 
    avg_distance_neighbours = []

    for j in range(n_regions):
        if len(nearest_neighbors_refined[j])>0:
            equilibrium_dist = dist_regions[j, nearest_neighbors_refined[j]] # find the normal distances. 
            paired_diff = tracks[j,:,:][None,:] - tracks[nearest_neighbors_refined[j],:,:] # should give n_neighbours x n_frames x 2
            paired_diff = np.sqrt(paired_diff[:,:,0]**2 + paired_diff[:,:,1]**2)
            paired_diff_tension = np.abs(paired_diff - equilibrium_dist[:,None]) # do we want the abs? 
            avg_distance_neighbours.append(np.nanmean(paired_diff_tension, axis=0))
        else:
            paired_diff_tension = np.zeros((1, n_frames))
            paired_diff_tension[:] = np.nan
            avg_distance_neighbours.append(paired_diff_tension)
        
    MOSES_mesh_strain_time = np.vstack(avg_distance_neighbours)
            
    return MOSES_mesh_strain_time.T, nearest_neighbors_refined


def construct_mesh_strain_vector(tracks, neighbours):
    """ Given an arbitrary graph defined by a neighbour list, constructs the mesh strain vector for all time

    Parameters
    ----------
    tracks : numpy array
        (n_superpixels, n_frames, 2), numpy array of superpixel tracks.
    neighbours : list of numpy arrays
        list of superpixel neighbours, the neighbours for n_superpixels over n_frames. If len() of list = 1 and n_frames>1 then we assume the neighbour is fixed and propagate for all time.

    Returns
    -------
    mesh_strain_time_vector : numpy array
        (n_frames x n_superpixels x 2), giving the mean mesh strain vector of each superpixel relative to its neighbours at each time point.
    
    """
    
    import numpy as np 

    n_regions, n_frames, _ = tracks.shape

    if len(neighbours) == 1:
        pass
    else: 
        if len(neighbours)!= n_frames:
            raise Exception('number of neighbour lists should equal number of frames or 1')

    # now for each superpixel we can compute the average distance to its neighbours. 
    mesh_strain_time_vector = []
    
    for i in range(n_frames):

        mesh_strain_frame = [] # initialise.
        if len(neighbours)==1:
            neighbours_frame = neighbours[0]
        else:
            neighbours_frame = neighbours[i]
        for j in range(n_regions):
            effective_strain = np.mean( tracks[neighbours_frame[j], i, :] - tracks[j,i,:][None,:] , axis=0)
            mesh_strain_frame.append(effective_strain)
            
        mesh_strain_time_vector.append(np.array(mesh_strain_frame))
        
    mesh_strain_time_vector = np.array(mesh_strain_time_vector)
            
    return mesh_strain_time_vector
    
    
def compute_MOSES_mesh_strain_curve(mesh_strain_time, normalise=False):
    """ Computes the MOSES mesh strain (normalized/unnormalized) across superpixels for each time. 
    
    Parameters
    ----------
    mesh_strain_time : numpy array
        (n_frames, n_superpixels), the strain of each superpixel at each frame.
    normalise : bool
        if 'true' divide the mesh strain curve by the maximum value, if false return the raw values.
    
    Returns
    --------
    MOSES_mesh_strain_curve : numpy array
        (n_frames,) numpy array vector giving the moshes mesh strain (normalised if the normalise flag is set to true)
    
    """
    import numpy as np 
    
    if normalise:
        
        MOSES_mesh_strain_curve = np.nanmean(mesh_strain_time, axis=-1)
        return MOSES_mesh_strain_curve / np.max(MOSES_mesh_strain_curve)
        
    else:
        
        MOSES_mesh_strain_curve = np.nanmean(mesh_strain_time, axis=-1)
        return MOSES_mesh_strain_curve
    

    
def from_neighbor_list_to_graph(tracks, neighbourlist, frame):
    """ Converts the adjacency graph definition from a list of neighbours of superpixels to a networkx graph. 
    
    Parameters
    ----------
    tracks : numpy array
        (n_superpixels, n_frames, 2), numpy array of superpixel tracks
    neighbourlist : list of numpy arrays
        (list of numpy arrays), giving the various neighbours to superpixels
    frame : int
        which frame fo which to construct the graph for visualisation.
    
    Returns
    -------
    G : networkx graph
        networkx graph representation of the neighbourhood relationships.
    
    """
    import networkx as nx
    import numpy as np 
    
    points_frame = tracks[:, frame, :] #(y,x)
    
    # we can easily transfer to network x if we can build a graph. 
    lines = []

    for i in range(len(neighbourlist)):
        neighbors = neighbourlist[i]
        point = points_frame[i,:]
        neighbor_points = points_frame[neighbors, :]

        dist_points = neighbor_points - point[None,:]        
        dist_points = np.sqrt(dist_points[:,0]**2 + dist_points[:,1]**2)
        
        for j in range(len(neighbors)):
            
            lines.append((i, neighbors[j], {'weight':dist_points[j]}))
            
    G=nx.from_edgelist(lines)
            
    return G
    

def compute_MOSES_mesh_stability_index(MOSES_mesh_strain_time_r, MOSES_mesh_strain_time_g, last_frames=24):
    
    """ Compute the mesh stability index and return the normalised mesh strain curve for the whole video.
    
    Parameters
    ----------
    MOSES_mesh_strain_time_r : numpy array
        (n_frames, n_superpixels), numpy array of strain of each superpixel at each frame for the red channel
    MOSES_mesh_strain_time_g : numpy array
        (n_frames, n_superpixels), numpy array of strain of each superpixel at each frame for the green channel
    last_frames : int
        (int) the number of frames interval for averaging and computing the motion stability index.
    
    Returns
    -------
    mesh_stability_index : float
        1 - normalised gradient where gradient is computed from the average difference over the last last_frames frames.
    norm_mean_strain : numpy array
        normalised mesh strain curve, the average of the individual divided by max value over the time interval.
    
    """
    
    import numpy as np 
    
    mean_strain_r = MOSES_mesh_strain_time_r.mean(axis=-1)
    mean_strain_g = MOSES_mesh_strain_time_g.mean(axis=-1)
    
    av_mean_strain = (mean_strain_r + mean_strain_g) / 2.
    norm_mean_strain = av_mean_strain / float(np.max(av_mean_strain))
    
    dx_eff = 1./len(norm_mean_strain)
    grad = norm_mean_strain[-last_frames:] - norm_mean_strain[-last_frames-1:-1]
    grad = np.mean(np.abs(grad))
    grad /= dx_eff
    
    mesh_stability_index = 1. - grad
    
    return mesh_stability_index, norm_mean_strain
    
#==============================================================================
# Additional mesh implementations
#==============================================================================
def construct_radial_neighbour_mesh(tracks, dist_thresh, spixel_size, use_counts=False):
    """ constructs the radial neighbours mesh, linking a superpixel to any superpixels separated by a maximum distance < dist_thresh * spixel_size 

    Parameters
    ----------
    tracks : numpy array
        (n_superpixels, n_frames, 2), numpy array of superpixel tracks.
    dist_thresh : float
        radial distance cut-off as a multiplier of the average superpixel width.
    spixel_size : float
        average superpixel width.
    use_counts : bool
        if True returns the number of neighbours over time instead of the strain
    
    Returns
    -------
    mesh_strain_time : numpy array
        (n_frames x n_superpixels) array specifying either the average neighbourhood distances of each superpixel in each frame (if use_counts=False) or the number of neighbours of each superpixel in each frame
    nearest_neighbours_time : list of list of numpy arrays
        adjacency graphs of how superpixels are connected over time
    
    """

    from sklearn.neighbors import BallTree
    import numpy as np 
    
    # ignore for now the dist_thresh:
    n_frames = tracks.shape[1]
    n_spixels = tracks.shape[0]

    if use_counts==True:
        mesh_strain_time = np.zeros((n_spixels, n_frames), dtype=np.int)
    else:
        mesh_strain_time = np.zeros((n_spixels, n_frames))
        
    nearest_neighbours_time = []

    for frame in range(n_frames):
        # prefilter the tracks, find the tracks whose coordinates are on the boundaries.    
        points = np.squeeze(tracks[:,frame,:])
        # select = np.logical_and(np.logical_and(points[:,1] > 0, points[:,0] > 0), np.logical_and(points[:,1] < limits[0]-1, points[:,0] < limits[1]-1))
        # bad = np.logical_not(select)
        tree = BallTree(points, leaf_size=30)
        ind, dist = tree.query_radius(points, r = dist_thresh, count_only = False, return_distance = True) # return distance.        
        # remove self from neighbours.
        ind = [np.setdiff1d(ind[ii], ii)  for ii in range(len(ind))]

        nearest_neighbours_time.append(ind)
        # report the number of neighbours.
        if use_counts == True:        
            num_neighbour_counts = np.array([len(ind_) for ind_ in ind]).ravel()
            # num_neighbour_counts[bad] = np.median(num_neighbour_counts[select])
            # save counts into array.
            mesh_strain_time[:, frame] = num_neighbour_counts
        else:
            mean_neighbour_distance = np.hstack([np.nanmean(d) for d in dist])
            # save counts into array.
            mesh_strain_time[:, frame] = mean_neighbour_distance

    return mesh_strain_time.T, nearest_neighbours_time


def construct_knn_neighbour_mesh(tracks, k=4):
    """ constructs the K nearest neighbours mesh, linking a superpixel to its closest K superpixels

    Parameters
    ----------
    tracks : numpy array
        (n_superpixels, n_frames, 2), numpy array of superpixel tracks.
    k : int 
        the number of closest superpixels to link to
    
    Returns
    -------
    mesh_strain_time : numpy array
        (n_frames x n_superpixels) specifying the average neighbourhood distances of each superpixel at each frame
    nearest_neighbours_time : list of list of numpy arrays
        adjacency graphs of how superpixels are connected over time
    
    """
    from sklearn.neighbors import BallTree
    import numpy as np 
    
    # ignore for now the dist_thresh:
    n_frames = tracks.shape[1]
    n_spixels = tracks.shape[0]

    mesh_strain_time = np.zeros((n_spixels, n_frames))
    nearest_neighbours_time = []

    for frame in range(n_frames):
        # prefilter the tracks, find the tracks whose coordinates are on the boundaries.    
        points = np.squeeze(tracks[:,frame,:])
        # select = np.logical_and(np.logical_and(points[:,1] > 0, points[:,0] > 0), np.logical_and(points[:,1] < limits[0]-1, points[:,0] < limits[1]-1))
        # bad = np.logical_not(select)
        tree = BallTree(points, leaf_size=30)
        ind, dist = tree.query_radius(points, k = k + 1, count_only = False, return_distance = True) # return distance.        
        # remove self from neighbours.
        ind = [np.setdiff1d(ind[ii], ii)  for ii in range(len(ind))]

        nearest_neighbours_time.append(ind)
        mean_neighbour_distance = np.hstack([np.nanmean(d) for d in dist])
        # save counts into array.
        mesh_strain_time[:, frame] = mean_neighbour_distance

    return mesh_strain_time.T, nearest_neighbours_time

    
#==============================================================================
# Boundary formation tools
#==============================================================================
def count_dynamic_nn_neighbors(tracks, dist_thresh, limits):
    
    """ Uses an efficient radiusBallTree structure in sklearn to efficiently compute the number of neighbours to each superpixel efficiently for each frame.
    
    Parameters
    ----------
    tracks : numpy array
        (n_superpixels, n_frames, 2) numpy array of superpixel tracks.
    dist_thresh : float
        distance cut-off to count number of neighbours for each superpixel. 
    limits : tuple
        (n_rows, n_cols), image shape size, used to correct for edge artifacts. 
    
    Returns
    -------
    num_neighbours_time : numpy array
        (n_superpixels, n_frames) numpy array, giving the total number of counts it is near. 
    
    """
    from sklearn.neighbors import BallTree
    import numpy as np 
    
    # ignore for now the dist_thresh:
    n_frames = tracks.shape[1]
    n_spixels = tracks.shape[0]

    num_neighbours_time = np.zeros((n_spixels, n_frames), dtype=np.int)

    for frame in range(n_frames):
        # prefilter the tracks, find the tracks whose coordinates are on the boundaries.    
        points = np.squeeze(tracks[:,frame,:])
        select = np.logical_and(np.logical_and(points[:,1] > 0, points[:,0] > 0), np.logical_and(points[:,1] < limits[0]-1, points[:,0] < limits[1]-1))
        bad = np.logical_not(select)
        
        tree = BallTree(points, leaf_size=30)
        ind, dist = tree.query_radius(points, r = dist_thresh, count_only = False, return_distance = True)         
                
        num_neighbour_counts = np.array([len(ind_[1:]) for ind_ in ind]).ravel()
        num_neighbour_counts[bad] = np.median(num_neighbour_counts[select])
        
        # save back into array.
        num_neighbours_time[:, frame] = num_neighbour_counts

    return num_neighbours_time


def compute_motion_saliency_map(meantracks, dist_thresh, shape, max_frame=None, filt=0, filt_size=5):
    
    """ Compute the saliency map based on counting the number of neighbours in the radius of each superpixel, then binning counts over regular grid of superpixels to produce a kernel density estimate and average smooth the images (if specified)
    
    Parameters
    ----------
    meantracks : numpy array
        (n_superpixels x n_frames x 2), numpy array of superpixel tracks
    map_vals : numpy array
        (n_superpixels x n_frames), the temporal mesh strain
    shape : tuple-like 
        (n_rows x n_cols), image shape
    max_frame : int
        the final frame to which we aggregate the statistic and compute maps for.
    filt : int
        (0 or 1), run over the heat map with an average filter or not. if 1, filt_size sets the kernel size for smoothing.
    filt_size : float
        (float), sigma setting the standard deviation of the gaussian for kernel smoothing.
    
    Returns
    -------
    final_saliency_map : numpy array
        (n_rows x n_cols), heatmap of saliency 
    spatial_time_saliency_map : numpy array
        (n_frames x n_rows x n_cols), frame by frame heatmap of saliency.
    
    """
    import numpy as np 
    from skimage.segmentation import slic, relabel_sequential 
    from skimage.morphology import square
    import cv2
    
    # compute the saliency map
    map_vals = count_dynamic_nn_neighbors(meantracks, dist_thresh, limits=shape)
    
    # coerce meantracks into the intended shape 
    meantracks = np.transpose(meantracks,(1,0,2))
    nframes, nregions, _ = meantracks.shape
    
    # create superpixel map 
    nrows, ncols = shape
    
    # Create the superpixel canvas we will aggregate over. 
    nothing = np.zeros((nrows, ncols, 3))
    spixels = slic(nothing, n_segments=nregions)
    spixels, _,_ = relabel_sequential(spixels)	
 
    regs = np.unique(spixels)
    # iterate over all the tracks and collate positions. 
    all_positions  = np.zeros((nrows, ncols))

    if max_frame:
        end_frame = max_frame
    else:
        end_frame = nframes
    
    spatial_time_saliency_map = []
    
    for frame in range(end_frame):

        blank = np.zeros((nrows, ncols))
        posframe = meantracks[frame,:,:]

        # multiplication factor is just for the running average. 
        blank[posframe[:,0], posframe[:,1]] += map_vals[:, frame] 
        blank[posframe[:,0], posframe[:,1]] *= (frame+1) / float(frame+2)
#* (frame+1 / float(frame+2)) # add on the values in the thing already.. 
        all_positions += blank
        
        spatial_time_saliency_map.append(blank[None,:])
        
    spatial_time_saliency_map = np.concatenate(spatial_time_saliency_map, axis=0)

    if filt:
        kernel = square(filt_size)
        all_positions = cv2.filter2D(all_positions,-1,kernel)
        
        for ii in range(spatial_time_saliency_map.shape[0]):
            spatial_time_saliency_map[ii] = cv2.filter2D(spatial_time_saliency_map[ii],-1,kernel)

    final_saliency_map = np.zeros((nrows, ncols))

    # final aggregation, should we sum here ? or should we average? 
    for reg in regs:
        final_saliency_map[spixels == reg] = np.mean(all_positions[spixels==reg])
            
    return final_saliency_map, spatial_time_saliency_map
    
    
def compute_boundary_formation_index(saliency_map_r, saliency_map_g, spixel_size, pad_multiple=3):
    
    """ Compute the saliency map based on counting the number of neighbours in the radius of each superpixel, then binning counts over regular grid of superpixels to produce a kernel density estimate and average smooth the images (if specified). In addition return the boundary formation index as defined in the paper.
    
    Parameters
    ----------
    saliency_map_r : numpy array
        (n_rows, n_cols) heatmap from the red tracks
    saliency_map_g : numpy array
        (n_rows, n_cols) heatmap from the green tracks
    spixel_size : float
        average superpixel width
    
    Returns
    -------
    boundary_formation_index : float
        the propensity of boundary formation between red and green sheets.
    av_saliency_map : numpy array
        (n_rows, n_cols) averaged saliency map.
    
    """
    from skimage.filters import threshold_otsu
    import numpy as np 
    
    av_saliency_map = .5*(saliency_map_r + saliency_map_g)
    cropped_saliency_map = av_saliency_map[pad_multiple*spixel_size:-pad_multiple*spixel_size, pad_multiple*spixel_size:-pad_multiple*spixel_size]

    thresh = threshold_otsu(cropped_saliency_map)
    max_mask = cropped_saliency_map >= thresh
    
    # grab the average high and average low.... to characterise the distribution.  
    max_av_density = np.mean(cropped_saliency_map[max_mask==1])
    max_bg_density = np.mean(cropped_saliency_map[max_mask==0])
    
    # compute the metric ...
    boundary_formation_index = (max_av_density - max_bg_density)/ max_av_density
    
    return boundary_formation_index, cropped_saliency_map
    
#==============================================================================
#  Velocity cross-correlation function tools
#==============================================================================

def vccf(tracks1, tracks2, mode='full'):
    
    """ Compute the normalised velocity correlation function (-1,1) between a series of tracks1 and series of tracks2 whose lengths can vary.
    
    Parameters
    ----------
    tracks1 : numpy array
        (n_superpixels x length1) numpy array
    tracks2 : numpy array
        (n_superpixels x length2) numpy array
    mode : str
        convolutional mode of operation, refer to numpy.correlate. 
    
    Returns
    -------
    vccf : numpy array 
        (length1+length2-1) long velocity cross-correlation vector as defined in the SI materials and methods if mode='full' 
    """
    import itertools
    import numpy as np

    n_1 = tracks1.shape[0]
    n_2 = tracks2.shape[0]
    
#    list_index = itertools.permutations([n_1, n_2])
    list1 = np.arange(n_1)
    list2 = np.arange(n_2)    
    list_index = list(itertools.product(list1, list2))
    
    ccfs = 0
    
    for index in list_index:
        
        i, j = index
        
        sig1 = tracks1[i]
        sig1_x = sig1[:,0]
        sig1_y = sig1[:,1]
        
        sig2 = tracks2[j]
        sig2_x = sig2[:,0]
        sig2_y = sig2[:,1]
               
        eps = np.finfo(np.float32).eps

        ccf_x = np.correlate((sig1_x-np.mean(sig1_x))/((np.std(sig1_x)+eps)*len(sig1_x)), (sig2_x-np.mean(sig2_x))/(np.std(sig2_x)+eps), mode=mode)
        ccf_y = np.correlate((sig1_y-np.mean(sig1_y))/((np.std(sig1_y)+eps)*len(sig1_y)), (sig2_y-np.mean(sig2_y))/(np.std(sig2_y)+eps), mode=mode)
        
        ccf_top = (ccf_x + ccf_y)

        ccfs += ccf_top/ float(len(list_index))
        
    return ccfs


def compute_max_vccf_cells_before_after_gap(meantracks_r, meantracks_g, wound_heal_frame, err_frame=5):
    
    """ Compute maximum velocity cross correlation index before and after gap closure as defined.
    
    Parameters
    ----------
    meantracks_r : numpy array
        (n_superpixels, n_frames, 2), numpy array of superpixel tracks for red
    meantracks_g : numpy array
        (n_superpixels, n_frames, 2), numpy array of superpixel tracks for green
    wound_heal_frame : int
        the frame number for which sheets close
    err_frame : int
        this is the margin of error to assure before and after gap closure.
    
    Returns
    -------
    (max_vccf_index_before, vccf_all_before) : tuple of numpy arrays
        tuple where max_vccf_index_before is a float of the maximum velocity correlation index before closure, vccf_all_before is the entire correlation function. 
    (max_vccf_index_after, vccf_all_after) : tuple of numpy arrays
        tuple where max_vccf_index_after is a float of the maximum velocity correlation index after closure, vccf_all_after is the entire correlation function. 
    """
    import numpy as np 
    
    # 1. create the subset tracks before and after wound frame.
    meantracks_r_before = meantracks_r[:,:np.maximum(2,wound_heal_frame-err_frame),:]
    meantracks_g_before = meantracks_g[:,:np.maximum(2,wound_heal_frame-err_frame),:]
                                      
    meantracks_r_after = meantracks_r[:, wound_heal_frame+err_frame:,:]
    meantracks_g_after = meantracks_g[:, wound_heal_frame+err_frame:,:]
    
    # 2. create the respective velocity vectors.
    velocity_r_before = meantracks_r_before[:, 1:, :].astype(np.float) - meantracks_r_before[:,:-1,:].astype(np.float)
    velocity_g_before = meantracks_g_before[:, 1:, :].astype(np.float) - meantracks_g_before[:,:-1,:].astype(np.float)
    
    velocity_r_after = meantracks_r_after[:, 1:, :].astype(np.float) - meantracks_r_after[:,:-1,:].astype(np.float)
    velocity_g_after = meantracks_g_after[:, 1:, :].astype(np.float) - meantracks_g_after[:,:-1,:].astype(np.float)
    
    # 3. compute the VCCF before and after.                      
    vccf_all_before = vccf(velocity_r_before, velocity_g_before)
    vccf_all_after = vccf(velocity_r_after, velocity_g_after)
    
    max_vccf_index_before = np.max(np.abs(vccf_all_before))
    max_vccf_index_after = np.max(np.abs(vccf_all_after))
    
    return (max_vccf_index_before, vccf_all_before), (max_vccf_index_after, vccf_all_after)
    

#==============================================================================
#   Mesh Disorder Index
#==============================================================================
def compute_local_point_density(tracks, frame, radius):
    """ Computes the number of neighbours within a radius for each superpixel at a given frame.
    
    Parameters
    ----------
    tracks : numpy array
        (n_superpixels, n_frames, 2), numpy array of superpixel tracks
    frame : int
        (int) video frame number for which to compute the number of neighbours.
    radius : float
        radial distance, smaller than which points are considered neighbours.
    
    Returns
    -------
    n_neighbours_regions : numpy array
        number of neighbours at distance <= radius for each superpixel.
    
    """
    from sklearn.metrics.pairwise import pairwise_distances
    import numpy as np 
    
    # for first frame determine the partners. 
    frame_tracks = tracks[:,frame,:]    
    
    dist_regions = pairwise_distances(frame_tracks, metric='euclidean')
    mask_dist = dist_regions <= radius
    
    n_neighbours_regions = np.sum(mask_dist, axis=0) - 1 #remove the self connection 
    
    return n_neighbours_regions


def compute_mesh_principal_strain_angle_ellipse(tracks, neighbours, point_thresh):
    
    """ Compute the principal strain ellipses at each superpixel point in the constructed mesh where the mesh connections is given in terms of a region adjacency graph (RAG) and the RAG is a neighbour list.
    
    Parameters
    ----------
    tracks : numpy array
        (n_superpixels, n_frames, 2) numpy array of superpixel tracks.
    neighbours : list
        n_superpixels neighbour list describing the neighbours of each superpixel.
    point_thresh : float 
        (float or int) threshold for detecting and masking out regions of natural high point density which may introduce artifacts such as the image borders and interfaces.
 
    Returns
    -------
    eigenvalues_time : numpy array
        n_frames x n_superpixels x 2 eigenvalue of 1st (max_strain) and 2nd (min_strain) eigenvectors.
    eigenangles_time : numpy array
        n_frames x n_superpixels or principal angles in rads
    stretch_ratio_time : numpy array
        n_frames x n_superpixels of the stretch ratio which is the eccentricity of an ellipse, sqrt(abs(1-(min_strain**2)/(max_strain**2)))
    select_time : numpy array
        n_frames x n_superpixels, the binary mask of where the local density exceeds the threshold of high point density
    
    """
    import numpy as np 
    
    nframes = tracks.shape[1]
    spixel_size = tracks[1,0,1] - tracks[0,0,1]
    
    eigenvalues_time = []
    eigenangles_time = []
    stretch_ratio_time = []
    select_time = []
    
    for frame in range(nframes):
        
        # think i need the other strain. 
        eigenvalues, eigenangles, angle_binning  = compute_principal_strain_angles(tracks, neighbours, frame)
    
        # Compute strains
        max_strain = np.maximum(eigenvalues[:,0], eigenvalues[:,1])
        min_strain = np.minimum(eigenvalues[:,0], eigenvalues[:,1]) 

        stretch_ratio = np.sqrt( np.abs(1. - (min_strain**2) / (max_strain**2) ) )
        
        eigenvalues_time.append(eigenvalues)
        eigenangles_time.append(eigenangles)
        stretch_ratio_time.append(stretch_ratio)

        """
        Accounting for density defects... 
        """
        point_neighbors = compute_local_point_density(tracks, frame=frame, radius=spixel_size) 
        select = point_neighbors <= point_thresh
        
        select_time.append(select)

    eigenvalues_time = np.array(eigenvalues_time)
    eigenangles_time = np.array(eigenangles_time)
    stretch_ratio_time = np.array(stretch_ratio_time)
    select_time = np.array(select_time)

    if len(eigenvalues_time.shape)==4:
        eigenvalues_time = eigenvalues_time[:,0]
    if len(eigenangles_time.shape)==3:
        eigenangles_time = eigenangles_time[:,0]
    if len(stretch_ratio_time.shape)==3:
        stretch_ratio_time = stretch_ratio_time[:,0]
    if len(select_time.shape)==3:
        select_time = select_time[:,0] 

    return eigenvalues_time, eigenangles_time, stretch_ratio_time, select_time
    
    
def fit_spline(x, y, k=4, smooth_factor=50):

    """ General boiler function to fit a spline to (x,y)
    
    Parameters
    ----------
    x : numpy array
        numpy array of x values
    y : numpy array
        numpy array of desired y values
    k : int 
        is the order of the spline for interpolation
    smooth_factor : float
        how smooth the interpolation, essentially controls the number of knots.
    
    Returns
    -------
    y_pred : numpy array 
        numpy array of predicted y values evaluated using the fitted spline at x
    spl: scipy spline fit object 
        the fitted spline function.
    """
    from scipy.interpolate import UnivariateSpline

    spl = UnivariateSpline(x, y, k=k, s=smooth_factor)
    y_pred = spl(x)
    
    return y_pred, spl


def vector_order(vectors, remove_mean=False):
    
    """ also known as the polar order parameter. (with optional subtraction of the mean vector)
    
    Parameters
    ----------
    vectors : numpy array
        n_vectors x 2, numpy array of directional (x,y) or (y,x) values

    Returns
    -------
    vorder : float
        scalar, return vector order parameter a.k.a normalised mean vector.
    """
    
    import numpy as np 
    
    mean_vector = np.mean(vectors, axis=0)
    
    if remove_mean:
    # remove the mean     
        v = vectors - mean_vector[None,:]
    else:
        v = vectors.copy()
        
    # vector order is computed by taking the magnitude effective velocity and dividing by the total magnitude of the vectors
    vorder = np.linalg.norm(np.sum(v, axis=0))/ float(len(v)  * (np.nanmean( np.linalg.norm(v, axis=1))) + 1e-16)
    
    return vorder

def vector_order_curve(vector_field, remove_mean=False):
    
    """ Given a temporally varying vector field, compute the averaged normalized vector as a measure of order.
    
    Parameters
    ----------
    vector_field : numpy array
        n_frames x n_vectors x 2, numpy array of directional (x,y) or (y,x) values over n_frames.
    remove_mean : bool
        flag to either compute the order after removal of mean or not.
    
    Returns
    -------
    vorder : numpy array 
        n_frames, return vector order parameter at each frame.
    
    """

    # here we can use the cosine directionality for measuring this ? and the vorder. 
    n_frames, n_spixels, _ = vector_field.shape #(x,y) convention.
    
    order_curve = []

    for frame in range(n_frames):
        vectors = vector_field[frame,:]
        order = vector_order(vectors, remove_mean=remove_mean)
        order_curve.append(order)
    return np.hstack(order_curve)
    

def compute_mesh_order(mesh_strain_vector, remove_mean=False):

    """ Computes the proposed mesh order presented in the paper. This is the vector order but using mesh strain vectors instead as a 'velocity'
    
    Parameters
    ----------
    mesh_strain_vector : numpy array 
        n_frames x n_superpixels x 2, local average mesh strain vector as returned from mesh_statistics_tools.construct_mesh_strain_vector() function.
    remove_mean : bool
        flag to either compute the order after removal of mean or not.   
    
    Returns
    -------
    mesh_order_curve : numpy array
        n_frames, return mesh order parameter at each frame based on the constructed mesh strain vector.
    """
    mesh_order_curve = vector_order_curve(mesh_strain_vector, remove_mean=remove_mean)
    
    return mesh_order_curve

    
def compute_principal_strain_angles(tracks, neighbourhood, frame):
    
    """ Compute the principal strain statistics of the PCA ellipses.
    
    Parameters
    ----------
    tracks : numpy array
        (n_superpixels x n_frames x 2), numpy array of superpixel tracks.
    neighbourhood : list of numpy arrays
        n_superpixels long list of neighbours of each superpixel
    frame : int
        the frame number in which we compute the pca ellipse statistics.
    
    Returns
    ------- 
    eigs : numpy array
        (n_superpixels, 2) of the normalised PCA strain sorted descending by magnitude,  
    angles : numpy array
        (n_superpixels,) associated principle angle in degrees.
    mesh_quadrant_counts : numpy array 
        counts in which quadrant the strain vectors fall in using the principal strain axes as the orthogonal axes. It is supposed to assess the distortion of the mesh.
    """
    import numpy as np 
    
    n_regions = tracks.shape[0]    
    n_neighbors = [len(nhood) for nhood in neighbourhood]
    max_neighbours = np.max(n_neighbors)

    eigs = []
    angles = []
    quadrant_binning = []

    for j in range(n_regions):
        
        # directions. 
        current_point = tracks[j, frame, :]
        neighbour_points = tracks[neighbourhood[j], frame, :]

        points = np.vstack([current_point, neighbour_points])
        
        # create normalised displacement vectors to fingure out direction 
        disps = points[1:,:] - points[0,:][None,:]  # displacements
        mags = np.sqrt(disps[:,0] ** 2 + disps[:,1]**2) + 1e-10 #give some magnitude to normalise..... 
        norm_disps = disps / mags[:, None]

        # now compute the PCA to find the principal angles 
        s, V, angle = find_principal_directions(norm_disps)
        
        # count the deformities by binning into the quadrants. 
        ang_deviation = 180/max_neighbours
#        ang_limits = angle + ang_deviation + np.arange(0, max_neighbours)*ang_deviation*2  
        ang_limits = angle + 1e-8 + np.arange(0, max_neighbours)*ang_deviation*2  
        ang_limits = np.mod(ang_limits, 360) - 180.
        ang_limits = ang_limits / 180. * np.pi
        ang_limits = np.sort(ang_limits)
        
        #### find the displacement angs from the normal displacement vectors 
        disp_angles = np.arctan2(norm_disps[:,1], norm_disps[:,0])
        
        quadrant_counts = count_quadrant_bins(disp_angles, ang_limits)
        quadrant_binning.append(quadrant_counts)
        

        if len(s) == 2:
            eigs.append(s)
        else:
            s = np.array([s[0], s[0]]).ravel()
            eigs.append(s)

        angles.append(angle)
 
    eigs = np.vstack(eigs)
    angles = np.array(angles).ravel()
    mesh_quadrant_counts = np.vstack(quadrant_binning)
    
    return eigs, angles, mesh_quadrant_counts
    
    
def find_principal_directions(points):
    
    """ Given a set of (y,x) points, use PCA to compute the average directionality that covers the variance.
    
    Parameters
    ----------
    points : numpy array
        (n_points, 2) in (y,x) coordinate points giving the main displacement directions.
    
    Returns
    -------
    s : numpy array
        (2,) PCA strain largest to smallest.
    V : numpy array
        (2,2) associated eigenvectors to the eigenvalues.
    angle: numpy array
        (float [0,360]) principal angle using (min_strain, max_strain) as (x,y) 
    """
    import numpy as np 
    
    disps = points[1:,:] - points[0,:][None,:]
    disps -= np.mean(disps, axis=0)

    # do the svd decomposition. 
    U, s, V = np.linalg.svd(disps)
    
    # reorder so the first is the largest:
    sort_order = np.argsort(s)[::-1]

    s = s[sort_order]
    V = V[sort_order,:]

    # return the principal axis orientation for plotting purposes. 
    angle = np.arctan2(V[0,1], V[0,0])
    angle = 180 * angle / np.pi  # convert to degrees
    
    return s, V, angle
    
    
def count_quadrant_bins(angles_pi, bin_limits_ascend):
    
    """ Count the number of displacement vectors if we bin each displacement vector in the region adjacent graph if we use the principal strain to define a set of orthogonal axes.
    
    Parameters
    ----------
    angles_pi : numpy array
        (n_angles), vector of angles to each displacement vector given in radians
    bin_limits_ascend : numpy array
        this is the angles at which we partition the angles in increasing angles.
    
    Returns
    -------
    counts : numpy array
        (length of bin limits), the number of displacement vectors falling into designated bins. 
        
    """
    import numpy as np 
    # bin limits are in ascending intervals. 
    n_bins = len(bin_limits_ascend)
    counts = np.zeros(n_bins)
    
    for ang in angles_pi:
        for i in range(n_bins):
            
            if i == n_bins-1:
                # for the last one, we need to wrap around ... 
                if ang >= bin_limits_ascend[i] or ang < bin_limits_ascend[0]:
                    counts[i] += 1
            else:
                if ang >= bin_limits_ascend[i] and ang < bin_limits_ascend[i+1]:
                    counts[i] += 1

    return counts 
    
    
#==============================================================================
#   Spatial Correlation based on superpixel distance.
#==============================================================================

def powerfit(x, y):
    """
    fit a form of y=Ax**b by taking logarithms of both sides.
    
    Parameters
    ----------
    x : numpy array
        x-axis, a numpy vector
    y : numpy array
        signal, a numpy vector
    
    Returns
    -------
    (slope, intercept, r_value) : tuple
        tuple, output from scipy.stats.linregress
    y_pred : numpy array
        predicted y values based on the fitted relationship at the same x values.
    
    """
    from scipy.stats import linregress
    import numpy as np 
    
#    x = np.arange(len(y)) + 1
    # form log y = log A + B log x
    slope, intercept, r_value, p_value, std_err = linregress(np.log(x), np.log(y))
    y_pred = np.exp(intercept) * x ** slope

    return (slope, intercept, r_value), y_pred


def compute_spatial_correlation_function(meantracks, wound_heal_frame, wound_heal_err=5, dist_range=(1,5)):
    
    """Given the superpixel tracks of all time, the gap closure frame, the error of the wound heal frame and the distance range we want to test correlation, this function returns the spatial correlation function. 
    
    Parameters
    ----------
    meantracks : numpy array
        (n_superpixels x n_frames x 2), numpy array of superpixel tracks.
    wound_heal_frame : int
        (int) the frame number where the sheets are closed.
    wound_heal_err : int
        we truncate tracks beyond wound_heal_frame - wound_heal_err
    dist_range : tuple-like
        a vector of the distance to consider as a multipler given in number of superpixels away. creates a vector incrementing by 1. 
    
    Outputs:
    --------
    spatial_corr : numpy array
        (len(dist_range)) vector giving the mean superpixel correlation for each distance.
    
    (spatial_corr_pred,  a_value, b_value, r_value) : tuple
        tuple giving the fitted spatial corr, and the a and b values of y = a exp(-x/b) curve. r_value is the correlation of the fit.
    
    """
    
    import numpy as np 
    
    tracks = meantracks[:,: np.maximum(2, wound_heal_frame-wound_heal_err),:]
    disps = tracks[:,1:,:] - tracks[:,:-1,:]

    spatial_corr = np.zeros(len(dist_range))
    
    for kk in range(len(dist_range)):
                    
        dist_multiplier = dist_range[kk]
        nearest_neighbours = find_neighbours_initial(tracks, dist_thresh=dist_multiplier*1.2)
                    
        spatial_corr[kk] = np.nanmean(compute_spatial_correlation(disps, np.ones(tracks.shape[0], dtype=np.bool), nearest_neighbours))
        
    (slope, intercept, r_value), spatial_corr_pred = powerfit(dist_range, spatial_corr)
    
    b_value = -1./slope
    a_value = np.exp(intercept)
        
    return spatial_corr, (spatial_corr_pred,  a_value, b_value, r_value)
    
    
def find_neighbours_initial(tracks, dist_thresh):
    
    """ Generates a nieghbourlist specifying the region adjacency graph according to a distance threshold given as a multiplier of the average superpixel width.
    
    Parameters
    -----------
    tracks : numpy array
        (n_superpixels, n_frames, 2), a numpy array of superpixel tracks.
    dist_thresh : float
        (float) multiplier which is multiplied by the average superpixel distance to generate a Euclidean distance cutoff.
    
    Returns
    -------
    nearest_neighbors_refined : list of numpy arrays
        (n_superpixels,) list specifying region adjacency of each superpixel.
    
    """
    from sklearn.metrics.pairwise import pairwise_distances
    import numpy as np 
    # for first frame determine the partners. 
    
    first_frame = tracks[:,0,:]    
    n_regions = first_frame.shape[0] 

    spixel_size = tracks[1,0,1] - tracks[1,0,0]
    dist_thresh = spixel_size * dist_thresh
    
    dist_regions = pairwise_distances(first_frame, metric='euclidean')
    nearest_neighbors = np.argsort(dist_regions, axis=1)
    sort_dist_regions = np.sort(dist_regions, axis=1) #sort the regions. 
    
    # iterate and for each one, remove those that have inf. 
    nearest_neighbors_refined = []
    
    for i in range(n_regions):
            
        dist_i = sort_dist_regions[i]
        nearest_neighbors_refined.append(nearest_neighbors[i][dist_i<=dist_thresh][1:]) # start from 1, since the first is always the identity.
            
    return nearest_neighbors_refined

    
def compute_spatial_correlation(disps, filt, neighbors):
    """Computes the spatial correlation for each superpixel using a time dependent velocity vector.
    
    Parameters
    ----------
    disps : numpy array
        (n_superpixels, n_frames-1, 2), a numpy array of superpixel displacement vectors, n_frames here is equal to the truncated length if we choose not to use the entire duration to compute correlation.
    filt : numpy bool array
        (n_superpixels,) binary mask to indicate which superpixels to take for the calculation
    neighbors : list of numpy arrays
        (n_superpixels,) list of numpy array to detail which superpixel ids are the neighbours of the current superpixel index.
    
    Returns
    -------
    spatial_corr_superpixels : numpy array
        for the given region adjacency specified by neighbours return the local spatial correlation for each superpixel.
    
    """
    # neighbours is a neighbour list for each superpixel 
    import numpy as np 
    
    n_regions = len(neighbors)
    n_frames = disps.shape[1]
    good_index = np.arange(n_regions)[filt]

    time_correlations = []

    for i in range(n_regions):
        
        if i in good_index:
            
            neighbor = neighbors[i]
            move_vector0 = disps[i,:,:]
            move_vector_neighbours = disps[neighbor,:,:]

            velocity_corr = []

            for j in range(len(neighbor)):
                vcorr = correlation_tracks_time(move_vector0, move_vector_neighbours[j])
                velocity_corr.append(vcorr) 
                
            velocity_corr = np.hstack(velocity_corr)
        
            time_correlations.append(np.nanmean(velocity_corr))
    
    spatial_corr_superpixels = np.array(time_correlations).ravel()
    return spatial_corr_superpixels
    
    
def correlation_tracks_time(track1, track2):
    
    """ returns the pearson correlation (cov(track1,track2)/sqrt(var(track1)*var(track2))) between time vectors. 
    
    Parameters
    ----------
    track1 : numpy array 
        (n_frames, 2) single displacement vector for a superpixel (y,x) coordinates
    track2 : numpy array
        (n_frames, 2) single displacment vector for a superpixel (y,x) coordinates
    
    Returns
    -------
    av_corr : float
        average correlation coefficient across the superpixels.
    
    """
    import numpy as np 
    from scipy.stats import pearsonr
        
    track1_copy = track1.astype(np.float)
    track2_copy = track2.astype(np.float)
    
    x_corr = pearsonr(track1_copy[:,1], track2_copy[:,1])[0]
    y_corr = pearsonr(track1_copy[:,0], track2_copy[:,0])[0]

    av_corr = np.nanmean([x_corr, y_corr])

    return av_corr
    
    
#==============================================================================
#   Local motion statistics 
#==============================================================================
def compute_mesh_curl(meantracks, neighbors, filt=None):
    """ Computes the spatial correlation for each superpixel using a time dependent velocity vector.
    
    Parameters
    ----------
    disps : numpy array
        (n_superpixels, n_frames-1, 2), a numpy array zof superpixel displacement vectors, n_frames here is equal to the truncated length if we choose not to use the entire duration to compute correlation.
    neighbors : list of numpy array
        (n_superpixels,) list of numpy array to detail which superpixel ids are the neighbours of the current superpixel index.
    filt : numpy bool array
        (n_superpixels,) binary mask to indicate which superpixels to take for the calculation
    
    Returns
    --------
    mesh_curl : numpy array
        for the given region adjacency specified by neighbours return the local cross product of the neighbour vectors relative to the central.
    
    """
    # neighbours is a neighbour list for each superpixel 
    import numpy as np 
    
    disps = meantracks[:,1:] - meantracks[:,:-1] # get the velocity of the position.
    
    n_regions = len(neighbors)
    n_frames = meantracks.shape[1]-1
    good_index = np.arange(n_regions)[filt]

    mesh_curl = np.zeros((n_regions, n_frames))

    for i in range(n_regions):
        if i in good_index:
            
            neighbor = neighbors[i]

            if len(neighbor) > 0:
                move_vector0 = disps[i,:,:]
                pos_vector0 = meantracks[i,:,:]

                move_vector_neighbours = disps[neighbor,:,:]
                pos_vector_neighbours = meantracks[neighbor, :n_frames, :] # only up to the same number of frames as displacements.            

                for frame_no in range(n_frames):
                    
                    vector0 = move_vector0[frame_no]
                    pos0 = pos_vector0[frame_no]

                    vectorN = move_vector_neighbours[:,frame_no]
                    posN = pos_vector_neighbours[:,frame_no]
                    
                    # curl:  dFy_dx - dFx_dy in 2D.
                    dxy = posN - pos0[None,:] # physical deltas
                    dvector = vectorN - vector0[None,:] # velocity deltas
                     
#                    print(dvector.min(), dvector.max())
#                    print(dxy.min(), dxy.max())
                    val = dvector[:,0]/np.float32(dxy[:,1]) - dvector[:,1]/np.float32(dxy[:,0])
                    val = np.nanmean(val)
                    if np.isnan(val) or np.isinf(val):
                        val = 0
                    mesh_curl[i, frame_no] = val

    return mesh_curl
    

#==============================================================================
#   batch_processing function
#==============================================================================
def compute_aggregated_stats_map(meantracks, shape, dist_thresh=None, map_vals=None, max_frame=None, filt=0, filt_size=5):
    """ Compute the saliency map based on counting the number of neighbours in the radius of each superpixel, then binning counts over regular grid of superpixels to produce a kernel density estimate and average smooth the images (if specified)
    
    Parameters
    ----------
    meantracks : numpy array
        (n_superpixels x n_frames x 2), numpy array of superpixel tracks
    map_vals : numpy array
        (n_superpixels x n_frames), the temporal mesh strain
    shape : tuple
        (n_rows x n_cols), image shape
    max_frame : int 
        the final frame to which we aggregate the statistic and compute maps for.
    filt : bool
        (0 or 1), run over the heat map with an average filter or not. if 1, filt_size sets the kernel size for smoothing.
    filt_size : float
        (float), sigma setting the standard deviation of the gaussian for kernel smoothing.
    
    Returns
    -------
    final_saliency_map : numpy array
        (n_rows x n_cols), heatmap of saliency 
    spatial_time_saliency_map : numpy array
        (n_frames x n_rows x n_cols), frame by frame heatmap of saliency.
    
    """
    import numpy as np 
    from skimage.segmentation import slic, relabel_sequential 
    from skimage.morphology import square
    import cv2
    
    # compute the saliency map
    if map_vals is None:
        map_vals = count_dynamic_nn_neighbors(meantracks, dist_thresh, limits=shape)
    
    # coerce meantracks into the intended shape 
    meantracks = np.transpose(meantracks,(1,0,2))
    nframes, nregions, _ = meantracks.shape
    
    # create superpixel map 
    nrows, ncols = shape
    
    # Create the superpixel canvas we will aggregate over. 
    nothing = np.zeros((nrows, ncols, 3))
    spixels = slic(nothing, n_segments=nregions)
    spixels, _,_ = relabel_sequential(spixels)	
 
    regs = np.unique(spixels)
    # iterate over all the tracks and collate positions. 
    all_positions  = np.zeros((nrows, ncols))

    if max_frame:
        end_frame = max_frame
    else:
        end_frame = nframes
    
    spatial_time_saliency_map = []
    
    for frame in range(end_frame):

        blank = np.zeros((nrows, ncols))
        posframe = meantracks[frame,:,:]

        # multiplication factor is just for the running average. 
        blank[posframe[:,0], posframe[:,1]] += map_vals[:, frame] 
        blank[posframe[:,0], posframe[:,1]] *= (frame+1) / float(frame+2)
#* (frame+1 / float(frame+2)) # add on the values in the thing already.. 
        all_positions += blank
        
        spatial_time_saliency_map.append(blank[None,:])
        
    spatial_time_saliency_map = np.concatenate(spatial_time_saliency_map, axis=0)

    if filt:
        kernel = square(filt_size)
        all_positions = cv2.filter2D(all_positions,-1,kernel)
        
        for ii in range(spatial_time_saliency_map.shape[0]):
            spatial_time_saliency_map[ii] = cv2.filter2D(spatial_time_saliency_map[ii],-1,kernel)

    final_saliency_map = np.zeros((nrows, ncols))

    # final aggregation, should we sum here ? or should we average? 
    for reg in regs:
        final_saliency_map[spixels == reg] = np.mean(all_positions[spixels==reg])
            
    return final_saliency_map, spatial_time_saliency_map
    