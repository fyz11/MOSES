#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 21:32:25 2017

@author: felix
@email: felixzhou1@gmail.com

This script and all the code within is licensed under the Ludwig License.

see README.md for details on usage.


Information
------------
This is a specialised track cleaning module for migrating epithelial sheets
currently it is framed for 2 coloured sheets, but one can easily adapt for phase contrast migrating sheets too.

1) applied superpixel clique cleaning
2) dynamic propagation with continuity enforcement to grab all the relevant tracks. 

N.B. assumes diametrically opposing eptheliod sheets. 

"""

import numpy as np # global import since a lot of the functions share.

 
def find_cliques_superpixels(meantracks, select, thresh_factor=1.2):

    """
    For the given selected points, link points close together in a graph.
    
    Input:
    ------
    meantracks: (n_superpixels, n_frames, 2) numpy array representing the (y,x) superpixel positions
    select: logical mask of True or False indicating which superpixels to construct graph for
    thresh_factor: the radial distance cut-off as a multiplier of the average superpixel size.
    
    Output:
    -------
    coords: (y,x)-coordinates of the selected superpixel points at frame 0
    nearest_neighbors_refined: corresponding python list of numpy arrays giving the ids of superpixels that are neighbours to each superpixel. If there is no neighbours for a superpixel, the numpy array is empty. 
    
    """
    
    from sklearn.metrics.pairwise import euclidean_distances

    # Based on initial distances of superpixels construct neighbourhood graph.
    coords = meantracks[select, 0, :]
    spixel_size = np.abs(meantracks[0,0,1] - meantracks[1,0,1])
    dist_thresh = thresh_factor * spixel_size

    # build a dist matrix
    dist_regions = euclidean_distances(coords)
    dist_regions[dist_regions > dist_thresh] = np.inf #set to ridiculously high

    # sort the regions
    nearest_neighbors = np.argsort(dist_regions, axis=1)
    n_regions = dist_regions.shape[0]

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

    return coords, nearest_neighbors_refined

    
def find_cliques_neighbour_list(nearest_neighbors_refined, thresh_size=4):

    """
    This is similar to the networkx function for finding graph components. Here we find all the isolated connected components of the neighbourhood graph. The basic idea is that the superpixels constituting the initial epithelial sheet is approximated by the largest set of interconnected superpixels. 
    
    Inputs:
    -------
    nearest_neighbor_refined: python list of numpy arrays for expressing the superpixel adjacency graph.
    thresh_size: filters which graph components to keep by specifying a minimum number of superpixels that much belong to the component. This is to reduce consideration of singleton.s
    
    Outputs:
    --------
    final_uniq: list of cliques given as a list of array of superpixel ids e.g. [(clique1), (clique2),...] 
    
    """
    
    ### Initialise the storage facilities
    # now we just loop over all the nearest neighbours and create cliques.
    cliques = []
    to_compute = np.ones(len(nearest_neighbors_refined), dtype=np.int)

    ### first filter all singletons, iterating over the entire dataset.
    for i in range(len(nearest_neighbors_refined)):

        query = nearest_neighbors_refined[i]

        # these are not singletons.
        if len(query) > 0:
            full = [query]
            for j in range(len(query)):
                full.append(nearest_neighbors_refined[query[j]])
            full = np.hstack(full)

            # filter the unique connections
            cliques.append(np.unique(full))

        else:
            cliques.append(i) # thats the end of that, its a singleton.
            to_compute[i] = 0 # do not want to bother with these any more.


    #==============================================================================
    #   Here we iterate through
    #==============================================================================
    array_index = np.arange(len(nearest_neighbors_refined))

    while np.sum(to_compute) > 1:

        array_index_compute = array_index[to_compute==1]

        # now we just look through these.
        for i in range(len(array_index_compute)):

            query_index = array_index_compute[i]
            query = cliques[query_index]

            # these are all not singletons.
            full = [query]
            for j in range(len(query)):
                full.append(cliques[query[j]])
            full = np.hstack(full)

            uniq_cluster = np.unique(full)

            if np.array_equal(query, uniq_cluster):
                to_compute[query_index] = 0

            else:
                cliques[query_index] = uniq_cluster

    #==============================================================================
    #   Final Filter into unique list
    #==============================================================================
    uniques = []
    for arr in cliques:

        check = False

        for item in uniques:
            if np.array_equal(arr, item):
                check=True
                break

        if check:
            # do nothing:
            continue
        else:
            uniques.append(arr)

    # we thresh based on size to remove annoying singletons and too small areas that is probably spurious.
#    flattened_uniques = [np.array(uniq).ravel() for uniq in uniques]
#    flattened_uniques = np.hstack(flattened_uniques)
#    print len(flattened_uniques), len(np.unique(flattened_uniques))
    final_uniq = []

    for uniq in uniques:

        if len(np.array(uniq).ravel()) > thresh_size:
            final_uniq.append(uniq)

    return final_uniq
    
    
def filter_tracks_superpixel_cliques(tracks, select_tracks, thresh_factor=1.1):
    
    """
    Constructs a neighbourhood graph based on initial points of tracks according to a distance cut-off for selected points specified by a binary mask. It does this twice, the second time to account for central nodes without which we would have two disconnected graph components
    
    Code is tested for a 4 connected graph, thresh_factor <=1.2
    
    Inputs:
    -------
    tracks: (n_superpixels x n_frames x 2) numpy array of superpixel tracks
    select_tracks: binary mask selecting which superpixels to construct graph for
    thresh_factor: (float) distance cut-off specified as a multiplicative factor of the average superpixel size
        
    Outputs:
    --------
    keep_spixels: a numpy vector of numbered ids of which superpixels to keep
    
    """
    
    # attempts to clean up the tracks and eliminate those not part of the 'bulk' for either color based on clique/graph component neighbourhood analysis. 
    
    # 1. first round of filtering - identify the largest collection of tracks. 
    coords, partners = find_cliques_superpixels(tracks, select_tracks, thresh_factor=thresh_factor)
                
    select_spixels = np.arange(tracks.shape[0])[select_tracks==1]
    track_cliques = find_cliques_neighbour_list(partners) 
    
    # only get the largest track clique 
    good_id = np.argmax([len(traq) for traq in track_cliques])
    good_tracks = track_cliques[good_id]

    # apply the filter to the tracks.
    filt_tracks = tracks[select_spixels,:,:]
    filt_tracks = filt_tracks[good_tracks]

    
    # 2. second round of filtering - recalculate the neighbours this time, this time we go for the 2-links and remove all of them.., and find the cliques. 
    coords, partners = find_cliques_superpixels(filt_tracks, np.ones(len(good_tracks), dtype=np.bool), thresh_factor=thresh_factor)
                
    sum_partners = np.array([len(part) for part in partners]).ravel()
    
    select_partners = sum_partners > 2 # select to retain ids that have more than 2 neighbours. 
    
    filt_filt_tracks = filt_tracks[select_partners]

    
    # 3. final application of largest clique finding. 
    coords, partners = find_cliques_superpixels(filt_filt_tracks, np.ones(filt_filt_tracks.shape[0], dtype=np.bool), thresh_factor=thresh_factor)
    track_cliques = find_cliques_neighbour_list(partners)
    
    good_clique_id = np.argmax([len(traq) for traq in track_cliques])
    good_clique = track_cliques[good_clique_id]

    
    # 4. Finally collate together all the filters
    keep_spixels = (select_spixels[good_tracks[select_partners]])[good_clique]

    return keep_spixels
    
        
# this implements the above only once... in case we only need to do it once. 
def filter_tracks_superpixel_cliques_once(tracks, select_tracks, thresh_factor=1.1):
    
    """
    Constructs a neighbourhood graph based on initial points of tracks according to a distance cut-off for selected points specified by a binary mask. It does this once.
    
    Code is tested for a 4 connected graph, thresh_factor <=1.2
    
    Inputs:
    -------
    tracks: (n_superpixels x n_frames x 2) numpy array of superpixel tracks
    select_tracks: binary mask selecting which superpixels to construct graph for
    thresh_factor: (float) distance cut-off specified as a multiplicative factor of the average superpixel size
        
    Outputs:
    --------
    keep_spixels: a numpy vector of numbered ids of which superpixels to keep
    
    """
    # attempts to clean up the tracks and eliminate those not part of the 'bulk' for either color based on clique neighbourhood analysis. 
    
    # 1. first round of filtering - identify the largest collection of tracks. 
    coords, partners = find_cliques_superpixels(tracks, select_tracks, thresh_factor=thresh_factor)
                
    select_spixels = np.arange(tracks.shape[0])[select_tracks==1]
    track_cliques = find_cliques_neighbour_list(partners)

    # only get the largest track clique 
    good_id = np.argmax([len(traq) for traq in track_cliques])
    good_tracks = track_cliques[good_id]


    return select_spixels[good_tracks]

    
# this expands the initial superpixel selection and turns it into a proper meantracks object .... 
def apply_filter_tracks(meantracks, select_spixels):
    
    """
    This is a helper function to basically turn inactive superpixels into constant tracks.
    
    Inputs:
    -------
    meantracks: (n_superpixels, n_frames, 2), numpy array of superpixel tracks.
    select_pixels: binary numpy mask indicating superpixels to keep, those that are False are to be set to constant tracks.
    
    Outputs:
    --------
    new_meantracks: (n_superpixels, n_frames, 2), final numpy array of superpixel tracks after replacing inactivated with constant tracks.
    
    """
    
    new_meantracks = meantracks.copy()

    select = np.ones(meantracks.shape[0], dtype=np.bool)
    select[select_spixels] = False # do not need to modify those that are being kept.
    
    index = np.arange(meantracks.shape[0], dtype=np.int)
    sel_index = index[select]

    for index in sel_index:    
        # they do not move past the initial position
        new_meantracks[index,:,0] = meantracks[index,0,0]
        new_meantracks[index,:,1] = meantracks[index,0,1]

    return new_meantracks
    

# critical at the start in order to narrow the search... important for producing spixel cliques. 
def find_moving_tracks(forward_tracks, frame2=1):
    
    """
    Find which superpixels are moving between frame0 to frame2 assessed by if the superpixel has been displaced by computing the distance in location between the two frames.
    
    Input:
    ------
    forward_tracks: (n_superpixels, n_frames, 2) numpy array of superpixel tracks from optical flow.
    
    Output:
    -------
    select: (n_superpixels,) binary logical mask numpy vector
    """
    
    move = forward_tracks[:, frame2,:] - forward_tracks[:,0,:]
    move = move[:,0]**2 + move[:,1]**2 
    select = move > 0 
    
    return select
    

def find_moving_tracks_generic(tracks_prev, tracks_current, thresh=0):
    
    move = tracks_prev - tracks_current
    move = move[:,0]**2 + move[:,1]**2
    select = move > thresh
    return select    

    
# self explanatory, plot both tracks on the same axes. 
def make_plot(tracks_r, tracks_g, ax):

    n_r = tracks_r.shape[0]
    n_g = tracks_g.shape[0]

    for i in range(n_r):
        ax.plot(tracks_r[i,:,1], tracks_r[i,:,0], 'r')
    for i in range(n_g):
        ax.plot(tracks_g[i,:,1], tracks_g[i,:,0], 'g')

    return []
    
# this raster fills missing superpixels... (allows maximum retention of information and continuity especially at the start. )
def fill_in_missing_superpixels( meantracks, keep_spixels, shape):
    
    """
    Exploiting the regularity of the initial superpixels, it allows filling in of the missing superpixels within the bulk sheet that wasn't identified as moving.
    
    Inputs:
    -------
    meantracks: (n_superpixels, n_frames, 2) numpy array of superpixel tracks
    keep_spixels: numpy array of superpixel ids that was kept (activated).
    shape: (n_rows, n_cols) tuple of image shape
    
    Outputs:
    --------
    final_spixels_keep: a new numpy array of superpixel ids 
    
    """
    # just need the superpixels i have kept and the coordinates of the superpixels which ofc ... is available from the track information!. 
    # shape is used for the image limits. 
    # we do this by line fill. 
    y_limit = shape[0]
    x_limit = shape[1]

    initial_coords = meantracks[:,0,:] # in the form (y,x)
    
    n_regions = meantracks.shape[0]
    all_spixels = np.arange(n_regions)
    
    remain_spixels = np.setdiff1d(all_spixels, keep_spixels) #find those that remain. 
    
    # grab the individual spixel locations. 
    remain_coords = initial_coords[remain_spixels, :]
    keep_coords = initial_coords[keep_spixels, :]


    # find the bounds of the kept coords.
    x_min = np.min(keep_coords[:,1])
    x_max = np.max(keep_coords[:,1])
    
    #==============================================================================
    #   Determine all the unique spixels  
    #==============================================================================
        
    # from this range we can determine the direction to fill in .
    uniq_x_coords = np.unique(initial_coords[:,1])
    uniq_y_coords = np.unique(initial_coords[:,0])
    
    spixel_x_size = uniq_x_coords[1] - uniq_x_coords[0]
    spixel_y_size = uniq_y_coords[1] - uniq_y_coords[0]


    # Organise two lists according to the unique y_coords. (equivalent to a horizontal scan.)
    keep_list_y = [keep_coords[keep_coords[:,0] == uniq_y_coord,1] for uniq_y_coord in uniq_y_coords]
    remain_list_y = [remain_coords[remain_coords[:,0] == uniq_y_coord,1] for uniq_y_coord in uniq_y_coords]
    
    remain_y_ids = [remain_spixels[remain_coords[:,0] == uniq_y_coord] for uniq_y_coord in uniq_y_coords]

#    print len(remain_spixels), len(keep_spixels)
#    print remain_y_ids

    #==============================================================================
    #   Determine right or left.   
    #==============================================================================
    additional_spixels = []

    if x_max <= x_limit - 2 * spixel_x_size:
        
#        print 'left' 
        for j in range(len(keep_list_y)):
            
            candidate_y = remain_list_y[j]
            ref_y = keep_list_y[j]

            if len(candidate_y) > 0 and len(ref_y) > 0: #make sure have something to compare. 
                additional_spixels.append(remain_y_ids[j][candidate_y <= np.max(ref_y)])

        
    if x_min >= 2*spixel_x_size:
        
#        print 'right'
        for j in range(len(keep_list_y)):
            
            candidate_y = remain_list_y[j]
            ref_y = keep_list_y[j]

            if len(candidate_y) > 0 and len(ref_y) > 0: #make sure have something to compare. 
                additional_spixels.append(remain_y_ids[j][candidate_y >= np.min(ref_y)])
        
        
#    print additional_spixels
#    print np.hstack(additional_spixels)

    if len(additional_spixels) > 0:
        final_spixels_keep = np.hstack([keep_spixels, np.hstack(additional_spixels)])
    else:
        final_spixels_keep = keep_spixels.copy()
    
    return final_spixels_keep
    

def count_points_spixels(spixels, regions2count, points):
    
    """
    Checking if a superpixel centroid is in the superpixel area of another using indexing.
    
    Inputs:
    -------
    spixels: (n_rows, n_cols) image with each distinct superpixel region assigned a unique int (output of scikit-image slic algorithm)
    regions2count: numpy array of candidate superpixel regions to check
    points: (y,x) positions of activated superpixels.
    
    Outputs:
    --------
    counts:  
        
    """
    # the first is superpixel map... the ids of which can be used to access the region counts. 
    counts_all_spixels = np.zeros(len(np.unique(spixels)))
    counts_all_spixels[spixels[points[:,0], points[:,1]]] += 1 # index and add. 
    counts = counts_all_spixels[regions2count]
        
    return counts
    
    
def dynamic_propagation_spixels_fast(tracks, start_spixels, shape):
    
    """
    Dynamically propagates the initial assigned superpixels to the shape of the initial sheet in order to capture the full sheet dynamics. An inactivated superpixel becomes activated if an activated superpixel comes into the superpixel area (area of influence)
    
    Inputs:
    -------
    tracks: (n_superpixels, n_frames, 2), numpy array of superpixel tracks from optic flow.
    start_spixels: ids of superpixels belonging to the initial epithelial sheet
    shape: (n_rows, n_cols) image shape.
    
    
    Outputs:
    --------
    meantracks_new: (n_superpixels, n_frames, 2), new numpy array of superpixel tracks, with inactivated superpixels replaced by constant tracks.
    activated_spixels: numpy array of superpixel ids that became activated.
    running_spixels: numpy array of all activated/kept superpixel ids
    
    """
    
    # this function dynamically projects the initial superpixels and 'activates' other superpixels on request. 
    from skimage.segmentation import slic, relabel_sequential
    
    m, n = shape 
    n_spixels = tracks.shape[0]
    n_frames = tracks.shape[1]
    
    spixel_size = np.abs(tracks[1,0,1] - tracks[0,0,1])

    # 1. create the superpixels 
    spixels = slic(np.zeros(shape), n_segments=n_spixels, compactness=10)
    spixels, _, _ = relabel_sequential(spixels)
 
    # 2. create save list.
    all_spixels = np.arange(n_spixels)
    
    # these are the two lists we have to keep track of ... 
    other_spixels = np.setdiff1d(all_spixels, start_spixels)
    cand_spixels = other_spixels.copy()
    
    running_spixels = start_spixels.copy()
    activated_spixels = []


    # 3. propagate positions 
    for frame in range(1, n_frames,1 ):
        
        
        # check first other_spixels is not simply empty ...
        if len(other_spixels) > 0: 
        
            # don't have to do the first frame. 
            points = tracks[running_spixels, frame, :] #(y,x) format. 
    
            # count the number of points falling into the regions. 
            spixel_counts = count_points_spixels(spixels, other_spixels, points)
            
            # add the activated ids 
            activated_ids = other_spixels[spixel_counts > 0]
        
            if len(activated_ids) > 0 : # if not nothing. 
                activated_spixels.append([frame, activated_ids])
            
                # update the running spixels 
                running_spixels = np.append(running_spixels, activated_ids)
            
                # delete the activated spixels from the other spixels
                other_spixels = np.delete(other_spixels, np.arange(len(other_spixels))[spixel_counts > 0])
            
        else:
            
            # no point in continuing.
            break
        
    # 4. Construct a new meantracks. 
    meantracks_new = tracks.copy() # copy the old tracks. 
    # for regions that were not retained... set to their initial values.
    meantracks_new[cand_spixels, :, :] = tracks[cand_spixels, 0, :][:,None,:].copy() # the padding is to allow propagation  
    
    # then .... we add in the activated
    for activation_data in activated_spixels:
        
        frame, regions = activation_data
        
        # remember we checking against the recorded... 
        check = tracks[regions, frame,:] - meantracks_new[regions, frame-1, :]
        check = np.sqrt(check[:,0]**2 + check[:,1]**2)
        
#        print check
        check = check <= spixel_size 
        
        if np.sum(check) > 0:
            meantracks_new[regions[check], frame:, :] = tracks[regions[check], frame:, :].copy() #add this back in. 
    
    return meantracks_new, activated_spixels, running_spixels
    
    
def check_move_all_frames(meantracks, move_thresh=0):
    
    """
    Checks which superpixels are now moving based on averaging the total distance moved across all frames. 
    
    Inputs:
    -------
    meantracks: (n_superpixels, n_frames, 2), numpy array of superpixel tracks.
    move_thresh: distance to be moved in pixel units for determining if a superpixel is moving.
    
    Outputs:
    --------
    move_check: binary mask (True/False) of which superpixels is moving.
    
    """
    
    move_check = np.sqrt(np.mean(np.sum((meantracks[:,1:,:] - meantracks[:,:-1,:]) ** 2, axis=2), axis=1)) > move_thresh

    return move_check
    
    
    
def check_overlap(meantracks_r, meantracks_g, move_r, move_g):
    
    """
    Joint filtering of superpixels based on movement in order to uniquely assign superpixels to either the red or green sheet.
    
    Inputs:
    -------
    meantracks_r: (n_superpixels, n_frames, 2), unfiltered red tracks from optical flow superpixel tracking
    meantracks_g: (n_superpixels, n_frames, 2), unfiltered green tracks from optical flow superpixel tracking
    move_r: binary mask of superpixel ids from meantracks_r that are moving
    move_g: binary mask of superpixel ids from meantracks_g that are moving
    
    Outputs:
    --------
    new_move_r: new binary mask of superpixel ids to be kept for meantracks_r 
    new_move_g: new binary mask of superpixel ids to be kept for meantracks_g
    
    """
    move_spixels_r = np.arange(meantracks_r.shape[0])[move_r]
    move_spixels_g = np.arange(meantracks_g.shape[0])[move_g]

    intersect_pixels = np.intersect1d(move_spixels_r, move_spixels_g)
    
    
    if len(intersect_pixels) > 0:
        
        move_speed_r = np.sqrt(np.mean(np.sum((meantracks_r[:,1:,:] - meantracks_r[:,:-1,:]) ** 2, axis=2), axis=1))
        move_speed_g = np.sqrt(np.mean(np.sum((meantracks_g[:,1:,:] - meantracks_g[:,:-1,:]) ** 2, axis=2), axis=1)) 

        # compare the speeds and difference.  
        intersect_pixel_speed_r = move_speed_r[intersect_pixels] 
        intersect_pixel_speed_g = move_speed_g[intersect_pixels] 

        green_bad = intersect_pixels[intersect_pixel_speed_g < intersect_pixel_speed_r]
        red_bad = intersect_pixels[intersect_pixel_speed_r < intersect_pixel_speed_g]

        new_move_r = move_r.copy()
        new_move_g = move_g.copy()
        
        new_move_r[red_bad] = False
        new_move_g[green_bad] = False


        return new_move_r, new_move_g

    else: 
        
        new_move_r, new_move_g = move_r, move_g
        
        return new_move_r, new_move_g
    
    
#==============================================================================
#   Finally this is the all-in-all module function, that pieces together the above.
#==============================================================================
def filter_red_green_tracks(tracks_r, tracks_g, img_shape, mag_thresh=0, frame2=1, spixel_neighbour_factor=1.2, overlap_thresh=0.7):
 
    """
    Filtering of superpixel tracks using the prior knowledge that red and green epithelial sheets should be separated on 2 sides and migrates towards each other. Works by constructing adjacency graphs on the superpixels that move and requires no image segmentation. 
    
    Auto-infers which side left or right sheets are.
    
    Inputs:
    -------
    tracks_r: (n_superpixels x n_frames x 2), unfiltered meantracks from optical flow tracking for the red channel.
    tracks_g: (n_superpixels x n_frames x 2), unfiltered meantracks from optical flow tracking for the green channel
    img_shape: (n_rows x n_cols) pixel dimensions of each frame
    mag_thresh: movement threshold for defining a moving superpixel in pixel units
    frame2: frame from frame0 for which we assess initially moving superpixel. This is a user-set cut-off that tries to balance seeing movement and before sheets touch.
    spixel_neighbour_factor: radial cut-off for defining neighbours given as a multiplicative factor of the average superpixel size.
    
    Outputs:
    --------
    meantracks_r: (n_superpixels x n_frames x 2), filtered final red superpixel tracks
    meantracks_g: (n_superpixels x n_frames x 2), filtered final green superpixel tracks
    
    """
    
    meantracks_r = tracks_r.copy()
    meantracks_g = tracks_g.copy()
    
    m, n = img_shape    
        
    #==============================================================================
    #  Attempt Filter superpixel tracks  
    #==============================================================================
    move_select_r = find_moving_tracks(meantracks_r, frame2=frame2)
    move_select_g = find_moving_tracks(meantracks_g, frame2=frame2)
    
    # use this to filter the red tracks, compensating for the bleedthrough
    keep_spixels_r = filter_tracks_superpixel_cliques(meantracks_r, move_select_r, thresh_factor=spixel_neighbour_factor)
    keep_spixels_r = fill_in_missing_superpixels( meantracks_r, keep_spixels_r, (m,n))
    
    keep_spixels_g = filter_tracks_superpixel_cliques(meantracks_g, move_select_g, thresh_factor=spixel_neighbour_factor)
    keep_spixels_g = fill_in_missing_superpixels( meantracks_g, keep_spixels_g, (m,n))
    

    #==============================================================================
    #     If need check for overlap 
    #==============================================================================
    n_spixels_r = meantracks_r.shape[0]
    n_spixels_g = meantracks_g.shape[0]
        
    if (len(keep_spixels_r) / float(n_spixels_r) > overlap_thresh) or (len(keep_spixels_g) / float(n_spixels_g) > overlap_thresh):
    
        # make new selections. 
        select_all = np.zeros(meantracks_g.shape[0], dtype=np.bool)
        select_r = select_all.copy()
        select_r[keep_spixels_r] = True
        select_g = select_all.copy()
        select_g[keep_spixels_g] = True
        
        move_select_r, move_select_g = check_overlap(meantracks_r, meantracks_g, select_r, select_g)
        
        keep_spixels_r = fill_in_missing_superpixels( meantracks_r, np.arange(len(move_select_r))[move_select_r], (m,n))
        keep_spixels_g = fill_in_missing_superpixels( meantracks_g, np.arange(len(move_select_g))[move_select_g], (m,n))
        
        
        # make new selections. 
        select_all = np.zeros(meantracks_g.shape[0], dtype=np.bool)
        select_r = select_all.copy()
        select_r[keep_spixels_r] = True
        select_g = select_all.copy()
        select_g[keep_spixels_g] = True
        
        
        # refine the keep superpixels    
        keep_spixels_r = filter_tracks_superpixel_cliques(meantracks_r, select_r, thresh_factor=spixel_neighbour_factor)
        keep_spixels_g = filter_tracks_superpixel_cliques(meantracks_g, select_g, thresh_factor=spixel_neighbour_factor)
        
        
        keep_spixels_r = fill_in_missing_superpixels( meantracks_r,  keep_spixels_r, (m,n))
        keep_spixels_g = fill_in_missing_superpixels( meantracks_g, keep_spixels_g, (m,n))
    #==============================================================================
    #   Dynamic Track propagation 
    #==============================================================================
    tracks_r_new, activated_r, keep_r = dynamic_propagation_spixels_fast(meantracks_r, keep_spixels_r, (m, n))
    tracks_g_new, activated_g, keep_g = dynamic_propagation_spixels_fast(meantracks_g, keep_spixels_g, (m, n))
    
    
    #==============================================================================
    #   Recheck that regions are actually moving and clean up one more time, dynamic propagation may have picked up some non moving tracks.... hm....  
    #==============================================================================
    
    still_move_r = check_move_all_frames(tracks_r_new, move_thresh=0)
    still_move_g = check_move_all_frames(tracks_g_new, move_thresh=0)
    
    # refine keep_r and keep_g accordingly. 
    keep_r = np.arange(len(still_move_r), dtype=np.int)[still_move_r]
    keep_g = np.arange(len(still_move_g), dtype=np.int)[still_move_g]
    

    #==============================================================================
    #   Final Cleanup  using track_superpixels. 
    #==============================================================================    
    select_all = np.zeros(tracks_r_new.shape[0], dtype=np.bool)
    select_r = select_all.copy()
    select_r[keep_r] = True

    select_g = select_all.copy()
    select_g[keep_g] = True


    keep_r_final = filter_tracks_superpixel_cliques_once(tracks_r_new, select_r, thresh_factor=spixel_neighbour_factor)
    final_meantracks_r = apply_filter_tracks(tracks_r_new, keep_r_final)
    
    keep_g_final = filter_tracks_superpixel_cliques_once(tracks_g_new, select_g, thresh_factor=spixel_neighbour_factor)
    final_meantracks_g = apply_filter_tracks(tracks_g_new, keep_g_final)
    
    
    return final_meantracks_r, final_meantracks_g
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    