#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 12:54:50 2017

@author: felix
@email: felixzhou1@gmail.com

This script and all the code within is licensed under the Ludwig License.

see README.md for details on usage.
 
"""

import numpy as np 
import pylab as plt 

    
def find_direction_wound(points, limits, spixel_size):
    
    """
    Finds the direction in which the epithelial sheet is moving in by looking at the position of the sheet points ...
    
    Inputs:
    -------
    points: (y,x) numpy array of points, (n_points x 2)
    limits: (n_rows, n_cols), essentially the image dimension.
    spixel_size: (float) average superpixel width
    
    Outputs:
    --------
    direction: a string either 'left' or 'right' designating the direction in which the superpixel moves. 
    """
    y_limit, x_limit = limits
    
    all_x = np.unique(points[:,1])
    all_y = np.unique(points[:,0])
    
    x_min = np.min(all_x)
    x_max = np.max(all_x)
    
    if x_max <= x_limit - 2 * spixel_size:
        # this is from left 
        direction = 'left'
        return direction
        
    if x_min >= 2*spixel_size:
        direction = 'right'
        return direction
        
    
def find_nearest_pairs_points(points_r, points_g):
    
    """
    Match the boundary points of the red sheet, (points_r) with those of the green sheet (points_g)
    
    n_sweeps - the number of horizontal strips dividing the image.
    n_points_to_keep - the number of points kept to approximate boundary for each strip.
    
    Inputs:
    -------
    points_r: (n_points_to_keep*n_sweeps, 2) boundary points of red sheet
    points_g: (n_points_to_keep*n_sweeps, 2) boundary points of green sheet
    
    Outputs:
    --------
    spixels_r: the green ids in points_g to match each r point to.
    spixels_g: the red ids in points_r to match each g point to.
    
    """
    
    from sklearn.metrics.pairwise import euclidean_distances
    
    n_pixels_r = points_r.shape[0]
    n_pixels_g = points_g.shape[0]
        
    cross_distance_r_g = euclidean_distances(points_r, points_g)
    
    if n_pixels_r <= n_pixels_g:
        neighbours_g = np.argsort(cross_distance_r_g, axis=1)[:,0]
        neighbours_r = np.arange(points_r.shape[0])
    else:
        neighbours_g = np.arange(points_g.shape[0])
        neighbours_r = np.argsort(cross_distance_r_g, axis=0)[0,:]
        
    spixels_r = np.arange(n_pixels_r)[neighbours_r]
    spixels_g = np.arange(n_pixels_g)[neighbours_g]

    return spixels_r, spixels_g
    
    
def baseline_als(y, lam, p, niter=10):
    
    """
    Code adapted from https://stackoverflow.com/questions/29156532/python-baseline-correction-library. 
    
    Implements paper of "Asymmetric Least Squares Smoothing" by P. Eilers and H. Boelens in 2005.
    
    Inputs:
    -------
    y: numpy vector of observations to fit.
    lam: (float) smoothness parameter, paper recommends 10**2 ≤ λ ≤ 10**9 for most applications.
    p: (float) asymmetry parameter, the higher this is the more asymmetric we consider the fitting errors, paper recommends 0.001 ≤ p ≤ 0.1 for positive peaks.
    niter: (int), the number of iterations to run the iterative fitting algorithm, the longer you run this, the more flat the baseline. Only a few iterations is needed. 
    
    Outputs:
    --------
    z: fitted baseline, same numpy vector size as y.
    
    """
    
    from scipy import sparse
    import numpy as np 
    from scipy.sparse.linalg import spsolve
    
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)

    for i in xrange(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)

    return z
    
    
def locate_thresh(array, thresh):
    
    """
    Find the first entry in array (left to right) which just above a defined cutoff given by thresh  (assumes array represents a function that is approximately monotonically decreasing).
    
    Inputs:
    -------
    array: a numpy array of numbers 
    thresh: (float) specified cutoff 
    
    Outputs:
    --------
    wound_close: the position of the array for which the value is just above thresh.
    """
    wound_close = -1
    
    for i in range(len(array)):

        val = array[i]
#        print val, thresh[i]
        if val <= thresh[i]:
            wound_close = i 
            break
        else:
            continue
    
    if wound_close == -1:
        return [] #nothing found
    else:
        return wound_close
    
    
def postprocess_img(img, close_ksize=5, size_factor=0.05):
    
    """
    Postprocess the binary mask from kmeans clustering in order to fill holes and remove small elements to grab the main epithelial sheet.
    
    Inputs:
    -------
    img: (n_rows x n_cols) binary image
    close_ksize: int, kernel size for binary_closing with a disk kernel.
    size_factor: (0-1 float) the proportion of image area below which are regarded as small objects and removed from the binary mask
    
    Outputs:
    --------
    filt: (n_rows x n_cols), post-processed binary image.
    """
    
    from scipy.ndimage.morphology import binary_fill_holes
    from skimage.morphology import disk, binary_closing, remove_small_objects 
    
    img_rows, img_cols = img.shape
    
    filt = binary_closing(img, disk(close_ksize))
    filt = remove_small_objects(filt, size_factor*img_rows*img_cols)        
    filt = binary_fill_holes(filt)
    
    return filt
    
    
    
def locate_boundary_points_sweepline_segmentation( areas_img, spixel_size, direction=None, n_sweeps=20, n_points_keep=1):
    
    """
    Given the binary mask of an epithelial sheet, locate the boundary points of the sheet and infer which vertical strip and direction the sheet is moving in.
    
    Inputs:
    -------
    areas_img: (n_rows x n_cols) binary image of the epithelial sheet.
    frame: int, which frame number to compute.
    spixel_size: (float) the average superpixel width.
    direction: (default None), which direction the sheet is moving, takes 'left' or 'right'
    n_sweeps: (int), the number of vertical divisions of the image.
    n_points_keep: (int) for each vertical strip, the number of points to sample to approximate the sheet boundary.
    
    Outputs:
    --------
    good_boundary_coords: the (y,x) coordinates, (n_points_keep*n_sweeps x 2) numpy array that mark the boundary.
    sweepline_seg_y: (n_points_keep*n_sweeps,) numpy vector which strip id the boundary points are in. 
    direction: 'left' or 'right' string, denoting the direction in which the epithelial sheet is moving in 
    """
    m, n = areas_img.shape
    
    limits = (m, n)
    
    # set up the possible points. 
    grid_X, grid_Y = np.meshgrid(range(n), range(m))
    
    # identify the points to consider. 
    move_points = np.vstack([grid_Y[areas_img>0].ravel(), grid_X[areas_img>0].ravel()]).T  # set up the points

    # next we define and find the direction of movements... (this can just be done by the area segmentation... )
    """
    # Determine direction ----> use the trick in the clean superpixels, result of this is left or right
    """
    if direction is None:    
        direction = find_direction_wound(move_points, limits, spixel_size)

    # check if blank image... 
    if np.sum(areas_img == 0) == m*n:
        # initial image is totally blank and worthless. set one pixel of the boundary to 1:
        if direction == 'right':
            move_points = np.vstack([np.arange(m, dtype=np.int), (n-1)*np.ones(m, dtype=np.int)]).T 

        if direction == 'left':
            move_points = np.vstack([np.arange(m, dtype=np.int), 0*np.ones(m, dtype=np.int)]).T 
    
#    print direction
    """
    # create the sweep lines intervals. 
    """
    y_lines = np.linspace(0, m, n_sweeps+1)
    y_bounds = np.floor(y_lines).astype(np.int)
    
    """
    # Sweep through the column ... 
    """
    # bin the points into the bounds
    spixel_regions = []
    choice = np.arange(move_points.shape[0])
    
    # bin the points according the y range. 
    for i in range(len(y_bounds)-1):
        lower_y = y_bounds[i]
        upper_y = y_bounds[i+1]
        spixel_regions.append(choice[np.logical_and(move_points[:,0] >= lower_y, move_points[:,0] < upper_y)])
        
        
    """
    Find the points to keep ... 
    """
    kept_points = []
    sweepline_seg = []
    # go through each y_bin and find the furthest x point 
    
    for i in range(len(spixel_regions)):
        spixel_region = spixel_regions[i]
        
        if len(spixel_region) > 0:
            spixel_region_x = move_points[spixel_region, 1]

            if direction == 'left':
                sort_order = np.argsort(spixel_region_x)[::-1]
                good_x = spixel_region[sort_order[:n_points_keep]]
                                       
            if direction == 'right':
                sort_order = np.argsort(spixel_region_x)
                good_x = spixel_region[sort_order[:n_points_keep]]
                                     
#            print direction  
            kept_points.append(good_x)
            sweepline_seg.append(i*np.ones(len(good_x), dtype=np.int))
            
    if n_points_keep == 1:
        kept_points = np.array(kept_points).ravel()
        sweepline_seg = np.array(sweepline_seg).ravel()
    if n_points_keep > 1:
        kept_points = np.hstack(kept_points)
        sweepline_seg = np.hstack(sweepline_seg)
        

    """ 
    # Filter the outlier points by statistical outlier analysis. 
    """
    good_points = move_points[kept_points] # get the goodness. 

    X = good_points[:,1]
    y = good_points[:,0]

    xline = np.arange(len(X))
    meanX = np.median(X)

    if np.std(X) > 1e-8:
        inlier_mask = np.logical_and( X < meanX + 2*np.std(X), X > meanX - 2*np.std(X) )
    else:
        inlier_mask = np.ones(len(X), dtype=np.bool)
    
    
    # return the coordinates and the sweepline...
    good_boundary_coords = good_points[inlier_mask]
    sweepline_seg_y = sweepline_seg[inlier_mask]

    return good_boundary_coords, sweepline_seg_y, direction

    

def kmeans_cluster(image, n_clusters=2, train_split=0.1):
    
    """
    Uses kmeans cluster to group the image intensities into n_clusters, sorted by image intensity, the lowest intensity pixels are grouped as 0 etc...
    
    Inputs:
    -------
    image: (n_rows x n_cols), grayscale image
    n_clusters: the number of clusters to partition the image intensity, the higher the number, the finer the partitioning, allows for better handling of lower intensity cells.
    train_split: (0-1 float) proportion of pixels to use for fitting the kmeans cluster. 
    
    Outputs:
    --------
    Y_new: (n_rows x n_cols), integer image, labelled by the cluster numbers.
    """
    
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    model = KMeans(n_clusters=n_clusters)
    
    img_rows, img_cols = image.shape
    
    X = image.reshape(-1,1).astype(np.float)

    # speed this section up by bootstrapping. 
    choose = np.arange(X.shape[0])
    np.random.shuffle(choose)
    
    X_train = X[choose[:int(train_split*len(choose))]]

    model.fit(X_train)
    Y = model.predict(X)
    Y = Y.reshape((img_rows, img_cols))
    
    
    intensity = [np.mean(image[:,:][Y==label]) for label in np.unique(Y)]
    sort_order = np.argsort(intensity)
    
    Y_new = np.zeros(Y.shape, dtype=np.int)

    for i in range(len(sort_order)):
        Y_new[Y==sort_order[i]] = i 
        
    Y_new = Y_new.astype(np.uint8)
        
    return Y_new
    

def wound_sweep_area_segmentation(vid_stack, spixel_size, max_frame, n_sweeps=50, n_points_keep=1, n_clusters=2, p_als=0.001, to_plot=False):
    
    """
    Inputs:
    --------
    vid_stack: (n_frames, n_rows, n_cols, 3),     
    spixel_size: average superpixel width (int)
    max_frame: maximum frame number to which to compute the gap closure over.
    n_sweeps: number of strips to divide vertically the image.
    n_points_keep: the number of points to keep for summarising the boundary.
    n_clusters: number of kmeans cluster for identifying the epithelial sheet. A larger number is better for segmenting weaker fluorescence/intensity staining.
    p_als: this controls the assymmetry in fitting the baseline. see stackoverflow: https://stackoverflow.com/questions/29156532/python-baseline-correction-library 
    to_plot: plot the diagnostic image or not to check fitting and inference.
    
    Outputs:
    --------
    wound_frame: predicted frame where gap closure occurs.
    
    """

    from skimage.filters.rank import median
    from skimage.morphology import square
    from skimage.util import img_as_float
    from skimage.filters import threshold_otsu
    from scipy.interpolate import UnivariateSpline
    import numpy as np 
     
    """
    1. read in video
    """
    r_channel = vid_stack[:,:,:,0]
    g_channel = vid_stack[:,:,:,1]

    r_channel = np.transpose(r_channel, (1,2,0))
    g_channel = np.transpose(g_channel, (1,2,0))

    """
    2. iterate over the frames and find the boundaries of segmentation
    """
    distances_time = []

    direction_r = None
    direction_g = None
        
    for frame in range(max_frame):
        
        """
        Preprocess the image and segmentate ...
        """
        img_r = r_channel[:,:,frame]
        img_g = g_channel[:,:,frame]

        # first do the median smoothing filter. 
        img_r = median(img_r, square(spixel_size))
        img_g = median(img_g, square(spixel_size))
        
        # turn into img float.
        img_r = img_as_float(img_r)
        img_g = img_as_float(img_g)
        
        m, n = img_r.shape
        
        filt_r = kmeans_cluster(img_r, n_clusters=n_clusters)
        filt_r = filt_r > 0 
        filt_r = postprocess_img(filt_r, close_ksize=5, size_factor=0.05)   
        
        filt_g = kmeans_cluster(img_g, n_clusters=n_clusters)
        filt_g = filt_g > 0 
        filt_g = postprocess_img(filt_g, close_ksize=5, size_factor=0.05)   
        
        """
        Locate boundary points.
        """
        boundary_points_r, sweep_cols_r, direct_r = locate_boundary_points_sweepline_segmentation( filt_r, spixel_size, direction=direction_r, n_sweeps=n_sweeps, n_points_keep=n_points_keep)
        boundary_points_g, sweep_cols_g, direct_g = locate_boundary_points_sweepline_segmentation( filt_g, spixel_size, direction=direction_g, n_sweeps=n_sweeps, n_points_keep=n_points_keep)
    
        if frame == 0:
            direction_r = direct_r
            direction_g = direct_g
        
        """
        Match boundary points along horizontal lines. 
        """
        # having located the number of points now we match up the points using nearest euclidean distance
        ids_r, ids_g = find_nearest_pairs_points(boundary_points_r, boundary_points_g)
        
        """
        Stitch adjacent boundary points. 
        """
        # now we stitch the points together... 
        points_r = boundary_points_r[ids_r]
        points_g = boundary_points_g[ids_g]

        """
        Compute the distance and make into curve. 
        """
        dist_points = np.sqrt((points_r[:,0] - points_g[:,0]) ** 2 + (points_r[:,1] - points_g[:,1]) ** 2)        
        distances_time.append(np.mean(dist_points))
        
    distances_time = np.array(distances_time).ravel()
        
    """
    Spline approximation and intersection finding. 
    """
    norm_dist = np.array(distances_time).ravel() / np.max(distances_time)
        
    # remember to add 1... to this..
    x_dist = np.arange(1, max_frame+1)
    
    spl = UnivariateSpline(x_dist, norm_dist, k=1, s = 0.1*np.var(norm_dist))
    spl_dist = spl(x_dist)
    
    n_length = len(spl_dist)
    
    # this is the normal affair ..... 
    tailed = False
    
    if norm_dist[-1] > norm_dist[0]:
        tailed = True
        
    if tailed:        
        aug_norm_dist = np.hstack([spl_dist[::-1], spl_dist[::-1], spl_dist[::-1]])
        fitted_dist = baseline_als(aug_norm_dist, lam=1e2, p=0.1*p_als, niter=10)
        fitted_dist = (fitted_dist[n_length:-n_length])[::-1]
    else:
        aug_norm_dist = np.hstack([spl_dist, spl_dist, spl_dist])
        
        fitted_dist = baseline_als(aug_norm_dist, lam=1e2, p=p_als, niter=10)
        fitted_dist = fitted_dist[n_length:-n_length]

    wound_frame = locate_thresh(spl_dist, thresh=fitted_dist + 2*np.std(fitted_dist))
    
#    print 'predicted wound heal time', wound_frame
    if to_plot:
        plt.figure(figsize=(6,4))
        plt.title('predicted wound heal frame number: %d' %(wound_frame))
        plt.plot(x_dist, norm_dist, 'ro', label='measured distances')
        plt.plot(x_dist, spl_dist, 'g-', label='fitted linear spline')
        plt.plot(x_dist, fitted_dist, 'k--', label='fitted baseline')
        plt.plot(x_dist, 2*np.std(fitted_dist) + fitted_dist, 'r--',label='fitted baseline + 2 s.t.d')
        plt.plot(wound_frame+1, norm_dist[wound_frame], 'b^', ms=15, label='predicted gap close frame')
        plt.legend(loc='best')
        plt.show()
    
    return wound_frame + 1 # remember to add one onto... 
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
    
    
    
    
    
    