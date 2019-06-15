# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 22:04:57 2018

@author: felix

set of tools for describing the wound boundary of wound healing assays. 
(may be quite experimental at present)

"""
import numpy as np 
import pylab as plt 

# =============================================================================
#   Wound Finding Tools
# =============================================================================
# option 1: detect boundary edges by image segmentation. 
def detect_edges_rgb( img_r, img_g, downscale=32, smooth=2, sigma=3):
    """ Detects edges of individual coloured sheets by running a Canny edge detector on dowsampled images.

    Parameters
    ----------
    img_r : numpy array
        (nrows, ncols) grayscale red channel image
    img_g : numpy array
        (nrows, ncols) grayscale green channel image
    downscale : int
        downsampling factor, resulting image will be reduced by 1./downscale times. downscale needs to divide image dimensions exactly.
    smooth : float
        width of the Gaussian used to smooth downsampled images
    sigma : float 
        width of the Gaussian used in the Canny detector. Use larger values if too many lines detected

    Returns
    -------
    labelled_r : numpy array
        (nrows/downscale, ncols/downscale) integer array image giving the edge of the downsampled red sheet
    labelled_g : numpy array
        (nrows/downscale, ncols/downscale) integer array image giving the edge of the downsampled green sheet
    img_r_ : numpy array
        (nrows/downscale, ncols/downscale) downsampled red sheet after gaussian smoothing
    img_g_ : numpy array
        (nrows/downscale, ncols/downscale) downsampled green sheet after gaussian smoothing

    """
    from skimage.filters import gaussian 
    from skimage.feature import canny 
    from skimage.measure import label 
    from skimage.exposure import rescale_intensity, equalize_hist
    from skimage.filters.rank import median
    from skimage.morphology import square
    import numpy as np 
    from .Utility_Functions.image import pool_numpy
    
    img = np.dstack([img_r, img_g])
    img = pool_numpy(img, factor=(downscale, downscale, 1), func=np.max)
    
    img_r = img[:,:,0] / 255. ; img_g = img[:,:,1] / 255.
    img_r = rescale_intensity(img_r)
    img_g = rescale_intensity(img_g)
   
    # implement smoothing to suppress noise. 
    # consider switching to using anisotropic filtering? 
    img_r = gaussian(img_r, sigma=smooth)  
    img_g = gaussian(img_g, sigma=smooth)
    
    img_r_ = img_r.copy()
    img_g_ = img_g.copy()
    
    # use Canny filter to locate edges.     
    edges_r = canny(img_r, sigma=sigma) ; edges_g = canny(img_g, sigma=sigma)
    
    labelled_r = label(edges_r)
    labelled_g = label(edges_g)
    
    return labelled_r, labelled_g, img_r_, img_g_
    

def detect_edges_gray( img, downscale=32, smooth=2, sigma=3, return_img=False):
    """ Detects edges of a single colour sheet by running a Canny edge detector on dowsampled images.

    Parameters
    ----------
    img : numpy array
        (nrows, ncols) grayscale image
    downscale : int
        downsampling factor, resulting image will be reduced by 1./downscale times. downscale needs to divide image dimensions exactly.
    smooth : float
        width of the Gaussian used to smooth downsampled images
    sigma : float 
        width of the Gaussian used in the Canny detector. Use larger values if too many lines detected
    return_img : bool
        if True, additionally return the downsampled smooth intermediate image

    Returns
    -------
    labelled : numpy array
        (nrows/downscale, ncols/downscale) integer array image giving the edge of the downsampled sheet
    img_ : numpy array
        if return_img is True, return the (nrows/downscale, ncols/downscale) downsampled image.

    """
    from skimage.filters import gaussian 
    from skimage.feature import canny 
    from skimage.measure import label 
    import numpy as np 
    from .Utility_Functions.image import pool_numpy
    
    img = np.dstack([img, img])
    img = pool_numpy(img, factor=(downscale, downscale, 1), func=np.max)
    
    img_ = img[:,:,0] / 255. 
    img_ = gaussian(img_, sigma=smooth) 
    edges = canny(img_, sigma=sigma) 
    labelled = label(edges)  
    
    if return_img:
        return labelled, img_
    else:
        return labelled
    
# use this function to preserve continuity in the detected wound boundary edges!.  
def filter_edges(labelled_edges, seed_img=None):
    """
    """
    # return the closest to the centre, if no seed. 
    from skimage.measure import regionprops
    import numpy as np 
    from skimage.feature import match_template
    
    uniq_labels = np.unique(labelled_edges)[1:]
    regs = regionprops(labelled_edges)

    line_centroids = []
    # derive the centroid coordinate. 
    for reg in regs:
        centre = reg.centroid
        line_centroids.append(centre[::-1]) # put in x,y coords? 
    
    line_centroids = np.vstack(line_centroids)
    
    if seed_img is not None:
        # choose that which is closest to the seed by cross-correlating the masks
        xcorr_masks = []    
        for lab in uniq_labels:
            test_mask = labelled_edges==lab
            ref_mask = seed_img.copy()
            
            xcorr_masks.append(np.max(match_template(ref_mask.astype(np.float),test_mask.astype(np.float))))
        return labelled_edges == uniq_labels[np.argmax(xcorr_masks)]    
    else:
        m, n = labelled_edges.shape
        centre_img = np.hstack([n/2., m/2.])
        dists_centre = np.linalg.norm(line_centroids - centre_img[None,:], axis=1)
        return labelled_edges == uniq_labels[np.argmin(dists_centre)]
            
        
# utility function to match the nearest neighbour pairs between spixel points. 
def find_all_epsilon_pairs(dist_matrix, min_dist, max_dist, nearest_N=2):
    """ Given a distance matrix find the maximum nearest_N neighbours of each point whose distance, d satisfies min_dist<=d<=max_dist
    
    Parameters
    ----------
    dist_matrix : numpy array
        precomputed distance matrix between two point sets
    min_dist : float
        lower distance threshold (in pixels or other relevant unit of the distance matrix)
    max_dist : float
        upper distance threshold (in pixels or other relevant unit of the distance matrix) 
    nearest_N : int
        the maximum number of nearest neighbours to return

    Returns
    -------
    selected_ids : numpy array
        an (n_matches, 2) integer array, whose first column is the row_id and second column gives the column_id
    
    """
    import numpy as np
    select2 = np.arange(dist_matrix.shape[1], dtype=np.int)
    
    selected_ids = []
    
    for i in range(len(dist_matrix)):
        test = dist_matrix[i]
        match = select2[np.logical_and(test>=min_dist, test<=max_dist)]
        if len(match) > 0:
            match = select2[np.argsort(test)[:nearest_N]]
            selected_ids.append(np.vstack([i*np.ones(len(match)), match]).T)
            
    if len(selected_ids) > 0:
        selected_ids = np.vstack(selected_ids)
            
    return selected_ids
    
    
def generate_pair_distances(coords_r, coords_g, inf_distance=1000):
    """ Given two 2-D point sets construct the pairwise distance matrix between points in each set independently without self-connections between points. 
    
    Parameters
    ----------
    coords_r : numpy array
        (N_r, d) array of points where d is the dimensionality e.g. 2 for 2-D points and N_r the number of points
    coords_g : numpy array
        (N_g, d) array of points where d is the dimensionality e.g. 2 for 2-D points and N_g the number of points
    inf_distance : float
        a very large float, which serves the same role as the machine precision epsilon. The number should be bigger than any distance between a point and other points. It serves to avoid finding self-connections in subsequent analysis

    Returns
    -------
    sim_r : numpy array
        an (N_r, N_r) distance matrix between all coords_r as measured by the Euclidean distance between points
    sim_g : numpy array
        an (N_g, N_g) distance matrix between all coords_g as measured by the Euclidean distance between points
    
    """
    from sklearn.metrics.pairwise import pairwise_distances
    import numpy as np
    
    sim_r = pairwise_distances(coords_r)
    sim_g = pairwise_distances(coords_g)
    
    # make self-connections invalid. 
    sim_r = sim_r + np.diag(inf_distance*np.ones(len(sim_r)))
    sim_g = sim_g + np.diag(inf_distance*np.ones(len(sim_g)))
    
    return sim_r, sim_g
    
    
def remove_singletons(coords_r, coords_g, dist_thresh, inf_distance=1000, min_no=3):
    """ Given two sets of coordinates, cleans them by removing points in each set which have less than the min_no of neighbours where a neighbour is defined as another point whose Euclidean distance is <= dist_thresh (disregarding the point itself)
    
    Parameters
    ----------
    coords_r : numpy array
        (N_r, d) array of points where d is the dimensionality e.g. 2 for 2-D points and N_r the number of points
    coords_g : numpy array 
        (N_g, d) array of points where d is the dimensionality e.g. 2 for 2-D points and N_g the number of points
    dist_thresh : float 
        upper distance threshold (in pixels or other relevant unit of the distance matrix) for a point to be considered a neighbour
    inf_distance : float
        a very large float, which serves the same role as the machine precision epsilon. The number should be bigger than any distance between a point and other points. It serves to avoid finding self-connections in subsequent analysis
    min_no : int 
        the minimum number of neighbours a point should have to be retained

    Returns
    -------
    clean_coords_r : numpy array 
        the filtered 'red' coordinate set
    clean_coords_g : numpy array
        the filtered 'green' coordinate set
    
    """
    import numpy as np 
    p_r, p_g = generate_pair_distances(coords_r, coords_g, inf_distance=inf_distance)

    select_r = np.sum(p_r<=dist_thresh, axis=1) >= min_no
    select_g = np.sum(p_g<=dist_thresh, axis=1) >= min_no

    clean_coords_r = coords_r[select_r]
    clean_coords_g = coords_g[select_g]

    return clean_coords_r, clean_coords_g
    
    
def sample_and_match_red_green_segs(red_mask, green_mask, n_samples=1000, min_dist=10, dist_thresh=2, nearest_N=2, debug=False):
    """ sample the red and green points in the red and green segmentation masks respectively and attempt to pair red and green points together based on spatial proximity to find points that lie at the boundary.
    
    Parameters
    ----------
    red_mask : numpy array
        (nrows x ncols) binary segmentation mask based on the red (first) colour channel
    green_mask : numpy array
        (nrows x ncols) binary segmentation mask based on the green (second) colour channel
    n_samples : int
        number of points to sample from the red and green mask
    min_dist : float
        lower distance threshold (in pixels or other relevant unit of the distance matrix) for a positive neighbour
    dist_thresh : float
        upper distance threshold (in pixels or other relevant unit of the distance matrix) for a positive neighbour
    nearest_N : int
        the maximum number of nearest neighbours to consider
    debug : bool
        if True, plots the found points for visual checking

    Returns
    -------
    select_coords_r : [] or numpy array
        if no matches are found given the settings a [] is returned else a numpy array of new coordinate positions with first colour from red channel is returned
    select_coords_g : [] or numpy array
        if no matches are found given the settings a [] is returned else a numpy array of new matched coordinate positions of the second colour from green channel is returned

    """
    from sklearn.metrics.pairwise import pairwise_distances
    import numpy as np 
    nrows, ncols = red_mask.shape
    X, Y = np.meshgrid(np.arange(ncols), np.arange(nrows))
    
    x_r = X[red_mask].ravel()
    y_r = Y[red_mask].ravel()

    x_g = X[green_mask].ravel()
    y_g = Y[green_mask].ravel()

    # now sample... 
    select_r = np.arange(len(x_r))
    select_g = np.arange(len(x_g))
    np.random.shuffle(select_r)
    np.random.shuffle(select_g)
    
    x_r_ = x_r[select_r[:n_samples]]
    y_r_ = y_r[select_r[:n_samples]]
    x_g_ = x_g[select_g[:n_samples]]
    y_g_ = y_g[select_g[:n_samples]]


    if debug:
        fig, ax = plt.subplots()
        ax.plot(x_r_, y_r_, 'ro')
        ax.plot(x_g_, y_g_, 'go')
        plt.ylim([nrows,0])
        plt.xlim([0,ncols])
        plt.show()
    
    """
    Compute the pairwise distances between R and G points 
    """
    coords_r = np.vstack([x_r_, y_r_]).T
    coords_g = np.vstack([x_g_, y_g_]).T

    """
    Remove Singletons 
    """
    coords_r, coords_g = remove_singletons(coords_r, coords_g, max_dist=dist_thresh, inf_distance=2*np.max([nrows,ncols]))
    p = pairwise_distances(coords_r, coords_g)
    valid_pairs = find_all_epsilon_pairs(p, min_dist=min_dist, dist_thresh=dist_thresh, nearest_N=nearest_N)

    """
    stack together valid pairs and filter for unique
    """
    if len(valid_pairs) > 0: 
#        print valid_pairs.shape
        unique_r = np.unique(valid_pairs[:,0]).astype(np.int) # filter for the unique r and g coords
        unique_g = np.unique(valid_pairs[:,1]).astype(np.int)
        
        if debug:
        # graph for checking . 
            fig, ax = plt.subplots()
            ax.plot(coords_r[unique_r,0], coords_r[unique_r,1], 'ro')
            ax.plot(coords_g[unique_g,0], coords_g[unique_g,1], 'go')
            plt.ylim([nrows,0])
            plt.xlim([0,ncols])
            plt.show()

        select_coords_r = coords_r[unique_r]
        select_coords_g = coords_g[unique_g]
            
        return select_coords_r, select_coords_g
    else:
        select_coords_r = []
        select_coords_g = []
        return select_coords_r, select_coords_g


def match_red_green_segs(coords_r, coords_g, shape, min_dist=0, max_dist=2, nearest_N=1):
    """ exhaustively enumerate and pair red and green points given a coordinate list

    Parameters
    ----------
    coords_r : numpy array
        (N_r x d) array of points where d is the dimensionality e.g. 2 for 2-D points and N_r the number of points
    coords_g : numpy array
        (N_g x d) array of points where d is the dimensionality e.g. 2 for 2-D points and N_g the number of points
    shape : array-like
        tuple specifying the original image dimensions i.e. (nrows, ncols)
    min_dist : float
        lower distance threshold (in pixels or other relevant unit of the distance matrix) for a positive neighbour
    max_dist : float
        upper distance threshold (in pixels or other relevant unit of the distance matrix) for a positive neighbour
    nearest_N : int
        the maximum number of nearest neighbours to consider
    
    Returns
    -------
    select_coords_r : [] or numpy array
        if no matches are found given the settings a [] is returned else a numpy array of new coordinate positions with first colour from red channel is returned
    select_coords_g : [] or numpy array
        if no matches are found given the settings a [] is returned else a numpy array of new matched coordinate positions of the second colour from green channel is returned

    """
    from sklearn.metrics.pairwise import pairwise_distances
    import numpy as np 
    
    nrows,ncols = shape
    
    p = pairwise_distances(coords_r, coords_g)
    valid_pairs = find_all_epsilon_pairs(p, min_dist=min_dist, max_dist=max_dist, nearest_N=nearest_N)
    
    # filter for uniques. 
    if len(valid_pairs) > 0: 
        
        unique_r = np.unique(valid_pairs[:,0]).astype(np.int) # filter for the unique r and g coords
        unique_g = np.unique(valid_pairs[:,1]).astype(np.int)
        
        return coords_r[unique_r], coords_g[unique_g]
    else:
        return [], []


def eval_boundary_coords(spl, yline):
    """ wrapper to evaluate the corresponding x-coordinates given the y-coordinates according to a learnt spline model, spl

    returns :math:`(xline,yline)` where :math:`xline=spl(yline)`

    Parameters
    ----------
    spl : SciPy spline object or equivalent
        spline model for the boundary coords
    yline : numpy array
        the array of y-coordinates (row positions) to get the corresponding x-coordinates for

    Returns
    -------
    line_coords : numpy array
        an array of (x,y) coordinates describing the wound boundary after using the spline model to get the x-coordinate from the y-coordinate

    """
    import numpy as np 
    
    xline = np.polyval(spl, yline)
    line_coords = np.vstack([xline, yline]).T
    
    return line_coords
    
# def find_stable_wound_frame(coefs_time, min_length=10):
#     """ 
#     """
#     import numpy as np 
#     select = np.arange(len(coefs_time))
#     curve_valid = np.hstack(coefs_time) > 0

#     all_good_times = select[curve_valid]

#     stretches = [] 
#     # group into consecutive regions. 
#     stretch = [all_good_times[0]] # add this in already. 
#     for i in range(len(all_good_times)-1):
#         time0 = all_good_times[i]
#         time1 = all_good_times[i+1]

#         if (time1 - time0) == 1:
#             stretch.append(time1)
#             # last time whatever happens we append. 
#             if i == len(all_good_times) - 2:
#                 stretches.append(stretch)
#         else:
#             stretches.append(stretch)
#             stretch=[]

#     # count the lengths and if satisfies min length criteria take the first frame.
#     for s in stretches:
#         len_stretch = len(s)
#         if len_stretch >= min_length:
#             return s[0]
#         else:
#             continue

def stack_curves(curves, n_samples=100, shape=None):
    """ Given a list of variable length curves (2-dimension coordinate list) compile these into the same length array. The output array will be of dimensions (n_curves x min_len x d) where min_len is the minimum number of points of all curves and d is the dimensionality of the coordinates.
    
    Parameters
    ----------
    curves : list of numpy arrays
        list of 2-dimension coordinates (len_curve x d) where len_curve is the size of the curve and d is the dimensionality of coordinates e.g. 2 for 2-D images.
    n_samples : int 
        length of each curve if no valid curve (length=0) found
    shape : tuple or None (default: None)
        shape of each curve (n x d) where n is the number of points per curve

    Returns
    -------
    lines : numpy array
        (n_curves x shape[0] x shape[1]) or  (n_curves x min_len x 2) stacked array of line coordinates or else a (n_curves x n_samples x 2) array of NaN's
    
    """
    lines = []
    lens = np.hstack([len(c) for c in curves])
    select = lens[lens>0]
    if len(select) == 0: 
        # there is nothing.
        lines = np.ones((len(curves), n_samples, 2))
        lines[:] = np.nan # nothing. 
    else:
        min_len = np.min(lens[lens>0])

        for c in curves:
            if len(c) == 0:
                if shape is None:
                    lines.append(np.ones((min_len,2))*np.nan)
                else:
                    lines.append(np.ones(shape)*np.nan)
            else:
                if shape is None:
                    lines.append(c[:min_len])
                else:
                    lines.append(c)
    lines = np.array(lines)

    return lines


def find_boundary_lines(boundary_curves, yline):
    """ Wrapper function to eavluate a list of spline functions that capture the boundary e.g. for boundary in each frame to get the x-coordinates given the y-coordinates

    Parameters
    ----------
    boundary_curves : list of Scipy spline functions
        list of Scipy spline functions such that each spline function given y-coordinates gives the corresponding x-coordinates
    yline : numpy array
        the list of y-coordinates to evaluate each spline function at to get the corresponding x-coordinates

    Returns 
    -------
    lines : numpy array
        (n_curves x len_curve) a stacked array of all corresponding x-coordinate for each spline 

    """
    import numpy as np
    lines = []
    
    for ii in range(len(boundary_curves)):
        spl_curve = boundary_curves[ii]

        if len(spl_curve) == 0:
            lines.append(np.ones(len(yline))*np.nan)
        else:
            b_curve = eval_boundary_coords(spl_curve, yline)
            lines.append(b_curve[:,0]) # only store the x coordinates. 
    lines = np.vstack(lines)

    return lines

#==============================================================================
#   Cleaning up tools.
#==============================================================================
def outlier_detect_axis(X, axis=0, thresh=1.5):
    """ detect outlier points along a particular axis. 

    points along the designated numpy array axis are classified as outliers if they are more than a distance thresh * standard deviation of points from the median

    Parameters
    ----------
    X : numpy array
        an (N,d) array of coordinates where d is the dimensionality
    axis : int (default: 0)
        which of the coordinate columns to use
    thresh : float
        the multiple of standard deviation for which we consider points as outliers

    Returns
    -------
    inlier_vals : numpy bool array
        a boolean array specifying which points to keep based on the selected axis

    """
    import numpy as np 
    x = X[:,axis] # which of the coordinates will we use?

    thresh_lower = np.median(x) - thresh*np.std(x)
    thresh_upper = np.median(x) + thresh*np.std(x)
    
    inlier_vals = np.logical_and(x>=thresh_lower, x<=thresh_upper) # standard gaussian type fitting.
    return inlier_vals
            
    
def outlier_detect_graph(points, thresh=10):
    """ Outlier point detection based on constructing a graph between the points and identifying 'communities'. Points belonging to the largest community is kept

    Parameters
    ----------
    points : numpy array
        an (N,d) coordinate list 
    thresh : float
        maximum distance to consider a point as a neighbour

    Returns
    -------
    filt_points : numpy array
        a reduced version of points in the maximum connected graph component

    """
    import networkx as nx 
    import numpy as np 
    from sklearn.metrics.pairwise import pairwise_distances
    
    adj_matrix = pairwise_distances(points)
    nx_graph = nx.from_numpy_matrix(adj_matrix<=thresh)
    
    # get the connected components and sort by the largest connected... 
    cnn = sorted(nx.connected_components(nx_graph), key = len, reverse=True) # largest first.

    filt_points = points[np.array(list(cnn[0])).ravel()]
    return filt_points

# test with weighted polyfit?
# this was initially a test to see if we can play with the number of points but meh...    
#def fit_polyfit(x,y, max_x, bins=50, n_points=4, degree=5):
#    
#    from scipy.interpolate import UnivariateSpline
#    
#    # here we will fit a polynomial and use a robust fitting algorithm based on weighting the distance. naively we will first do this using equal binning.  
#    weights_x = np.zeros(len(x))
#    
#    xbins = np.linspace(0, max_x, bins+1)
#    
#    for i in range(bins):
#        x1 = xbins[i]
#        x2 = xbins[i+1]
#
#        select_x = np.logical_and(x>=x1, x<x2)
#        if np.sum(select_x) > 0: 
##            weights_x[select_x] = 1./np.abs(np.sum(select_x)-n_points) # maximum weight if there is only one value? hm... thats not good either... 
#            weights_x[select_x] = np.sum(select_x)
#    # we deal with outlier detection. by findiing components not close. 
##    p = np.polyfit(y,x, deg=degree, w=weights_x) #use the weights ... 
#    p = np.polyfit(y,x, deg=degree)
#    
#    return p
    
def fit_polyfit(x, y, degree=2):
    """ fit a polynomial f such that x = f(y)

    Parameters
    ----------
    x : numpy array 
        list of x-coordinates
    y : numpy array
        list of corresponding y-coordinates
    degree : int 
        degree of the polynomial to fit e.g. 2 for quadratic, 3 for cubic

    Returns
    -------
    p : polynomial coefficients
        result of calling np.polyfit, c.f. https://docs.scipy.org/doc/numpy/reference/generated/numpy.polyfit.html

    """
    import numpy as np 
    
    p = np.polyfit(y,x, deg=degree)
    
    return p


def test_y_density(points, max_y, bins=10, frac_thresh=0.5):
    """ compute the number of points when we divide the partition the image rows (y-coordinates) into a number of equal sized strips or bins where the number is given by bins. Points are considered dense if the fraction of occupied bins is greater than a fraction threshold given by frac_thresh.
   
    Parameters
    ----------
    points : numpy array
        (x,y) array of coordinates
    max_y : int
        maximum number of image rows
    bins : int
        number of 'bins' to divide the y-range [0,max_y] into
    frac_thresh : float (0-1)
        the fraction of bins occupied for each strip 

    Returns
    -------
    y_dense : bool
        whether the occupied fraction of bins is greater than the threshold frac_thresh

    """
    import numpy as np 
    y_bins = np.linspace(0, max_y, bins+1)
    
    y_pos = points[:,1]
    counts = []

    for i in range(len(y_bins)-1):
        y1 = y_bins[i]
        y2 = y_bins[i+1]

        select_ = np.logical_and(y_pos>=y1, y_pos<y2)
        c = np.sum(select_)
        counts.append(c)
       
    counts = np.hstack(counts)
    y_frac_ = np.sum(counts>0) / float(len(counts))
       
    y_dense = y_frac_ > frac_thresh
    return y_dense
    

def return_point_density(points, shape, sigma=3):
    """ Generates a spatial density map of the coordinates by mapping the points back into the image and applying a Gaussian filter of width sigma 
    
    Parameters
    ----------
    points : numpy array
        (x,y) array of coordinates
    shape : array-like
        shape of the original single-channel image
    sigma : int
        width of the sigma in the Gaussian used to smooth the point density to generate a continuous heatmap.

    Returns
    -------
    density : numpy array
        numpy array of size given by shape showing a heatmap of the coordinate density
    
    """
    import numpy as np 
    from skimage.filters import gaussian
    
    density = np.zeros(shape)
    density[points[:,1], points[:,0]] += 1
    density = gaussian(density , sigma=sigma)
    
    return density 
    
"""
Asymmetric least squares fitting tools
"""
def baseline_als(y, lam, p, niter=10):
    """ Asymmetric least squares smoothing of 1D signal to extract baseline (assumes positive residuals)
    Implements paper of "Asymmetric Least Squares Smoothing" by P. Eilers and H. Boelens (2005), [1]. 
    Code adapted from https://stackoverflow.com/questions/29156532/python-baseline-correction-library.

    Parameters
    ----------
    y : numpy array
        numpy vector of observations to fit.
    lam : float 
        smoothness parameter, paper recommends 10**2 ≤ λ ≤ 10**9 for most applications.
    p : float
        asymmetry parameter, the higher this is the more asymmetric we consider the fitting errors, paper recommends 0.001 ≤ p ≤ 0.1 for positive peaks.
    niter : int
        the number of iterations to run the iterative fitting algorithm, the longer you run this, the more flat the baseline. Only a few iterations is needed. 
    
    Returns
    -------
    z : numpy array
        fitted baseline, same numpy vector size as y.

    References
    ----------
    .. [1] Eilers PH, Boelens HF. "Baseline correction with asymmetric least squares smoothing." Leiden University Medical Centre Report. 2005 Oct 21;1(1):5.
    
    """
    from scipy import sparse
    from scipy.sparse import linalg
    import numpy as np
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in xrange(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = linalg.spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z
 
def baseline_als_direction(y, lam, p, niter=10, direction_x='right'):
    """ Asymmetric least squares smoothing of 1D signal to extract baseline given the direction 'left' or 'right'

    Modified version of the :func:`baseline_als` above to bias the residuals to postive or negative depending on the x-direction of movement (right or left respectively)
    This version uses the maximum or minimum 'y'-value for biasing. :func:`baseline_als_direction1` instead uses the distance to the image boundaries.

    Parameters
    ----------
    y : numpy array
        numpy vector of observations to fit.
    lam : float 
        smoothness parameter, paper recommends 10**2 ≤ λ ≤ 10**9 for most applications.
    p : float
        asymmetry parameter, the higher this is the more asymmetric we consider the fitting errors, paper recommends 0.001 ≤ p ≤ 0.1 for positive peaks.
    niter : int
        the number of iterations to run the iterative fitting algorithm, the longer you run this, the more flat the baseline. Only a few iterations is needed. 
    direction_x : str 
        either 'left' or 'right' to bias the asymmetry more towards moving sheet from the left or right respectively

    Returns
    -------
    z : numpy array
        smoothed version of y

    """
    from scipy import sparse
    from scipy.sparse import linalg
    import numpy as np
    # # for each y compute the difference
    # if direction_x == 'right':
    #     dist = np.abs(lims[1] - y) # pull everything this way. 
    # if direction_x == 'left':
    #     dist = np.abs( y - lims[0]) # pull everything to the left.    

    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in xrange(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = linalg.spsolve(Z, w*y)
        # mid_z = np.mean(z)
        if direction_x == 'right':
            mid_z = np.max(y)
            w = p * (y > mid_z) + (1-p) * (y < mid_z) # how much to weight this? or just use the median? 
        if direction_x == 'left':
            mid_z = np.min(y)
            w = p * (y < mid_z) + (1-p) * (y>mid_z)
    return z

def baseline_als_direction1(y, lam, p, lims, niter=3, direction_x='right'):
    """ Asymmetric least squares smoothing of 1D signal to extract baseline given the direction 'left' or 'right'

    Modified version of the :func:`baseline_als` above to bias the residuals to postive or negative depending on the x-direction of movement (right or left respectively)
    This version uses the maximum or minimum 'y'-value for biasing. :func:`baseline_als_direction` instead uses the distance to the image boundaries.

    Parameters
    ----------
    y : numpy array
        numpy vector of observations to fit.
    lam : float 
        smoothness parameter, paper recommends 10**2 ≤ λ ≤ 10**9 for most applications.
    lims : array-like
        shape of the image
    p : float
        asymmetry parameter, the higher this is the more asymmetric we consider the fitting errors, paper recommends 0.001 ≤ p ≤ 0.1 for positive peaks.
    niter : int
        the number of iterations to run the iterative fitting algorithm, the longer you run this, the more flat the baseline. Only a few iterations is needed. 
    direction_x : str 
        either 'left' or 'right' to bias the asymmetry more towards moving sheet from the left or right respectively

    Returns
    -------
    z : numpy array
        smoothed version of y

    """
    from scipy import sparse
    from scipy.sparse import linalg
    import numpy as np

    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in xrange(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = linalg.spsolve(Z, w*y)

        if direction_x == 'right':
            dists = np.abs(y-lims[0]) ** 2
            dists = dists/float(np.max(dists)) 
            w = (p)*dists 

        if direction_x == 'left':
            dists = np.abs(lims[1] - y) ** 2
            dists = dists/float(np.max(dists))
            w = (p) * dists
    return z

def generalised_sigmoid(x, cutoff, shape=1, slope=1):
    """ Computes a generalised sigmoid function based on the generalised logistic function which generalises the standard definition of a sigmoidal function to allow variations to shape and skew.

    :math:`\mathrm{sigmoid} = \\frac{1}{(1+\exp(-\\mathrm{slope}(\mathrm{x}-\mathrm{cutoff})))^{1/\mathrm{shape}}}`

    Parameters
    ----------
    x : numpy array
        input 1-D array of x values to evaluate for
    cutoff : float
        the shift in x value, this dictate approximately when the maximum slope is reached
    shape : float
        this parameter determines the skew, affecting which asymptote the function will be biased towards
    slope : float
        this parameter determines the growth rate of the slope

    Returns 
    -------
    sigmoid : numpy array
        evaluated generalised sigmoid function values

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Generalised_logistic_function
    """
    sigmoid = 1./(1+np.exp(-slope*(x-cutoff)))**(1./shape)
                
    return sigmoid

def baseline_als_density(x, y, lam, p, radius, niter=10, N_cutoff=5, shape=1, slope=1):
    """ Modifies the asymmetric least squares fitting procedure :func:`baseline_als` which moves according to the sign of the residual between y and z, to move towards density of 1d points instead. This gives improved results when filtering spatial (x,y) coordinates.
    
    point density is estimated by the number of neighbours in the given specified radius and then gated by computing a generalised sigmoid function which basically soft-binarises the concept of 'not dense' and 'dense'

    Parameters
    ----------
    x : numpy array 
        numpy vector of x-coordinates
    y : numpy array
        numpy vector of y-coordinates
    p : float
        asymmetry parameter, the higher this is the more asymmetric we consider the fitting errors, paper recommends 0.001 ≤ p ≤ 0.1 for positive peaks.
    radius : float
        the radial distance from each point to count the number of neighbouring points
    niter : int
        the number of iterations to run the iterative fitting algorithm, the longer you run this, the more flat the baseline. Only a few iterations is needed. 
    N_cutoff : int
        the cutoff value used in :func:`generalised_sigmoid`
    shape : float
        the shape value used in :func:`generalised_sigmoid`
    slope : float
        the slope value used in :func:`generalised_sigmoid`

    Returns
    -------
    z : numpy array
        smoothed version of y

    """
    from scipy import sparse
    from scipy.sparse import linalg
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    pts = np.vstack([x,y]).T
    model = NearestNeighbors(radius=radius)
    model.fit(pts)

    nbrs = model.radius_neighbors(pts, radius=radius, return_distance=False)
    nbrs = np.hstack([len(n) for n in nbrs]) # number of neighbours within the specified radius. 

    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in xrange(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = linalg.spsolve(Z, w*y)
        # w = p * (y > z) + (1-p) * (y < z)
        ind = generalised_sigmoid(nbrs, cutoff=N_cutoff, shape=shape, slope=slope)
        # w = p*(nbrs>N_cutoff) + (1-p)*(nbrs<N_cutoff)
        w = p*ind + (1-p)*ind
    return z
    
    
#def asymmetric_fit(x,y, lam, p, last_state=None, hysteresis=5):
#    
#    # determine whether to flip the estimation based on which lies more on either side. 
#    from scipy.stats import linregress
#    
#    slope, intercept, r_value, p_value, std_err = linregress(y, x)
#    
#    xlinear = y*slope + intercept
#    
#    # looks like need to do a polynomial fitting and then determine either side. 
#    n_less = np.sum(x<xlinear)
#    n_greater = np.sum(x>xlinear)
#    
#    if n_greater > n_less + hysteresis:
#        print 'flipping: %d/%d' %(n_less, n_greater)
#        xfit = baseline_als(-x, lam, p, niter=10)
#        return -xfit, 'flip'
#    elif n_less > n_greater + hysteresis:
#        print 'no flip: %d/%d' %(n_less, n_greater)
#        xfit = baseline_als(x, lam, p, niter=10)
#        return xfit, 'no_flip'
#    else:
#        if last_state == 'flip':
#            xfit = baseline_als(-x, lam, p, niter=10)
#            return -xfit, 'flip'
#        elif last_state == 'no_flip':
#            xfit = baseline_als(x, lam, p, niter=10)
#            return xfit, 'no_flip'
            
            
def asymmetric_fit(x,y, lam, p, flip_state=None):
    """ Utility function that helps to determine whether to flip the residual in the :func:`baseline_als` according to whether an epithelial sheet is moving to the left or the right

    Parameters
    ----------
    x : numpy array 
        1-D array of x-coordinates
    y : numpy array
        1-D array of y-coordinates (not used)
    p : float
        asymmetry parameter, the higher this is the more asymmetric we consider the fitting errors, paper recommends 0.001 ≤ p ≤ 0.1 for positive peaks.
    flip_state : str (default: None)
        either 'flip' or 'no_flip'. If 'flip' the x-coordinates are multiplied by -1 to get -x to get negative residuals

    Returns
    -------
    xfit : numpy array
        returned 1-D array of filtered x-coordinates

    """
    # determine whether to flip the estimation based on which lies more on either side. 
    if flip_state == 'flip':
        xfit = baseline_als(-x, lam, p, niter=10)
        xfit = -xfit
        return xfit
    elif flip_state =='no_flip':
        xfit = baseline_als(x, lam, p, niter=10)
        return xfit
                
def determine_flipping_direction_single(tracks, thresh=0.5):
    """ Determines based on the direction of the tracks whether the general movement is moving left or right and whether the coordinates need to be flipped before passing to :func:`baseline_als` for one timepoint only
    
    Parameters
    ----------
    tracks : numpy array
        (n_spixels, n_frames, 2) mean superpixel centroid coordinates over time
    thresh : float 
        the minimum fraction of superpixels whose sign points towards negative x-direction to warrant a 'flip'

    Returns 
    -------
    flip_state : str
        either 'flip' if mean velocities point towards negative x-direction or 'no_flip' for the converse
    """
    import numpy as np 
    diff = tracks[:,1:] - tracks[:,:-1]
    diff = np.mean(diff, axis=0) # derive the average movement vector. 
    
    x_direction_signs = np.sign(diff[:,1])
    
    if np.sum(x_direction_signs<0) / float(len(x_direction_signs)) > thresh:
        flip_state  = 'flip'
        return flip_state
    else:
        flip_state = 'no_flip'
        return flip_state


def determine_flipping_direction(tracks_r, tracks_g):
    """ Determines based on the direction of the tracks of red and green channels whether the wound is moving left or right and whether the coordinates need to be flipped before passing to :func:`baseline_als`.
    
    Parameters
    ----------
    tracks_r : numpy array
        (n_spixels x n_frames x 2) centroid superpixel positions of red channel
    tracks_g : numpy array
        (n_spixels x n_frames x 2) centroid superpixel positions of green channel

    Returns
    -------
    flip_state : numpy array
        array of size n_frames indicating whether to 'flip' or 'no_flip' in each frame to be used by :func:`asymmetric_fit`
    
    """
    import numpy as np 
    
    # impose a threshold on the movement in order to not bring this in the statistics. ?
    diff_r = tracks_r[:,1:] - tracks_r[:,:-1]
    diff_g = tracks_g[:,1:] - tracks_g[:,:-1]
    
    diff_r_mean = np.nanmean(diff_r, axis=0)
    diff_g_mean = np.nanmean(diff_g, axis=0)
    
    diff = (diff_r_mean + diff_g_mean) / 2.
    x_direction_signs = np.sign(diff[:,1])
    
#    return x_direction_signs
    flip_state = []
    
    for x in x_direction_signs:
        if x>0:
            flip_state.append('flip')
        else:
            flip_state.append('no_flip')
            
    flip_state = np.hstack(flip_state)
    
    return flip_state

        
# i'm not sure what the curve2stack is doing?
def curve2stack(curves, yline, shape):
    """ Given the x-coordinates of the line and its corresponding fixed y-coordinates yline and the image shape, map the line defined by the coordinates into a binary image

    Parameters
    ----------
    curves : numpy array
        an (nframes, n) where n is the number of sampled points to represent the line's x-coordinates
    yline : numpy array
        a (n,) 1-d array specifying the temporally consistent y-coordinates at which the wound boundary was sampled at 
    shape : array-like
        a (nrows, ncols) tuple specifying the original image dimensions

    Returns
    -------
    imgs : numpy array 
        (nframes, nrows, ncols) binary image with 1 indicating the line 

    """
    import numpy as np
    imgs = []

    for curve in curves:
        frame = np.zeros(shape)
        
        if np.sum(np.logical_not(np.isnan(curve))) > 0:
            xline = curve.copy()

            frame[yline.astype(np.int), np.clip(xline.astype(np.int), 0, shape[1]-1)] = 1
    
        imgs.append(frame[None,:])
    imgs = np.concatenate(imgs, axis=0)   

    return imgs  


def curve2img(curves, shape):
    """ Given the x-coordinates of the line and its corresponding fixed y-coordinates yline and the image shape, map the line defined by the coordinates into a binary image

    Parameters
    ----------
    curves : numpy array
        an (nframes, n) where n is the number of sampled points to represent the line's x-coordinates
    yline : numpy array
        a (n,) 1-d array specifying the temporally consistent y-coordinates at which the wound boundary was sampled at 
    shape : array-like
        a (nrows, ncols) tuple specifying the original image dimensions

    Returns
    -------
    imgs : numpy array 
        (nframes, nrows, ncols) binary image with 1 indicating the line 

    """
    import numpy as np
    imgs = []

    for ii, curve in enumerate(curves):
        # print ii
        frame = np.zeros(shape)
        
        if np.sum(np.logical_not(np.isnan(curve))) > 0:
            frame[curve[:,1].astype(np.int), np.clip(curve[:,0].astype(np.int), 0, shape[1]-1)] = 1
    
        imgs.append(frame[None,:])
        
    return np.concatenate(imgs, axis=0)      
    

def normalize_curves_2D(curves, shape):
    """ Given a curve (list of (x,y)) coordinates and the original image shape, returns the normalised set of coordinates where the x, y coordinate values have been divided by the number of columns and rows in the original image respectively
    
    Parameters
    ----------
    curves : numpy array 
        an (N,2) array of (x,y) coordinates. First column is the x-coordinate and second column is the y-coordinate. 
    shape : array-like
        a (nrows, ncols) tuple specifying the original image dimensions

    Returns
    -------
    curve_s : numpy array
        normalised curves of the same input shape with the value of each x and y scaled to the range [0-1].

    """
    m, n = shape
    curve_s = curves.copy()
    curve_s[:,1] = curves[:,1] / float(m) # normalise the y coordinates.
    curve_s[:,0] = curves[:,0] / float(n) # normalize the x coordinates.

    return curve_s


def uniquefy_curves(curves, axis=1):
    """ Given a (x,y) list of coordinates that describe (sample) a single curve, this function returns the reduced set of unique (x,y) points that describe the wound such each unique point along the given axis e.g. x corresponds to only one possible value y 
    
    Parameters
    ----------
    curves : numpy array 
        (N,2) array of (x,y) coordinates describing the wound
    axis : int
        the column to uniquefy values over, default = 1 corresponding to y-values in (x,y)

    Returns
    -------
    curve_s : numpy array
        reduced array of (x,y) coordinates where the appropriate axis x (axis=0) or y (axis=1) is unique.

    """
    uniq_x = np.unique(curves[:,axis])
    uniq_ids = [np.arange(len(curves))[curves[:,axis] == xx] for xx in uniq_x]
    curve_s = np.vstack([np.median(curves[ii], axis=0) for ii in uniq_ids])

    return curve_s


def fit_spline(x,y, k=3, s=None, ext=0):
    """ Fit a univariate spline function y = spl(x) to given x, y image coordinates. 

    Parameters
    ----------
    x : numpy array
        1-D array of x direction image coordinates 
    y : numpy array
        1-D array of y direction image coordinates
    k : int 
        order of the polynomical spline
    s : 
        Positive smoothing factor used to choose the number of knots. Number of knots will be increased until the smoothing condition is satisfied c.f. scipy.interpolate.UnivariateSpline function
    ext : int 
        the extrapolation mode of scipy.interpolate.UnivariateSpline function

    Returns
    -------
    spl : scipy UnivariateSpline object
        fitted scipy.interpolate.UnivariateSpline object. Can be used as a function given x values to get corresponding y values e.g. y=spl(x)
    """

    from scipy.interpolate import UnivariateSpline

    spl = UnivariateSpline(x, y, k=k, s=s, ext=ext)

    return spl
                        

def densify_boundary_line(curves, shape, axis=1):
    """ Given a set of points that lie on the boundary of the wound curve, this function uniquefies the coordinates to enable fitting of a cubic spline function which is then used to interpolate the curve at fixed y-coordinates for visualisation
    
    Parameters
    ----------
    curves : list of numpy array
        list of an array of (x,y) coordinates that sample the wound line between two epithelial sheets
    shape : array-like
        a tuple (nrow, ncol) specifying the size of the image shape
    axis : int (default:1)
        the axis to use for uniquefying the coordinates for spline fitting

    Returns
    -------
    blines : list of numpy array
        list of a corresponding array of (x,y) coordinates that have been reinterpolated after the cubic spline fitting

    """
    m, n = shape[:2]
    blines = [] 

    for c in curves:
        if len(c) > 0: 
            c =  c[np.argsort(c[:,axis])]
            c_uniq = uniquefy_curves(c, axis=axis)
            c_uniq = normalize_curves_2D(c_uniq, shape=shape) # by chucking this out might have become more unstable? 

            # fit the curves to densify the line... 
            spl = fit_spline(c_uniq[:,1], c_uniq[:,0], k=3, s=None)
            # spl = fit_polyfit(c_uniq[:,0], c_uniq[:,1], degree=2)

            y_max = np.max(c[:,1])
            y_min = np.min(c[:,1])
            bline_y = np.linspace(y_min, y_max, int(y_max-y_min)+1) / float(m)
            # bline_y = np.linspace(0, m, m+1) / float(m)
            bline_x = spl(bline_y) 
            # bline_y = c_uniq[:,1]
            # bline_x = eval_boundary_coords(spl, bline_y)[:,0] # evaluate the polynomial fit.

            bline = np.vstack([bline_x*n, bline_y*m]).T
            blines.append(bline)
        else:
            blines.append([])

    return blines


def boundary_roughness(curves_full):
    """ Computes the roughness of a boundary line for a given collective of lines (line that describes the interface between two sheets) defined as the ratio between the total length of the line and an equivalent straight line
    
    The equivalent straight line is derived from the first and last coordinate describing each line, thus each line is expected to be sorted in ascending / describe y-coordinate order (corresponding to image rows).

    Parameters
    ----------
    curves_full : list of numpy array
        list of (x,y) coordinate array specifying the coordinates of the curve. The number in the list equals the number of curves to evaluate e.g. the number of frames in the video
        
    Returns
    -------
    coefs : numpy array
        1-D numpy array of boundary roughness, one value per input curve

    """
    coefs = []

    for c in curves_full:
        if len(c) > 0: 
            l_0 = np.linalg.norm(c[0]-c[-1])
            l = np.diff(c, axis=0); 
            l = np.linalg.norm(l, axis=1)
            l = np.sum(l) # sum all the individual distances. 
            coef = l/float(l_0)
            coefs.append(coef)
        else:
            coefs.append(np.nan) # give a nan to aid in the later informatics. 

    coefs = np.hstack(coefs)
    return coefs


def boundary_superpixel_coords_RGB(rgb_video, meantracks_r, meantracks_g, movement_thresh=0.2, t_av_motion=3, robust=True, lenient=True, debug_visual=False, max_dist=1.5, y_bins=50, y_frac_thresh=0.60):
    """ Unbiasedly finds and tracks the moving interface between two colour epithelial sheets from the derived superpixel tracks of each colour

    Compared to :func:`boundary_superpixel_meantracks_RGB`, this function conducts the modified density biased asymmetric least squares algorithm which is better for (x,y) coordinates.

    Parameters
    ----------
    rgb_video : numpy array
        (n_frames, n_rows, n_cols, 3) RGB video 
    meantracks_r : numpy array
        (n_spixels, n_frames, 2) coordinates of superpixel centroids for red channel
    meantracks_g : numpy array
        (n_spixels, n_frames, 2) coordinates of superpixel centroids for green channel
    movement_thresh : float 
        the minimum displacement between 1 frame for superpixels to be considered 'moving'
    t_av_motion : int 
        the number of frames to average motion over 
    robust : bool
        if True utilise the density based asymmetric smoothing of :func:`baseline_als_density`
    lenient : bool
        if True utilises flipping in the asymmetric smoothing of :func:`baseline_als`. 
    debug_visual : bool
        if True plots intermediate results for debugging
    max_dist : float 
        upper distance to be considered a match between red and green points at the boundary
    y_bins : int
        the number of bins to divide the image rows into to assess the number of points sampled of the boundary
    y_frac_thresh : float (0-1) 
        the minimum number of bins that should contain points for the boundary to be estimated

    Returns
    -------
    boundary_curves : list of numpy array
        list of all superpixel track positions that model the boundary 
    als_curves : 
        list of all asymmetrically filtered superpixel track positions that model the boundary 

    """
    from tqdm import tqdm
    import pylab as plt
    import numpy as np
    from skimage.exposure import rescale_intensity
    from .tracks_statistics_tools import find_moving_spixels, find_const_tracks
    
    n_frames, m, n, _ = rgb_video.shape
    spixel_size = meantracks_r[1,0,1] - meantracks_r[1,0,0] # spixel size.
    
    """
    Find the moving spixels in the meantracks for each colour
    """
    select_r_track = find_const_tracks(meantracks_r)
    select_g_track = find_const_tracks(meantracks_g)
    
    meantracks_r_ = meantracks_r[np.logical_not(select_r_track)]
    meantracks_g_ = meantracks_g[np.logical_not(select_g_track)]
                                
    moving_r = find_moving_spixels(meantracks_r_, t_av=t_av_motion, thresh = movement_thresh)
    moving_g = find_moving_spixels(meantracks_g_, t_av=t_av_motion, thresh = movement_thresh)
    
    # print moving_r.shape

    # if robust==True or lenient==True:
    #     """
    #     flag switches on whether or not we should use asymmetric least squares method to fit the points.
    #     """
    #     # flip_state = determine_flipping_direction(np.vstack([meantracks_r, meantracks_g]))
    #     flip_state = determine_flipping_direction(meantracks_r[moving_r], meantracks_g[moving_g])
    
    """
    Frame Iteration to determine the boundary curves.
    """
    boundary_curves = []
    als_curves = []

    for frame in tqdm(range(n_frames-1)):
                    
        # which superpixels of each colour are moving.
        move_r_frame = moving_r[frame]
        move_g_frame = moving_g[frame]

        # fetch candidate points. 
        pos_r = meantracks_r_[move_r_frame,frame]
        pos_g = meantracks_g_[move_g_frame,frame]

        # print '---'
        # print move_r_frame.shape

        flip_state = determine_flipping_direction(meantracks_r[:,frame:frame+2], meantracks_g[:, frame:frame+2])
        # print flip_state

        # if the positions is empty then don't bother.
        if len(pos_r) > 0 and len(pos_g) > 0:
            
            # find the nearest green and red points that is 
            # how to deal with failures here? 
            coords_r, coords_g = match_red_green_segs(pos_r[:,[1,0]], pos_g[:,[1,0]], (m,n), max_dist=max_dist*spixel_size, nearest_N=1)
            # coords_r = meantracks_r_[:,frame][:,[1,0]]
            # coords_g = meantracks_g_[:,frame][:,[1,0]]

            # # screen out the coords that landed on the side... 
            # coords_r_select = np.logical_and(np.logical_and(coords_r[:,0] > spixel_size, coords_r[:,0] < n-spixel_size), 
            #                                  np.logical_and(coords_r[:,1] > spixel_size, coords_r[:,1] < m-spixel_size))
            # coords_g_select = np.logical_and(np.logical_and(coords_g[:,0] > spixel_size, coords_g[:,0] < n-spixel_size), 
            #                                  np.logical_and(coords_g[:,1] > spixel_size, coords_g[:,1] < m-spixel_size))
            # coords_r = coords_r[coords_r_select]
            # coords_g = coords_g[coords_g_select]

            if len(coords_r) > 0 and len(coords_g) > 0:
                
                all_pos = np.vstack([coords_r, coords_g])
                test_res = test_y_density(all_pos, m, bins=y_bins, frac_thresh=y_frac_thresh) # if doesn't cover at least this fraction of bins then the fitting will be ill-conditioned. 
                
                if test_res:

                    boundary_curves.append(all_pos) # save all pos out. 

                    # optionally emit the ALS filtered results.
                    # is there sufficient density ? 
                    # conduct median position filtering 
                    # outlier_select = outlier_detect_axis(all_pos, thresh=2) # this gives numerically better fitting and probably somewhat faster. 
                    # all_pos = all_pos[outlier_select]
                        
                    """
                    Filter the points using ALS algorithm before fitting the boundary curve. 
                    """
                    
                    if robust:
                        # print('robust')
                        # x_filt_ALS = asymmetric_fit(all_pos[:,0],all_pos[:,1], lam=1e4, p=0.1, flip_state=flip_state)    
                        x_filt_ALS = baseline_als_density(all_pos[:,1],all_pos[:,0], lam=1e1, p=0.5, radius=1*spixel_size, niter=10, N_cutoff=5, shape=1, slope=1)                                       
                        all_pos_filt = np.vstack([x_filt_ALS, all_pos[:,1]]).T
                    elif lenient:
                        # print('flipping')
                        # # set the opposite. 
                        # if flip_state =='flip':
                        #     flip_state = 'no_flip'
                        # elif flip_state == 'no_flip':
                        #     flip_state = 'flip'
                        flip = flip_state[0]
                        # print flip 
                        x_filt_ALS = asymmetric_fit(all_pos[:,0],all_pos[:,1], lam=1e4, p=0.01, flip_state=flip)                                            
                        all_pos_filt = np.vstack([x_filt_ALS, all_pos[:,1]]).T
                    else:
                        # do nothing more.
                        all_pos_filt = all_pos.copy()
                    
                        
                    """
                    Fitting of the final boundary curve :D.  (can we use a more smoothed spline? like a bspline that doesn't need to pass through all points?)
                    """
                    als_curves.append(all_pos_filt)

                    # spl_coords = fit_polyfit(all_pos_filt[:,0], all_pos_filt[:,1], degree=2) #use a simple quadratic model to fit. (a lower order might be more preferred.?)
                    # boundary_curves.append(spl_coords) # just save the model. 
                    
                    # """
                    # Optional Debug Visualization 
                    # """
                    # if debug_visual==True:
                    #     pos_heatmap = return_point_density(all_pos, (m,n), sigma=spixel_size/2)
                    #     yline = np.linspace(0, m, 100)
                    #     b_curve = eval_boundary_coords(spl_coords, yline) # evaluate the polynomial fit.
                        
                    #     fig, ax = plt.subplots()
                    #     ax.imshow(rgb_video[frame])
                    #     ax.imshow(pos_heatmap, cmap='coolwarm', alpha=0.3)
                    #     ax.plot(b_curve[:,0], b_curve[:,1], 'w--', lw=4.)
                    #     ax.set_xlim([0,n])
                    #     ax.set_ylim([m,0])
                    #     plt.show()
                    
                else:
                    # return no coefficients
                    boundary_curves.append([]) # just save the model. 
                    als_curves.append([])
            else:
                # return no coefficients. 
                boundary_curves.append([])
                als_curves.append([])
        else:
            boundary_curves.append([])
            als_curves.append([])            
    
    return boundary_curves, als_curves


def boundary_superpixel_meantracks_RGB(rgb_video, meantracks_r, meantracks_g, movement_thresh=0.2, t_av_motion=3, robust=False, lenient=True, debug_visual=False, max_dist=1.5, y_bins=50, y_frac_thresh=0.60):
    """ Unbiasedly finds and tracks the moving interface between two colour epithelial sheets from the derived superpixel tracks of each colour

    Parameters
    ----------
    rgb_video : numpy array
        (n_frames, n_rows, n_cols, 3) RGB video 
    meantracks_r : numpy array
        (n_spixels, n_frames, 2) coordinates of superpixel centroids for red channel
    meantracks_g : numpy array
        (n_spixels, n_frames, 2) coordinates of superpixel centroids for green channel
    movement_thresh : float 
        the minimum displacement between 1 frame for superpixels to be considered 'moving'
    t_av_motion : int 
        the number of frames to average motion over 
    robust : bool
        if True increases the asymmetric penalisation in :func:`baseline_als`
    lenient : bool
        if True utilises flipping in the asymmetric smoothing of :func:`baseline_als`. 
    debug_visual : bool
        if True plots intermediate results for debugging
    max_dist : float 
        upper distance to be considered a match between red and green points at the boundary
    y_bins : int
        the number of bins to divide the image rows into to assess the number of points sampled of the boundary
    y_frac_thresh : float (0-1) 
        the minimum number of bins that should contain points for the boundary to be estimated

    Returns
    -------
    boundary_curves : list 
        list of all spline (quadratic) fitted coefficients (functions) that model the boundary 
    curves_lines : list 
        list of all spline (quadratic) interpolated x-coordinates that model the boundary 
    curve_img : numpy array
        the (n_frames, n_rows, n_cols) corresponding binary video when the coordinates are remapped into an image.
    boundary_line :
        the (n_rows, n_cols) binary kymograph image illustrating the wound boundary evolution over time 

    """
    from tqdm import tqdm
    import pylab as plt
    import numpy as np
    from skimage.exposure import rescale_intensity
    from .tracks_statistics_tools import find_moving_spixels, find_const_tracks
    
    n_frames, m, n, _ = rgb_video.shape
    spixel_size = meantracks_r[1,0,1] - meantracks_r[1,0,0] # spixel size.

    """
    Find the moving spixels in the meantracks for each colour
    """
    select_r_track = find_const_tracks(meantracks_r)
    select_g_track = find_const_tracks(meantracks_g)
    
    meantracks_r_ = meantracks_r[np.logical_not(select_r_track)]
    meantracks_g_ = meantracks_g[np.logical_not(select_g_track)]
                                
    moving_r = find_moving_spixels(meantracks_r_, t_av=t_av_motion, thresh = movement_thresh)
    moving_g = find_moving_spixels(meantracks_g_, t_av=t_av_motion, thresh = movement_thresh)
    
    
    if robust==True or lenient==True:
        """
        flag switches on whether or not we should use asymmetric least squares method to fit the points.
        """
        flip_state = determine_flipping_direction_single(np.vstack([meantracks_r, meantracks_g]))
    
    """
    Frame Iteration to determine the boundary curves.
    """
    boundary_curves = []

    for frame in tqdm(range(n_frames)):
                    
        # which superpixels of each colour are moving.
        move_r_frame = moving_r[frame]
        move_g_frame = moving_g[frame]

        # fetch candidate points. 
        pos_r = meantracks_r_[move_r_frame,frame]
        pos_g = meantracks_g_[move_g_frame,frame]

        # if the positions is empty then don't bother.
        if len(pos_r) > 0 and len(pos_g) > 0:
            
            # find the nearest green and red points that is 
            coords_r, coords_g = match_red_green_segs(pos_r[:,[1,0]], pos_g[:,[1,0]], (m,n), max_dist=max_dist*spixel_size)
            
            if len(coords_r) > 0 and len(coords_g) > 0:
                
                all_pos = np.vstack([coords_r, coords_g])
                test_res = test_y_density(all_pos, m, bins=y_bins, frac_thresh=y_frac_thresh) # if doesn't cover at least this fraction of bins then the fitting will be ill-conditioned. 
                
                if test_res:
                    # conduct median position filtering 
                    outlier_select = outlier_detect_axis(all_pos, thresh=2) # this gives numerically better fitting and probably somewhat faster. 
                    all_pos = all_pos[outlier_select]
                        
                    """
                    Filter the points using ALS algorithm before fitting the boundary curve. 
                    """
                    
                    if robust:
                        # print('robust')
                        x_filt_ALS = asymmetric_fit(all_pos[:,0],all_pos[:,1], lam=1e4, p=0.1, flip_state=flip_state)                                            
                        all_pos_filt = np.vstack([x_filt_ALS, all_pos[:,1]]).T
                    elif lenient:
                        # print('flipping')
                        # set the opposite. 
                        if flip_state =='flip':
                            flip_state = 'no_flip'
                        elif flip_state == 'no_flip':
                            flip_state = 'flip'
                        x_filt_ALS = asymmetric_fit(all_pos[:,0],all_pos[:,1], lam=1e5, p=0.01, flip_state=flip_state)                                            
                        all_pos_filt = np.vstack([x_filt_ALS, all_pos[:,1]]).T
                    else:
                        # do nothing more.
                        all_pos_filt = all_pos.copy()
                    
                    """
                    Fitting of the final boundary curve :D.  (can we use a more smoothed spline? like a bspline that doesn't need to pass through all points?)
                    """
                    spl_coords = fit_polyfit(all_pos_filt[:,0], all_pos_filt[:,1], degree=2) #use a simple quadratic model to fit. (a lower order might be more preferred.?)
                    boundary_curves.append(spl_coords) # just save the model. 
                    
                    """
                    Optional Debug Visualization 
                    """
                    if debug_visual==True:
                        pos_heatmap = return_point_density(all_pos, (m,n), sigma=spixel_size/2)
                        yline = np.linspace(0, m, 100)
                        b_curve = eval_boundary_coords(spl_coords, yline) # evaluate the polynomial fit.
                        
                        fig, ax = plt.subplots()
                        ax.imshow(rgb_video[frame])
                        ax.imshow(pos_heatmap, cmap='coolwarm', alpha=0.3)
                        ax.plot(b_curve[:,0], b_curve[:,1], 'w--', lw=4.)
                        ax.set_xlim([0,n])
                        ax.set_ylim([m,0])
                        plt.show()
                    
                else:
                    # return no coefficients
                    boundary_curves.append([]) # just save the model. 
            else:
                # return no coefficients. 
                boundary_curves.append([])
        else:
            boundary_curves.append([])
            
    """
    Assemble the boundary curve.
    """
    if debug_visual==True:
        # Check if we find any curves by seeing if we fit the coefficient or not.
        n_curves = [len(c)>0 for c in boundary_curves] # and we es
    
        fig, ax = plt.subplots()
        ax.plot(n_curves)
        plt.show()
    
    # Step 1: raster the curves and make into an image. 
    yline = np.arange(m) # every single y value. (exploiting the integer value nature of y .)
    curves_lines = find_boundary_lines(boundary_curves, yline) # iterate through the boundary spline models and fit the entire y axis to get a value for all possible y's at every timepoint. 
    curve_img = curve2stack(curves_lines, yline, (m,n))
    kymograph_curve_img = curve_img.max(axis=1)
    
    boundary_line = fetch_skeletion_line(kymograph_curve_img, fun=np.median) # use the median which is more robust. 
    
#    if debug_visual==True:
    masked_curve_img = np.ma.masked_where(kymograph_curve_img == 0, kymograph_curve_img)

    if debug_visual == True:
        fig, ax = plt.subplots()
        ax.imshow(kymograph_curve_img)
        ax.set_aspect('auto')
        plt.show()
    
        fig, ax = plt.subplots()
        ax.imshow(rescale_intensity(rgb_video.max(axis=1)))
        ax.imshow(masked_curve_img, alpha=0.5, cmap='gray_r')
        ax.plot(boundary_line[:,0], boundary_line[:,1], 'w', lw=4)
        ax.set_aspect('auto')
        plt.show()
    
    return boundary_curves, curves_lines, curve_img, boundary_line
   
    

# =============================================================================
#   Post-processing and linking of wound boundary tools 
# =============================================================================
def temporal_filter_curves(curves, max_gaps=5):
    """ Given a temporal numpy array of equal length x-coordinates, find the frames with missing predictions and linearly impute the x-values based on the available data either side of the temporal gap as long as the temporal gap is smaller than that given by max_gaps

    In essence this function aims to ensure temporal consistency for frame by frame predicted wound coordinates (which treat frames independently)

    Parameters
    ----------
    curves : numpy array
        (n_frames, len_curve) array of (x-) coordinate values or (y-) coordinate values if the x coordinates were the ones fixed over time
    max_gaps : int 
        the maximum number of missing frames for linear imputation to be conducted over

    Returns
    -------
    curves_ : numpy array
        same array as input but with missing temporal frames < max_gaps imputed for a more continuous evolution

    """
    # first copy the curves object 
    curves_ = curves.copy()
    curves_[np.isnan(curves_)] = 0
    
    frames = []

    n_frames = len(curves)
    
    for i in range(n_frames):
        if np.sum(np.logical_not(np.isnan(curves[i]))) > 0:
            frames.append(i)
            
    frames = np.hstack(frames)
    
    """
    Find continuous segments:
    """
    stretches = [] 
    
    stretch = [frames[0]] # add this in already. 
    for i in range(len(frames)-1):
        time0 = frames[i]
        time1 = frames[i+1]

        if time1 - time0 == 1:
            stretch.append(time1)
            if i == len(frames)-2:
                stretches.append(stretch)
        else:
            stretches.append(stretch)
            stretch=[]

    stretches = [s for s in stretches if len(s)>0]

    if len(stretches) == 1:
        s = stretches[0]
        if s[0] == 0 and s[-1] == len(curves)-1:
#            print('nothing to do')
            return curves # nothing to be done.
        else:
            curves_[:s[0]] = curves[s[0]][None,:] # just copy this across.
            return curves_
    elif len(stretches) > 1:
        
#        print(stretches)
        # there is more than one segment and we iterate... 
        # 1. fill in before.
        for j in range(len(stretches)):
            if j == 0: 
                seg0 = stretches[j]
                curves_[:seg0[0]] = curves[seg0[0]][None,:]
            elif j >0 and j< len(stretches):
                seg0 = stretches[j-1]
                seg1 = stretches[j]
        
                curve0 = curves[seg0[-1]]
                curve1 = curves[seg1[0]]
#                print('===')
#                print(seg0[-1], seg1[0])
#                print('***')
                for kk in range(len(curve0)):
                    curves_[seg0[-1]+1:seg1[0],kk] = np.linspace(curve0[kk], curve1[kk], seg1[0]-1-seg0[-1])
                
        return curves_
    

# =============================================================================
# boiler scripts to describe the wound boundary to enable plotting for the kymograph.
# =============================================================================
def fetch_coordinates_sections(labelled):
    """ Given a labelled integer image such as that returned by doing connected components analysis by skimage.measure.label retrieve the coordinates belonging to each labelled region ignoring the label 0 which is assumed to refer to the background pixels.
    
    Parameters
    ----------
    labelled : numpy array
        (n_rows, n_cols) integer array where each unique integer i > 0 denotes a connected region one wants to retrieve all the image coordinates over. The integer 0 regions are treated as image background and their coordinates are not fetched
    
    Returns
    -------
    coords : list 
        list of (x,y) coordinate numpy arrays for each unique labelled region

    """
    nrows, ncols = labelled.shape
    X, Y = np.meshgrid(range(ncols), range(nrows))
    
    uniq_regions = np.unique(labelled)
    
    coords = []

    for i in uniq_regions[1:]:
        coords.append(np.vstack([X[labelled==i], Y[labelled==i]]).T)
        
    return coords
    
def check_intersect(ref, list_array):
    """ function to check whether a given list is present within a list array of other lists. Result is either 'yes' or 'add' meaning it should be added

    Parameters
    ----------
    ref : list 
        reference list that one wants to check if it is already in list_array
    list_array : list
        list of 1-D arrays or list

    Returns
    -------
    intersect : str
        'yes' for ref is already contained in one of the items in list_array or 'add' if it isn't and needs to be added
    """
    import numpy as np 
    intersect = []

    for item in list_array:
        if len(np.intersect1d(ref, item)) > 0:
            if len(np.intersect1d(ref, item)) == len(ref):
                intersect = 'yes' # how long is it ? 
#            elif len(ref) > len(np.intersect1d(ref, item)):
            else:
                intersect = 'add'
                
    if intersect == []:
        intersect = 'add'

    return intersect
     
def find_uniq_merge_groups(merge_groups):
    """ Given a list of items this function reduces this list into a minimal set of groups that captures their relationships. It is used for example when one has a list of pairwise relationships and wants to reduce this into a few groups that these elements belong to that would describe the relationships.

    Parameters
    ----------
    merge_groups : list 
        list of items to merge and combine together

    Returns 
    -------
    final_filt_list : list
        the filtered reduced list of items. Each item should contain mutually exclusive members

    """
    import numpy as np 
    
    merge_cand = list(merge_groups) # create a new copy

    if len(merge_cand) == 1:
        master_merge = list(merge_groups)
    else:
        master_merge = []

        # this only deals with the case when candidates are greater than 1. 
        while len(merge_cand) > 1:
             cand = merge_cand[0] # grab the first element.
             merge_cand.remove(cand)
             
             if len(merge_cand) == 0:
                 # check for intersection with any of the ones already in master merge. 
                  if len(master_merge) == 0: 
                      master_merge.append(cand)
                  else:
                      check = check_intersect(cand, master_merge)
                      if check == [] or check =='add':
                          master_merge.append(cand)
             else:
                  merged = []
                  for c in merge_cand:
                       intersect_cand = np.intersect1d(cand, c)

                       if len(intersect_cand) > 0:
                            cand = np.unique(np.array(cand+c).ravel())
                            cand = list(cand)
                            if cand not in merge_cand:
                                merge_cand.append(cand)
                            merged.append(c) # remove these.

                  if len(master_merge) == 0: 
                      master_merge.append(cand)
                  else:
                      check = check_intersect(cand, master_merge)

                      if check == [] or check =='add':
                          if cand not in master_merge:
                              master_merge.append(cand)
                              
        if len(merge_cand) == 1:
            master_merge.append(merge_cand[0])

    master_list = list(np.unique(master_merge))

    # final remove of conflicting bits... 
    final_filt_list = []

    for m in master_list:
        ref_list = list(master_list)
        ref_list.remove(m)
        overlap = check_intersect(m, ref_list)
        
        if overlap=='add':
            final_filt_list.append(m)
            
    final_filt_list = list(np.unique(final_filt_list))
    
    if type(final_filt_list[0]) is not list:
        final_filt_list = [final_filt_list]

    return final_filt_list
    
def match_labelled_regions(coord_list, dist_thresh=3):
    """ Given a coordinate list describing different image regions, merge the regions based on distance proximity given by a maximum distance threshold given by dist_thresh

    Parameters
    ----------
    coord_list : list of numpy array
        list of (x,y) coordinate array with each describing a region of the image to consider for merging
    dist_thresh : float 
        maximum distance to consider for region merging

    Returns
    -------
    cands : list
        list of region id groups after merging (not the combined coordinates)
    """
    # this aims to join together random matching elements -> actually there is a more robust method based on gradients! and prediction....    
    import itertools
    from sklearn.metrics.pairwise import pairwise_distances
    
    dist_matrix = np.zeros((len(coord_list), len(coord_list)))

    for i, j in itertools.combinations(range(len(coord_list)), 2):
        dist = pairwise_distances(coord_list[i], coord_list[j])
        min_dist = np.min(dist)
        dist_matrix[i,j] = min_dist
        dist_matrix[j,i] = min_dist

    dist_matrix += np.diag(1000*np.ones(len(coord_list)))
    
    # now iterate and find all pairs satisfying dist_thresh.
    cands = []
    
    for i in range(len(dist_matrix)):
        dists = dist_matrix[i]
        select = np.arange(len(dists))[dists<=dist_thresh]
        if len(select) > 0:
            cands.append([i, select[np.argmin(dists[select])]])
    
    # now find all cliques. 
    if len(cands) == 0:
        return []
    else:
        cands = find_uniq_merge_groups(cands)
#        print(cands)
        return cands
    
    
def fetch_skeletion_line(img, fun):
    """ Binarise the given line image and return the line describing the skeleton by running through every unique image row and applying fun to aggregate the x-coordinates. Used for returning the single wound line to overlay on a kymograph

    Parameters
    ----------
    img : numpy array 
        (n_rows, n_cols) binary image of a line
    fun : numpy function 
        aggregation function to merge coordinate values e.g. np.median, np.mean for median and mean respectively from the numpy library

    Returns
    -------
    coords : numpy array    
        the (x,y) coordinates of the skeleton line that pass through the 'centre' of the binary image

    """
    import numpy as np 
    binary = img > 0
    
    nrows, ncols = binary.shape
    X, Y = np.meshgrid(range(ncols), range(nrows))
    
    X_valid = X[binary]
    Y_valid = Y[binary]
    
    y = np.unique(Y_valid)
    x = []
    
    for y_ in y:
        x_ = X_valid[Y_valid==y_]
        x_mean = fun(x_)
        x.append(x_mean)
    x = np.hstack(x)    
    
    coords = np.vstack([x,y]).T

    return coords
    
def process_curve_img_multiple(curve_img, dilation=3, dist_thresh=5):
    """ Given a kymograph curve_img as a result of applying superpixel boundary tracking, :func:`boundary_superpixel_meantracks_RGB` automatically identify if there exist the unique boundary curves present returning them as a list of (x,y) coordinate list. 

    Parameters
    ----------
    curve_img : numpy array
        (nrows, ncols) numpy image of the boundary line between two populations with time as the image row and x- as the image columns as returned by :func:`boundary_superpixel_meantracks_RGB`
    dilation : int
        the size of the disk kernel used to dilate the binary curve_img to close small gaps between very close disconnected regions of the same line
    dist_thresh : float
        the maximum distance to consider two regions to be neighbours and merged

    Returns
    -------
    b_lines : list
        list of (x,y) coordinates describing each line found

    """
    # just fit the largest component. ? 
    from skimage.morphology import square, binary_closing, binary_dilation, diamond, disk
    from skimage.measure import label 
    
    binary = binary_dilation(curve_img, disk(dilation)) # do some dilation to link missing components. 
    labelled = label(binary)
    
    # now get the coordinates belonging to and find components to link. 
    labelled_coords = fetch_coordinates_sections(labelled)

#    print labelled_coords
    merge_regions = match_labelled_regions(labelled_coords, dist_thresh=dist_thresh)
    
    if len(merge_regions) > 0:
        # print('merging')
        # then we merge labelled regions. 
        labelled_new = np.zeros_like(labelled)
        regions_old = np.unique(labelled)[1:]

        for i in range(len(merge_regions)):
            m = merge_regions[i]
            for mm in m:
                labelled_new[labelled==regions_old[mm]] = i+1 
    else:
        labelled_new = labelled.copy()
    # fig, ax = plt.subplots()
    # ax.imshow(labelled_new)
    # ax.set_aspect('auto')
    # plt.show()
    b_lines = []

    for reg in np.unique(labelled_new)[1:]:
        b_line = fetch_skeletion_line(labelled_new==reg, np.median)
        b_lines.append(b_line)
    
    return b_lines
    
    
