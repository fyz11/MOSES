# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 22:04:57 2018

@author: felix

set of tools for describing the wound boundary of wound healing assays. 

"""
import numpy as np 
import pylab as plt 

# =============================================================================
#   Wound Finding Tools
# =============================================================================
# option 1: detect boundary edges by image segmentation. 
def detect_edges_rgb( img_r, img_g, downscale=32, smooth=2, sigma=3):
    
    """ 
    """
    from skimage.filters import gaussian 
    from skimage.feature import canny 
    from skimage.measure import label 
    from skimage.exposure import rescale_intensity, equalize_hist
    from skimage.filters.rank import median
    from skimage.morphology import square
    import numpy as np 
    
    img = np.dstack([img_r, img_g])
    img = maxpool_scale(img, scale=downscale)
    
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
    

def detect_edges_gray( img, downscale=32, smooth=2, sigma=3):
    
    """
    """
    from skimage.filters import gaussian 
    from skimage.feature import canny 
    from skimage.measure import label 
    import numpy as np 
    from .Utility_Functions.image import maxpool_scale
    
    img = np.dstack([img, img])
    img = maxpool_scale(img, scale=downscale)
    
    img = img[:,:,0] / 255. 
    img = gaussian(img, sigma=smooth) 
    edges = canny(img, sigma=sigma) 
    labelled = label(edges) # technically should rescale back... 
    
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
    """
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
    """
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
    """
    """
    import numpy as np 
    p_r, p_g = generate_pair_distances(coords_r, coords_g, inf_distance=1000)
    
#    select_r = np.min(p_r, axis=1) <= dist_thresh
#    select_g = np.min(p_g, axis=1) <= dist_thresh
    select_r = np.sum(p_r<=dist_thresh, axis=1) >= min_no
    select_g = np.sum(p_g<=dist_thresh, axis=1) >= min_no

    return coords_r[select_r], coords_g[select_g]
    
    
def sample_and_match_red_green_segs(red_mask, green_mask, n_samples=1000, min_dist=10, dist_thresh=2, nearest_N=2, debug=False):
    """
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
            
        return coords_r[unique_r], coords_g[unique_g]
    else:
        return [], []


def match_red_green_segs(coords_r, coords_g, shape, min_dist=0, max_dist=2, nearest_N=1):
    """
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
    """
    spl: spline model for the boundary coords. 
    """
    import numpy as np 
    
    xline = np.polyval(spl, yline)
    
    return np.vstack([xline, yline]).T
    
    
# not actually sure what this does? 
def find_stable_wound_frame(coefs_time, min_length=10):
    """
    """
    import numpy as np 
    select = np.arange(len(coefs_time))
    curve_valid = np.hstack(coefs_time) > 0

    all_good_times = select[curve_valid]

    stretches = [] 
    # group into consecutive regions. 
    stretch = [all_good_times[0]] # add this in already. 
    for i in range(len(all_good_times)-1):
        time0 = all_good_times[i]
        time1 = all_good_times[i+1]

        if (time1 - time0) == 1:
            stretch.append(time1)
            # last time whatever happens we append. 
            if i == len(all_good_times) - 2:
                stretches.append(stretch)
        else:
            stretches.append(stretch)
            stretch=[]

    # count the lengths and if satisfies min length criteria take the first frame.
    for s in stretches:
        len_stretch = len(s)
        if len_stretch >= min_length:
            return s[0]
        else:
            continue


def stack_curves(curves, shape=None):
    """
    """
    lines = []
    lens = np.hstack([len(c) for c in curves])
    select = lens[lens>0]
    if len(select) == 0: 
        # there is nothing.
        lines = np.ones((len(curves), 100, 2))
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

    return np.array(lines)


def find_boundary_lines(boundary_curves, yline):
    """
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
            
    return np.vstack(lines)

#==============================================================================
#   Cleaning up tools.
#==============================================================================
def outlier_detect_axis(X, axis=0, thresh=1.5):
    """
    """
#    from scipy.stats import iqr
    import numpy as np 
    x = X[:,axis] # which of the coordinates will we use?

    thresh_lower = np.median(x) - thresh*np.std(x)
    thresh_upper = np.median(x) + thresh*np.std(x)
    
    inlier_vals = np.logical_and(x>=thresh_lower, x<=thresh_upper) # standard gaussian type fitting.
    return inlier_vals
            
    
def outlier_detect_graph(points, thresh=10, n_components=1):
    """
    Outlier point detection based on 'communities'
    # how can this be further optimized? 
    """
    import networkx as nx 
    import numpy as np 
    from sklearn.metrics.pairwise import pairwise_distances
    
    adj_matrix = pairwise_distances(points)
    nx_graph = nx.from_numpy_matrix(adj_matrix<=thresh)
    
    # get the connected components and sort by the largest connected... 
    cnn = sorted(nx.connected_components(nx_graph), key = len, reverse=True) # largest first.

    return points[np.array(list(cnn[0])).ravel()]

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
    """
    """
    import numpy as np 
    
    p = np.polyfit(y,x, deg=degree)
    
    return p


def test_y_density(points, max_y, bins=10, frac_thresh=0.5):
   """
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
       
   return y_frac_ > frac_thresh
    
# do we have duplicates here of this function?
def return_point_density(points, shape, sigma=3):
    """
    """
    import numpy as np 
    from skimage.filters import gaussian
    
    blank = np.zeros(shape)
    blank[points[:,1], points[:,0]] += 1
    blank = gaussian(blank , sigma=sigma)
    
    return blank 
    

"""
Where best to use the baseline_als?
"""
def baseline_als(y, lam, p, niter=10):
    """
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

# distance based baseline fitting. 
def baseline_als_direction(y, lam, p, niter=10, direction_x='right'):
    """
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
    """
    """
    from scipy import sparse
    from scipy.sparse import linalg
    import numpy as np

    # # for each y compute the difference
    # if direction_x == 'right':
    #     dist = np.abs(lims[1] - y) # pull everything this way. 
    # if direction_x == 'left':
    #     dist = np.abs( y - lims[0]) # pull everything to the left.    
    
    # plt.figure()
    # plt.plot(y, 'o')

    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in xrange(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = linalg.spsolve(Z, w*y)
        # mid_z = np.mean(z)
        if direction_x == 'right':
            dists = np.abs(y-lims[0]) ** 2
            dists = dists/float(np.max(dists))
#                                print(dists)
#                                w = p * (z > mid_z) + (1-p) * (z < mid_z) # how much to weight this? or just use the median? 
            w = (p)*dists 
        if direction_x == 'left':
#                                mid_z = np.min(y)
            dists = np.abs(lims[1] - y) ** 2
            dists = dists/float(np.max(dists))
#                                print(dists)
#                                w = p * (z < mid_z) + (1-p) * (z>mid_z)
            w = (p) * dists
            
        # plt.plot(z, '.', alpha=0.5)
    return z

def generalised_sigmoid(x, cutoff, shape=1, slope=1):
    """
    """
    sigmoid = 1./(1+np.exp(-slope*(x-cutoff)))**(1./shape)
                
    return sigmoid

def baseline_als_density(x, y, lam, p, radius, niter=10, N_cutoff=5, shape=1, slope=1):
    """
    Instead of the above baseline which moves according the sign of the residual between y and z, we modify to bias towards density instead.
    suitable for use with (x,y)
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
    """
    """
    # determine whether to flip the estimation based on which lies more on either side. 
    if flip_state == 'flip':
#        print 'flipping: %d/%d' %(n_less, n_greater)
        xfit = baseline_als(-x, lam, p, niter=10)
        return -xfit
    elif flip_state =='no_flip':
        xfit = baseline_als(x, lam, p, niter=10)
        return xfit
    
            
# def determine_flipping_direction(tracks, thresh=0.5):
    
#     import numpy as np 
#     diff = tracks[:,1:] - tracks[:,:-1]
#     diff = np.mean(diff, axis=0) # derive the average movement vector. 
    
#     x_direction_signs = np.sign(diff[:,1])
    
#     if np.sum(x_direction_signs<0) / float(len(x_direction_signs)) > thresh:
#         return 'flip'
#     else:
#         return 'no_flip'
def determine_flipping_direction(tracks_r, tracks_g):
    """
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
    """
    """
    import numpy as np
    imgs = []

    for curve in curves:
        frame = np.zeros(shape)
        
        if np.sum(np.logical_not(np.isnan(curve))) > 0:
            xline = curve.copy()

            frame[yline.astype(np.int), np.clip(xline.astype(np.int), 0, shape[1]-1)] = 1
    
        imgs.append(frame[None,:])
        
    return np.concatenate(imgs, axis=0)      


def curve2img(curves, shape):
    """
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

    """
    shape = (m,n) of image. 
    """
    m, n = shape
    curve_s = curves.copy()
    curve_s[:,1] = curves[:,1] / float(m) # normalise the y coordinates.
    curve_s[:,0] = curves[:,0] / float(n) # normalize the x coordinates.

    return curve_s


def uniquefy_curves(curves, axis=1):

    """
    make curves manifold, 1-1 (x,y) mapping (not duply sampled....), where there is duplicates we use mode-seeking behaviour... ?
    """
    uniq_x = np.unique(curves[:,axis])
    uniq_ids = [np.arange(len(curves))[curves[:,axis] == xx] for xx in uniq_x]
    # keep_ids = np.hstack([i for i in uniq_ids if len(i)==1])
    # mod_ids = [i for i in uniq_ids if len(i)>1]
    # curve_s = curves[keep_ids]
    # if len(mod_ids) > 0:
    #     curve_mod = np.vstack([np.median(curves[ii], axis=0) for ii in mod_ids])
    #     curve_s = np.vstack([curve_mod, curve_s])
    #     curve_s = curve_s[np.argsort(curve_s[:,axis])]

    curve_s = np.vstack([np.median(curves[ii], axis=0) for ii in uniq_ids])
    return curve_s


def fit_spline(x,y, k=3, s=None, ext=0):

    from scipy.interpolate import UnivariateSpline

    spl = UnivariateSpline(x, y, k=k, s=s, ext=ext)

    return spl
                        

def densify_boundary_line(curves, shape, axis=1):

    """
    curves is a list or numpy array. 
    """
    m, n = shape[:2]
    blines = [] # 

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

    """
    Computes a boundary smoothness term. using l/l_0 
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


def boundary_superpixel_coords_RGB(rgb_video, meantracks_r, meantracks_g, movement_thresh=0.2, t_av_motion=3, robust=False, lenient=True, debug_visual=False, max_dist=1.5, y_bins=50, y_frac_thresh=0.60):
    
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

        # flip_state = determine_flipping_direction(meantracks_r[:,frame:frame+2], meantracks_g[:, frame:frame+2])
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
        flip_state = determine_flipping_direction(np.vstack([meantracks_r, meantracks_g]))
    
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
    
    """
    Find the maximum number of frame gaps and impute in the gap given a set of curves. 
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
    
    nrows, ncols = labelled.shape
    X, Y = np.meshgrid(range(ncols), range(nrows))
    
    uniq_regions = np.unique(labelled)
    
    coords = []

    for i in uniq_regions[1:]:
#        plt.figure()
#        plt.imshow(labelled==i)
#        plt.show()
        coords.append(np.vstack([X[labelled==i], Y[labelled==i]]).T)
        
    return coords
    
def check_intersect(ref, list_array):

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
    
# this function produces an equation for each boundary curve to plot on kymograph.
# =============================================================================
#   To do: utilise gradient info? and continuity... ideas.    
# =============================================================================
def process_curve_img_multiple(curve_img, dilation=3, dist_thresh=5):
    
    # just fit the largest component. ? 
    from skimage.morphology import square, binary_closing, binary_dilation, diamond, disk
    from skimage.measure import label 
    
    binary = binary_dilation(curve_img, disk(dilation)) # do some dilation to link missing components. 
    labelled = label(binary)
    
    # now get the coordinates belonging to and find components to link. 
    labelled_coords = fetch_coordinates_sections(labelled)
    
#    plt.figure()
#    plt.plot([len(c) for c in labelled_coords])
#    plt.show()
    
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
        
#    fig, ax = plt.subplots()
#    ax.imshow(labelled)
#    for i in range(len(labelled_coords)):
#        mean_coords = np.mean(labelled_coords[i], axis=0)
#        ax.text(mean_coords[0], mean_coords[1], str(i))
#    ax.set_aspect('auto')
#    plt.show()
    # having gotten we go ahead to derive the lines ...
            
    fig, ax = plt.subplots()
    ax.imshow(labelled_new)
    ax.set_aspect('auto')
    plt.show()
    
    b_lines = []

    for reg in np.unique(labelled_new)[1:]:
        b_line = fetch_skeletion_line(labelled_new==reg, np.median)
        b_lines.append(b_line)
    
    return b_lines
    
    
