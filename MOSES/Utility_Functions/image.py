# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 23:01:49 2018

@author: felix
"""

import numpy as np 

# # check the two are identical.
# def maxpool_scale(img, scale=32):
    
#     from keras.layers import MaxPooling2D
#     from keras.models import Sequential
    
#     im = img[None,:] # pad... 
    
#     mod = Sequential()
#     mod.add(MaxPooling2D(pool_size=(scale, scale), input_shape=img.shape))
#     out = mod.predict(im)
    
#     return out[0]

def pool_numpy(array, factor, func=np.max):
    """ Downscale the given array by the given factor by the method specified by func.

    N.B the array dimensions needs to be fully divisible by the specified factor along each array dimension

    Parameters
    ----------
    array : numpy array 
        any sized numpy array
    factor : numpy array
        a (n,) sized list or array specifying the integer size to pool each of the n array dimensions by, the array dimension along the corresponding axis must be able to be fully divided by the specified factor e.g. for an (m,n) array which we wish to downsample by 2 the factor=(2,2). For a (m,n,3) array we can use factor=(2,2,1) to downscale only the spatial dimensions by a factor of 2
    func : python/numpy function
        the function to use for pooling e.g. np.mean for taking the mean, np.median for taking the median 

    Returns
    -------
    reduced_array : numpy array
        the output downscaled array with each dimension shrunk by the given factor amount along each respective array dimension

    """
    from skimage.measure import block_reduce
    reduced_array = block_reduce(array, (factor,factor), func)
    return reduced_array


def ridge_filter_hessian(img, sigma=3):
    """ Apply Hessian filtering to enhance vessel-like structures

    Parameters
    ----------
    img : numpy array
        (n_rows, n_cols) grayscale input image
    sigma : int 
        width of the Gaussian used for smoothing gradients

    Returns
    -------
    i2 : numpy array
        image same size image showing enhanced ridge like structures

    """
    from skimage.feature import hessian_matrix, hessian_matrix_eigvals
    from skimage.filters import threshold_otsu
    from skimage.morphology import skeletonize

    hxx, hxy, hyy = hessian_matrix(img, sigma=sigma)
    i1, i2 = hessian_matrix_eigvals(hxx, hxy, hyy)
    
    i2 = i2 <= threshold_otsu(i2)
    i2 = skeletonize(i2)
    
    return i2

def min_max_scaling(img):
    """ Min-max scaling of image intensities, same as skimage.rescale_intensity()

    Min and Max of image are computed globally irrespective of the number of channels.

    Parameters
    ----------
    img : numpy array
        input image, any dimensionality

    Returns
    -------
    img_ : numpy array
        image intensity rescaled image

    """
    import numpy as np 

    img_ = (img - np.min(img)) / float(np.max(img) - np.min(img))

    return img_
    

def compute_centroid_binary(binary):
    """ Compute the centroid coordinates of a binary mask

    Can also use scipy.ndimage.measurements.center_of_mass

    Parameters
    ----------
    binary : numpy bool array
        2d (n_rows, n_cols) boolean image with the image region we wish to demarcate marked as True and background is False
    
    Returns
    -------
    centroid : numpy array
        (x,y) coordinate of the positively marked area

    """
    import numpy as np 
    m, n = binary.shape
    X, Y = np.meshgrid(range(n), range(m))
    
    centroid = [np.mean(X[binary>0]), np.mean(Y[binary>0])]
    centroid = np.hstack(centroid)
    
    return centroid


def DoG(img, sigmas):
    """ computes the different of two gaussians with widths specified by sigma. It is used to detect blob-like images and identify areas with relatively constant intensity levels in the image.
    
    Parameters
    ----------
    img : numpy array
        grayscale (n_rows, n_cols) image
    sigmas : list or array
        the widths of the two gaussian filter in the difference computation. Specified in the form, [sigma_1, sigma_2]
    
    Returns
    -------
    dog : numpy array
        grayscale image same size as input
    """
    from skimage.filters import gaussian

    gaussian_1 = gaussian(img, sigmas[0])
    gaussian_2 = gaussian(img, sigmas[1])

    dog = gaussian_1 - gaussian_2

    return dog

def entropy_img(img, size):
    """ Computes the Shannon Entropy of the Image.

    The Shannon entropy highlights regions of the image with most texture. Local Regions of the filter size whose pixel intensity distribution is very different from its local surroundings will score highly.
    
    Parameters
    ----------
    img : numpy array
        a float or integer valued (n_rows, n_cols) image

    Returns
    -------
    entropy_img : int
        the approximate size of the image region structures one wishes to highlight

    """
    from skimage.filters.rank import entropy
    from skimage.morphology import square

    entropy_img = entropy(img, square(size))
    
    return entropy_img

def BackgroundRemoval(im):
    """ Removes smooth second order image artifacts such as spherical aberration by global polynomial fitting of image intensities

    Parameters
    ----------
    im : numpy array
        (n_rows, n_cols) grayscale image. Intensity values should be floating with values [0-1]

    Returns
    -------
    imgInit : numpy array
        background corrected image of the same size
    imgBg : numpy array
        estimated background image of the same size

    """
    import numpy as np
    from skimage.exposure import rescale_intensity

    nrows, ncols = im.shape    
    N = nrows*ncols
    
    xx, yy = np.meshgrid(np.arange(0,ncols),np.arange(0,nrows))
    xx = xx.ravel()
    yy = yy.ravel()
    
    X =np.column_stack( (np.ones(N), xx, yy, np.square(xx), np.multiply(xx,yy), np.square(yy)) ) 
    p = np.linalg.lstsq(X,im.ravel()) #X\im(:)
    p = p[0] 

    imgBg = np.dot(X,p)
    temp = im.ravel() - imgBg
    imgInit = temp.reshape(nrows,ncols)

    # min-max intensity rescaling of the image.
    if np.min(imgInit) == np.max(imgInit):
        imgInit = 0.0 
    else:
        imgInit = rescale_intensity(imgInit, out_range=(0,1))

    # imgInit is the subtracted background (flattened image), normalised to between [0,1]
    return imgInit, imgBg


def perona_malik(img, iterations=10, delta=0.14, kappa=15):
    """ Anisotropic filtering of image intensities using Perona-Malik Diffusion Method

    Parameters
    ----------
    img : numpy array
        input floating grayscale (n_rows, n_cols) image with intensity values [0-1]
    iterations : int
        Number of iterations to run the diffusion process. Higher values give greater smoothing in the output.
    delta : float
        This is the time step :math:`\Delta t` in the diffusion equation. 
    kappa : float
        This regulates the sensitivity to edges in the Perona-Malik formulation.

    Returns 
    -------
    G : numpy array
        Anisotropically filtered image the same size as img

    References
    ----------
    .. [1] Perona, P et. al, "Anisotropic diffusion." Geometry-driven diffusion in computer vision. Springer, Dordrecht, 1994. 73-92.

    """
    from scipy import misc, ndimage
    import numpy as np 
    # center pixel distances
    dx = 1
    dy = 1
    dd = np.sqrt(2)
    u = img.copy()
    
    # 2D finite difference to estimate gradients
    windows = [
        np.array(
                [[0, 1, 0], [0, -1, 0], [0, 0, 0]], np.float64
        ),
        np.array(
                [[0, 0, 0], [0, -1, 0], [0, 1, 0]], np.float64
        ),
        np.array(
                [[0, 0, 0], [0, -1, 1], [0, 0, 0]], np.float64
        ),
        np.array(
                [[0, 0, 0], [1, -1, 0], [0, 0, 0]], np.float64
        ),
        np.array(
                [[0, 0, 1], [0, -1, 0], [0, 0, 0]], np.float64
        ),
        np.array(
                [[0, 0, 0], [0, -1, 0], [0, 0, 1]], np.float64
        ),
        np.array(
                [[0, 0, 0], [0, -1, 0], [1, 0, 0]], np.float64
        ),
        np.array(
                [[1, 0, 0], [0, -1, 0], [0, 0, 0]], np.float64
        ),
    ]
    
    for r in range(iterations):
        # approximate gradients
        nabla = [ ndimage.filters.convolve(u, w) for w in windows ]
    
        # approximate diffusion function
        diff = [ 1./(1 + (n/kappa)**2) for n in nabla]
    
        # update image
        terms = [diff[i]*nabla[i] for i in range(4)]
        terms += [(1/(dd**2))*diff[i]*nabla[i] for i in range(4, 8)]
        u = u + delta*(sum(terms))
    
    # Kernel for Gradient in x-direction
    Kx = np.array(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.int32
    )
    # Kernel for Gradient in y-direction
    Ky = np.array(
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.int32
    )
    # Apply kernels to the image
    Ix = ndimage.filters.convolve(u, Kx)
    Iy = ndimage.filters.convolve(u, Ky)
    
    # return norm of (Ix, Iy)
    G = np.hypot(Ix, Iy)
    
    return G


def construct_boundary_masks(boundary_spl, shape, eps=5):
    """ Takes the boundary_spl function and applies it to all integer y-coordinates to compute the corresponding x-coordinates of the boundary. Then returns two binary images thats marks all pixels left (y, x<x-eps) and all pixels right (y, x>x + eps) of the boundary line where eps is a tolerance in number of pixels.
    
    Parameters
    ----------
    boundary_spl : function
        a function that takes y coordinates and gives corresponding x coordinates such that all (y,x) cartesian coordinates subsample a boundary line between two sheets.
    shape : 2-tuple
        grayscale image shape i.e. (n_rows, n_cols) as returned by the .shape method of a numpy array
    eps : int
        the number of pixels tolerance to call all pixels left and right of the boundary line 

    Returns
    -------
    left_mask : numpy array 
        0 or 1 valued array marking all pixels left of boundary line
    right_mask : numpy array
        0 or 1 values array marking all pixels right of boundary line
    
    """
    left_mask = np.zeros(shape, dtype=np.bool)
    right_mask = np.zeros(shape, dtype=np.bool)

    # use the y coordinate to make the points 
    yline = np.arange(shape[0])
    xline = boundary_spl(yline)

    for ii, x in enumerate(xline):
        left_mask[ii,:int(x)-eps] = 1
        right_mask[ii, int(x)+eps:] = 1 # give some leeway either side of the line!.

    return left_mask, right_mask


def detect_peaks(img, min_distance=1):
    """ Detect local maxima peaks in the image with minimum separation specified by the min_distance parameter
    
    Parameters
    ----------
    img : numpy array
        grayscale (n_rows, n_cols) image
    min_distance : float
        minimum separation distance between any two called peaks
    
    Returns
    -------
    coords : numpy array
        array of (y,x) coordinates specifying the pixel location in the image with local maxima

    """
    from skimage.feature import peak_local_max

    coords = peak_local_max(img, min_distance=min_distance)
    
    return coords


def quant_boundary_cells(count_img, boundary_spl, min_distance=5, eps=5):
    """ Given a density image of cell counts, detect all cells as the local maxima of the image. Then count the number of cells left and right of the boundary line through image masking
    
    Parameters
    ----------
    count_img : numpy array
    boundary_spl : function
        a function that takes y coordinates and gives corresponding x coordinates such that all (y,x) cartesian coordinates subsample a boundary line between two sheets.  
    min_distance : float
        minimum separation distance between any two called peaks
    eps : int
        the number of pixels tolerance to call all pixels left and right of the boundary line 

    Returns 
    -------
    [left_count, peaks_left] : [int, numpy array]
        count of the number of peaks and its corresponding (y,x) coordinates to the left of the boundary line
    [right_count, peaks_right] : [int, numpy array] 
        count of the number of peaks and its corresponding (y,x) coordinates to the right of the boundary line
    
    """
    import pylab as plt 
    # we form masks and count both sides of the boundary given a functional approximation of the boundary.
    cell_peaks = detect_peaks(count_img, min_distance=min_distance)
    left_mask, right_mask = construct_boundary_masks(boundary_spl, count_img.shape, eps=eps)

    # do counts!
    select_left = left_mask[cell_peaks[:,0].astype(np.int), cell_peaks[:,1].astype(np.int)]
    select_right = right_mask[cell_peaks[:,0].astype(np.int), cell_peaks[:,1].astype(np.int)]
    
    # count this ?
    peaks_left = cell_peaks[select_left]
    peaks_right = cell_peaks[select_right]

    left_count = len(peaks_left)
    right_count = len(peaks_right)

    return [left_count, right_count], [peaks_left, peaks_right]

def find_largest_component(binary, thresh=0.05, apply_thresh=False):
    """ Given a binary segmentation, perform connected component analysis and return either all regions larger than a minimum size or just the area with the largest area that is not background.
    
    Parameters
    ----------
    binary : numpy boolean array
        (n_rows, n_cols) binary segmentation image
    thresh : float 
        minimum area of a component to be kept. Area is a specified as a fraction of the image size
    apply_thresh : bool
        if True, all areas with area greater than the minimum size is kept else only the region of largest area is kept

    Returns
    -------
    
    """
    from skimage.measure import label, regionprops
    from scipy.ndimage.morphology import binary_fill_holes
    import numpy as np 
    
    nrows, ncols = binary.shape
    # connected components analysis
    labelled = label(binary)
    regs = regionprops(labelled)
    # 0 = background 
    uniq_regs = np.unique(labelled)[1:]
    areas = []

    for re in regs:
        areas.append(re.area)
        
    if len(areas) > 0: 
        areas = np.hstack(areas)
        largest_area = areas[np.argmax(areas)]
            
        if apply_thresh:
            if largest_area >= thresh*nrows*ncols:
                largest = labelled == uniq_regs[np.argmax(areas)]
                return binary_fill_holes(largest)
            else:
                return spixel_map
        else:
            largest = labelled == uniq_regs[np.argmax(areas)]
            return binary_fill_holes(largest)
    else:
        return np.zeros(binary.shape, dtype=np.bool)


def train_sheet_segmentation_model( im, sigmas=[1,11], size=5, close_size=5, debug=False, method='GaussianBayes', *args, **kwargs):
    """ Uses image thresholding on the entropy images to get the ground-truth segmentation of epithelial sheet on a well-strained frame. Then learns a machine learning model of how to classify pixels in subsequent frames. The particular machine learning model learnt is specified by the user.

    Four features are computed to classify each pixel: red pixel intensity, red entropy image intensity, green pixel intensity and green entropy image intensity. The entropy image is computed on the difference of gaussian image of the respective channel image intensity.

    Parameters
    ----------
    im : numpy array
        a RGB image (n_rows, n_cols, 3) with red and green epithelial sheet images
    sigmas : list or array-like
        the two spatial scales to use to compute difference of gaussian filter responses.
    size : int
        size of kernel used for computation of entropy image of difference of gaussian image
    close_size : int
        size of kernel used for closing holes after image thresholding
    debug : bool
        if True plots intermediate steps for checking
    method : str
        One of the following strings to denote which classifier method in scikit-learn to use.

            * 'GaussianBayes' : uses Gaussian based Naive Bayes classification
            * 'SVM' : trains a support vector machine 
            * 'Kneighbours' : this uses K nearest neighbour classification
    *args : 
        optional arguments to pass to the machine learning classifier function in scikit-learn
    **kwargs : 
        optional keyword arguments to pass to the machine learning classifier function in scikit-learn

    Returns
    -------
    clf : scikit-learn classifier object
        a scikit-learn classifier object which computes the images to give red/green epithelial sheet segmentation of the image.

    """
    from skimage.filters import threshold_otsu
    from scipy.ndimage.morphology import binary_fill_holes
    from skimage.morphology import binary_closing, disk, binary_dilation
    from sklearn import svm
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from skimage.exposure import rescale_intensity
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    import pylab as plt
    
    """
    Generate the label 
    """
    r_img = im[:,:,0]
    g_img = im[:,:,1]

    r_ = DoG(r_img, sigmas=sigmas)
    g_ = DoG(g_img, sigmas=sigmas)
    
    r_ent = entropy_img(r_, size=size)
    g_ent = entropy_img(g_, size=size)
    
    y_r = r_img >= threshold_otsu(r_img)
    y_g = g_img >= threshold_otsu(g_img)
    
    y_r = binary_fill_holes(y_r)
    y_g = binary_fill_holes(y_g)
    
    y_r = binary_closing(y_r, disk(close_size))
    y_g = binary_closing(y_g, disk(close_size))
    y_r = find_largest_component(y_r)
    y_g = find_largest_component(y_g)
    
    y_r = binary_dilation(y_r, disk(close_size))
    y_g = binary_dilation(y_g, disk(close_size))
    y_r = binary_fill_holes(y_r)
    y_g = binary_fill_holes(y_g)
    
    y = np.zeros(y_r.shape)
    y[y_r] = 1
    y[y_g] = 2

    if debug:
        plt.figure()
        plt.imshow(y)
        plt.show()

    """
    Compute some textural information properties to augment. 
    """
    X = np.dstack([r_img, r_, r_ent, g_img, g_, g_ent])
#    X = np.dstack([r_img, g_img])
    nrows, ncols, _ = X.shape
    x = X.reshape((nrows*ncols, -1)) # make this from RGB into multichannel. 
    
#    sc_model = StandardScaler()
#    sc_model.fit(x)
#    x = sc_model.transform(x)
    y = y.ravel()

    if method=='GaussianBayes':
        clf = GaussianNB(*args, **kwargs)
    if method=='SVM':
        clf = svm.SVC(*args, **kwargs) # recommended kernel='linear'
    if method=='Kneighbours':
        clf = KNeighborsClassifier(*args, **kwargs) # e.g. n_neigbors=5
    clf.fit(x,y)
    
    return clf
    
    
def apply_SVM(im, clf, sigmas=[1,11], size=5, multiply=[1,1]):
    """ Convenience function to use the trained classifier returned by the MOSES.Utility_Functions.image.train_sheet_segmentation_model function given a new image.

    Parameters
    ----------
    im : numpy array
        an RGB (n_rows, n_cols, 3) image
    clf : scikit-learn classifier object
        This classifier object is trained and the output of :func:`train_sheet_segmentation_model`
    sigmas : list or array-like
        the two spatial scales used to compute difference of gaussian filter responses.
    size : int
        size of kernel used for computation of the entropy image of the difference of gaussian image
    multiply : list or array-like
        optional numerical weights to weight the contribution of green or red cell features

    Returns
    -------
    fdskfasjk : 

    """
    from skimage.exposure import rescale_intensity
    import numpy as np 
    nrows, ncols, _ = im.shape
    
    r_ = DoG(im[:,:,0], sigmas=sigmas)
    g_ = DoG(im[:,:,1], sigmas=sigmas)
    
    r_ent = entropy_img(r_, size=size)
    g_ent = entropy_img(g_, size=size)
    
    X = np.dstack([im[:,:,0]*multiply[0], r_*multiply[0], r_ent*multiply[0], im[:,:,1]*multiply[1], g_*multiply[1], g_ent*multiply[1]])
#    X = np.dstack([im[:,:,0]*multiply[0], im[:,:,1]*multiply[1]])
    x = X.reshape((nrows*ncols, -1))
#    x = sc_model.transform(x)
    
    y_out = clf.predict(x)
    
    return y_out.reshape((nrows, ncols))





