# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 23:01:49 2018

@author: felix
"""

import numpy as np 

def maxpool_scale(img, scale=32):
    
    from keras.layers import MaxPooling2D
    from keras.models import Sequential
    
    im = img[None,:] # pad... 
    
    mod = Sequential()
    mod.add(MaxPooling2D(pool_size=(scale, scale), input_shape=img.shape))
    out = mod.predict(im)
    
    return out[0]


def pool_numpy(array, factor=2, func=np.max):
    
    from skimage.measure import block_reduce
    
    return block_reduce(array, (factor,factor), func)


def ridge_filter_hessian(img, sigma=3):
    
    from skimage.feature import hessian_matrix, hessian_matrix_eigvals
    from skimage.filters import threshold_otsu
    from skimage.morphology import skeletonize
    #assume you have an image img
    
    hxx, hxy, hyy = hessian_matrix(img, sigma=sigma)
    i1, i2 = hessian_matrix_eigvals(hxx, hxy, hyy)
    
    i2 = i2 <= threshold_otsu(i2)
    i2 = skeletonize(i2)
    
    return i2

def min_max_scaling(img):
    import numpy as np 
    return (img - np.min(img)) / float(np.max(img) - np.min(img))
    

def compute_centroid_binary(binary):
    
    import numpy as np 
    
    m, n = binary.shape
    X, Y = np.meshgrid(range(n), range(m))
    
    centroid = [np.mean(X[binary>0]), np.mean(Y[binary>0])]
    centroid = np.hstack(centroid)
    
    return centroid


def DoG(img, sigmas):
    
    from skimage.filters import gaussian
    
    return gaussian(img, sigmas[0]) - gaussian(img, sigmas[1])

def entropy_img(img, size):
    from skimage.filters.rank import entropy
    from skimage.morphology import square
    
    return entropy(img, square(size))

def BackgroundRemoval(im):
    # given an image, utilises the pixels to subtract the background in the image 
    # remembering images are represented by a matrix with x axis = cols, y axis = rows,
    # The algorithm carries out the following steps:
    # 1) normalize the input image
    # 2) fits a 2nd order background model 
    # 3) subtracts this from the input image, then makes everything positive to be a proper image, 
    # 4) renormalises the image, 

    import numpy as np
    from skimage.exposure import rescale_intensity

    nrows,ncols = im.shape    
    N = nrows*ncols
    
    xx, yy = np.meshgrid(np.arange(0,ncols),np.arange(0,nrows))
    xx = xx.ravel()
    yy = yy.ravel()
    
    X =np.column_stack( (np.ones(N), xx, yy, np.square(xx), np.multiply(xx,yy), np.square(yy)) ) 
    p = np.linalg.lstsq(X,im.ravel()) #X\im(:)
    
    p = p[0] 

    temp = im.ravel() - np.dot(X,p)
    imgInit = temp.reshape(nrows,ncols)

    # visualize background. 
    # plt.
    # plt.imshow( np.reshape(temp, im.shape), cmap='gray')
    # plt.show()
    imgInit = rescale_intensity(imgInit, out_range=(0,1))
#    if np.min(imgInit) == np.max(imgInit):
#        imgInit = 0.0
#    else:
#        imgInit = (imgInit - np.min(imgInit))/(np.max(imgInit) - np.min(imgInit))
  
    # imgInit is the subtracted background (flattened image), normalised to between [0,1]
    return imgInit


def perona_malik(img, iterations=10, delta=0.14, kappa=15):
    
    from scipy import misc, ndimage
    import numpy as np 
    # center pixel distances
    dx = 1
    dy = 1
    dd = np.sqrt(2)
    
    u = img.copy()
    
    # 2D finite difference felix1990
    felix1990
    
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


def construct_boundary_masks(boundary_spl, shape, axis=1, eps=5):

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
    
    from skimage.feature import peak_local_max
    
    coords = peak_local_max(img, min_distance=min_distance)
    
    return coords


def quant_boundary_cells(count_img, boundary_spl, axis=1, min_distance=5, eps=5):

    import pylab as plt 
    # we form masks and count both sides of the boundary given a functional approximation of the spline.
    cell_peaks = detect_peaks(count_img, min_distance=min_distance)
    left_mask, right_mask = construct_boundary_masks(boundary_spl, count_img.shape, axis=axis, eps=eps)

    # do counts!
    select_left = left_mask[cell_peaks[:,0].astype(np.int), cell_peaks[:,1].astype(np.int)]
    select_right = right_mask[cell_peaks[:,0].astype(np.int), cell_peaks[:,1].astype(np.int)]
    
    # count this ?
    peaks_left = cell_peaks[select_left]
    peaks_right = cell_peaks[select_right]

    left_count = len(peaks_left)
    right_count = len(peaks_right)

    return [left_count, right_count], [peaks_left, peaks_right]



def find_largest_component(spixel_map, thresh=0.05, apply_thresh=False):
        
    from skimage.measure import label, regionprops
    from scipy.ndimage.morphology import binary_fill_holes
    import numpy as np 
    
    nrows, ncols = spixel_map.shape
    
    labelled = label(spixel_map)
    regs = regionprops(labelled)
    
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
        return np.zeros(spixel_map.shape, dtype=np.bool)


def train_sheet_segmentation_model( im, sigmas=[1,11], size=5, close_size=5, debug=False):
    
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

#    clf = svm.SVC(kernel='linear')
    clf = GaussianNB()
#    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(x,y)
    
    return clf
    
    
def apply_SVM(im, clf, sigmas=[1,11], size=5, multiply=[1,1,1]):

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





