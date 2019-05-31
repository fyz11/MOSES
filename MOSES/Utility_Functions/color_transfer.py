#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 16:57:30 2018

@author: felix
"""

import numpy as np 

def match_color(source_img, target_img, mode='pca', eps=1e-8, source_mask=None, target_mask=None):
    """ Matches the colour distribution of a target image to that of a source image using a linear transform based on matrix decomposition. 
    
    This effectively matches the mean and convariance of the distributions. 
    
    Images are expected to be of form (w,h,c) and can be either float in [0,1] or uint8 [0,255]
    
    Optionally image masks can be used to selectively bias the color transformation. 

    Parameters
    ----------
    source_img : numpy array
        (n_rows, n_cols, 3), either float in range [0,1] or uint8 and in range [0,255]
    target_img : numpy array
        (n_rows, n_cols, 3), either float in range [0,1] or uint8 and in range [0,255]
    mode : str (default='pca')
        one of 'chol', 'pca' or 'sym' for computing the matrix inverse. The effect is slightly different in each case.
    eps : float (default=1e-8)
        small tolerance for numerical stability.
    source_mask : numpy bool array (default=None)
        (n_rows, n_cols), optional binary mask to mask out particular colours.
    target_mask : numpy bool array (default=None)
        (n_rows, n_cols), optional binary mask to mask out particular colours.

    Returns
    -------
    matched_img : numpy array
        (n_rows, n_cols, 3), transformed target_img colour matched to source_img 

    """
    if target_img.max() > 1.0000000001:
        # ensure range in [0,1.], converts to float.  
        source_img = (source_img/255.).astype(np.float32)
        target_img = (target_img/255.).astype(np.float32)
    else:
        # ensure range in [0,255.]
        source_img = source_img.astype(np.float32); 
        target_img = target_img.astype(np.float32); 
    
  
    # 1. Compute the eigenvectors of the source color distribution (possibly masked)
    if source_mask is not None:
        mu_s = np.hstack([np.mean(source_img[:,:,0][source_mask==1]), np.mean(source_img[:,:,1][source_mask==1]), np.mean(source_img[:,:,2][source_mask==1])])
    else:   
        mu_s = np.hstack([np.mean(source_img[:,:,0]), np.mean(source_img[:,:,1]), np.mean(source_img[:,:,2])])
    s = source_img - mu_s # demean
    s = s.transpose(2,0,1).reshape(3,-1) # convert to (r,g,b), 3 x n_pixels
    
    if source_mask is not None:
        s = s[:, source_mask.ravel()==1] 
    Cs = s.dot(s.T) / s.shape[1] + eps * np.eye(s.shape[0]) # 3x3 covariance matrix. 
    
    # 2. Computes the eigenvectors of the target color distribution (possibly masked)    
    if target_mask is not None:
        mu_t = np.hstack([np.mean(target_img[:,:,0][target_mask==1]), np.mean(target_img[:,:,1][target_mask==1]), np.mean(target_img[:,:,2][target_mask==1])])
    else:   
        mu_t = np.hstack([np.mean(target_img[:,:,0]), np.mean(target_img[:,:,1]), np.mean(target_img[:,:,2])])
    t = target_img - mu_t
    t = t.transpose(2,0,1).reshape(3,-1)   
    
    if target_mask is not None:
        temp = t.copy()
        t = t[:, target_mask.ravel()==1] 
    Ct = t.dot(t.T) / t.shape[1] + eps * np.eye(t.shape[0]) # 3x3 covariance matrix.  

    """
    Color match the mean and covariance of the source image. 
    """    
    if mode == 'chol':
        chol_t = np.linalg.cholesky(Ct)
        chol_s = np.linalg.cholesky(Cs)
        ts = chol_s.dot(np.linalg.inv(chol_t)).dot(t)
    if mode == 'pca':
        eva_t, eve_t = np.linalg.eigh(Ct)
        Qt = eve_t.dot(np.sqrt(np.diag(eva_t))).dot(eve_t.T)
        eva_s, eve_s = np.linalg.eigh(Cs)
        Qs = eve_s.dot(np.sqrt(np.diag(eva_s))).dot(eve_s.T)
        ts = Qs.dot(np.linalg.inv(Qt)).dot(t)
    if mode == 'sym':
        eva_t, eve_t = np.linalg.eigh(Ct)
        Qt = eve_t.dot(np.sqrt(np.diag(eva_t))).dot(eve_t.T)
        Qt_Cs_Qt = Qt.dot(Cs).dot(Qt)
        eva_QtCsQt, eve_QtCsQt = np.linalg.eigh(Qt_Cs_Qt)
        QtCsQt = eve_QtCsQt.dot(np.sqrt(np.diag(eva_QtCsQt))).dot(eve_QtCsQt.T)
        ts = np.linalg.inv(Qt).dot(QtCsQt).dot(np.linalg.inv(Qt)).dot(t)
        
        
    # recover the image shape. 
    if target_mask is not None:
        matched_img_flatten = np.zeros_like(temp)
        matched_img_flatten[:,target_mask.ravel()==1] = ts.copy()
    else:
        matched_img_flatten = ts.copy()
        
    matched_img = matched_img_flatten.reshape(*target_img.transpose(2,0,1).shape).transpose(1,2,0)
    matched_img += mu_s
    
    if target_mask is not None:
        rgb_mask = np.dstack([target_mask, target_mask, target_mask])
        matched_img[rgb_mask==0] = target_img[rgb_mask==0]
    
    # clip limits. 
    matched_img[matched_img>1] = 1
    matched_img[matched_img<0] = 0
    
    return matched_img
