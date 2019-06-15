#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 16:48:04 2017

@author: felix
@email: felixzhou1@gmail.com

This script and all the code within is licensed under the Ludwig License.

see README.md for details on usage.
"""

def view_ang_flow(motion_field, ang_only=False):
    """ Create the angular flow from the computed optical flow or motion field and visualise it using HSV colour scheme.
    
    Parameters
    ----------
    motion_field : numpy array
        n_rows x n_cols x 2 numpy array where the two dimensions are assumed to be (x,y) giving the computed vector displacements
    ang_only : bool
        if True, the intensity of the colours will be fixed to the maximum brightness and not be correlated with the speed of motion

    Returns
    -------
    None, it is a void function.
    
    """
    import cv2
    import numpy as np 

    nrows, ncols, _ = motion_field.shape
    hsv = np.zeros((nrows, ncols,3)).astype(np.uint8)
    hsv[:,:,1] = 255
    hsv[:,:,2] = 255

    mag, ang = cv2.cartToPolar(motion_field[:,:,0], motion_field[:,:,1])
    hsv[:,:,0] = ang*180/np.pi/2
    if ang_only == False:
        hsv[:,:,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
    
    return rgb