#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 01:40:16 2017

@author: felix
@email: felixzhou1@gmail.com

This script and all the code within is licensed under the Ludwig License.

see README.md for details on usage.

"""

import numpy as np 

def curl_vector_flow(im, *args):
    """
    Input: im [nrows, ncols, 2] for image. or [n_z,zargrd nrows, ncols, 3] # for volumetric
    
    channels are in the order x,y,z (geometric)
    
    Curl is computed as \grad x F where F is the vector field.
    | i      j    k |
    | d/dx d/dy d/dz|
    | Fx    Fy   Fz |
    
    # do we also take into account time ? t can be time.
    """
    dim = im.shape[-1] # number of dimensions of vector.

    if dim == 2:
        Fx = im[...,0]  
        Fy = im[...,1]
        dFx_dy, dFx_dx = np.gradient(Fx, *args) 
        dFy_dy, dFy_dx = np.gradient(Fy, *args) 
        
        curl = np.dstack([np.zeros(Fx.shape), np.zeros(Fy.shape), dFy_dx - dFx_dy])
               
    if dim == 3:
        # get the flow in image convention order? 
        Fx = im[..., 0]
        Fy = im[..., 1]
        Fz = im[..., 2]

        dFx_dz, dFx_dy, dFx_dx = np.gradient(Fx, *args)
        dFy_dz, dFy_dy, dFy_dx = np.gradient(Fy, *args)
        dFz_dz, dFz_dy, dFz_dx = np.gradient(Fz, *args)
        
        # hard-code the equations.
        curl_x = dFz_dy - dFy_dz
        curl_y = -(dFz_dx - dFx_dz)
        curl_z = dFy_dx - dFx_dy
        
        curl = np.stack([curl_x, curl_y, curl_z])

    return curl
    
    
def div_vector_flow(im, *args):
    
    """
    Computes the divergence of the vector field f, corresponding to dFx/dx + dFy/dy + ...
    
    input im: channels are (x,y) or (x,y,z)
    
    image dimensions are z,y,x
    """
    dim = im.shape[-1] # number of dimensions of vector.
    
    if dim == 2:
        Fx = im[...,0]  
        Fy = im[...,1]
        dFx_dy, dFx_dx = np.gradient(Fx, *args) 
        dFy_dy, dFy_dx = np.gradient(Fy, *args) 
        
        div = dFx_dx + dFy_dy

    if dim == 3:
        Fx = im[..., 0]
        Fy = im[..., 1]
        Fz = im[..., 2]

        dFx_dz, dFx_dy, dFx_dx = np.gradient(Fx, *args)
        dFy_dz, dFy_dy, dFy_dx = np.gradient(Fy, *args)
        dFz_dz, dFz_dy, dFz_dx = np.gradient(Fz, *args)
        
        # hard-code the equations.        
        div = dFx_dx + dFy_dy + dFz_dz
        
    return div

    
