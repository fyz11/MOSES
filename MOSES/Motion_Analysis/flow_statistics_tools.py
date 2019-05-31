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
    """ Computes the curl vector field given as a 2D or 3D vector data.
    
    Parameters
    ----------
    im : numpy array
        (nrows, ncols, 2) for 2d image or (n_z, nrows, ncols, 3) for 3d volumetric image, last channel is the (x,y) or (x,y,z) vectors.
    args : 
        optional arguments passed to np.gradient

    Returns
    -------
    curl : numpy array
        (nrows, ncols, 3) array representing the curl in i,j,k. For 2d image, the first two channels are 0.
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
    
    """Computes the divergence of a vector field given as a 2D or 3D vector data.
    
    Parameters
    ----------
    im : numpy array
        (nrows, ncols, 2) for 2d image or (n_z, nrows, ncols, 3) for 3d volumetric image, last channel is the (x,y) or (x,y,z) vectors.
    args : 
        optional arguments passed to np.gradient

    Returns
    -------
    div : numpy array
        (nrows, ncols) array representing the divergence
    
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

    
