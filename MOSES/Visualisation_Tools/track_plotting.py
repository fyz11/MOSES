#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 23:09:37 2017

@author: felix
@email: felixzhou1@gmail.com

This script and all the code within is licensed under the Ludwig License.

see README.md for details on usage.
"""

def plot_tracks(tracks, ax, color=None, lw=1., alpha=1):
    """ plots tracks onto a given matplotlib axis 

    Parameters
    ----------
    tracks : numpy array
        (n_superpixels x n_frames x 2) numpy array, giving the centroid positions of superpixels over time
    ax : matplotlib ax object
        a matplotlib axis instance such as that returned from fig, ax = plt.subplots()
    color : str
        colour of the plotted lines given either named or as (r,g,b) value.   
    lw : float
        linewidth, c.f. matplotlib
    alpha : float
        transparency of the plotted lines, c.f. matplotlib value between 0-1 with 0 being fully transparent and 1 being solid
        
    Returns
    -------
    None, void function
    
    """
    n_spixels = tracks.shape[0]

    for i in range(n_spixels):
        ax.plot(tracks[i,:,1], tracks[i,:,0], c = color, lw=lw, alpha=alpha)
        
    return []