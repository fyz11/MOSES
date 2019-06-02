#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 15:11:36 2018

@author: felix

track statistics tools

"""

import numpy as np 

def find_const_tracks(tracks):
    """ Identifies all superpixels that do no move from its initial position for the whole duration of the video.

    Parameters
    ----------
    tracks : numpy array
        (n_superpixels x n_frames x 2) meantracks of superpixels

    Results
    -------
    select: numpy bool array
        (n_superpixels,) binary indicator of moving (False) or non-moving (True). 

    """
    select = []

    for track in tracks:

        test_y = track[:,0] == track[0,0]
        test_x = track[:,1] == track[0,1]
 
        if np.sum(np.logical_and(test_x, test_y)) == len(test_x):
            select.append(True)
        else:
            select.append(False)
    
    select = np.hstack(select).astype(np.bool)
    return select
    

def rolling_stats(fun, a ,w):
    """ Computes a moving average using a specified maths function given by fun applied to series of 1d signals

    Parameters
    ----------
    fun :  python function
        <function> any maths function e.g. np.mean, np.median which one wants to apply on a rolling basis
    a : numpy array
        (n_signals x signal_length), 2d array of n signals of length signal_length
    w : int
        window size should be odd. 

    Returns
    -------
    r : numpy array
        (n_signals x signal_length), 2d array of n signals of length signal_length after rolling statistics filtered

    """
    import numpy as np 
    a_pad = np.pad(a, w//2, mode='reflect')
    r = np.empty(a.shape)
    
    for i in range(len(a)):
        r[i] = fun(a_pad[i:i+w])
    return r

    
def find_moving_spixels(meantracks, t_av=3, thresh = 0.1):
    """ Finds which superpixels are moving based on the average distance they have moved over the imaged duration.

    Parameters
    ----------
    meantracks : numpy array
        (n_superpixels x n_frames x 2), track coordinates of superpixels in (y,x) convention 
    t_av : int
        the window to average statistics over. This is to smooth the velocity of single frames which may be noisy.
    thresh : float
        this is the minimum pixel distance they must move to be called 'moving'

    Returns
    -------
    moving : numpy bool array
        (n_superpixels,) binary vector of which superpixels are moving. 

    """
    import numpy as np
    
    diff = meantracks[:,1:] - meantracks[:,:-1]
    diff_pad = np.pad(diff,[[0,0],[t_av//2, t_av//2],[0,0]], mode='reflect')
    diff = np.concatenate([np.mean(diff_pad[:,i:i+t_av], axis=1)[None,:] for i in range(meantracks.shape[1])], axis=0)
    diff = np.sqrt(diff[:,:,0]**2 + diff[:,:,1]**2)
    
    moving = diff >= thresh
    return moving 
    
    
def extract_spixel_patch_seq(vid, tracks, size, pad_mode='constant'):
    
    """ Extract the image patches described by the superpixel track

    Parameters
    ----------
    vid : numpy array
        (n_frames, n_rows, n_cols) or (n_frames, n_rows, n_cols, 3), video, single channel or multiple channel.
    tracks : numpy array
        (n_superpixels, n_frames, 2), mean coordinates of superpixel over time in (y,x) coordinates.
    size : int
        patch_size around the centroid point.

    Outputs
    -------
    img_seqs : numpy array
        (n_superpixels x n_frames x size x size x dim) array containing img patches of superpiels over all time. 
    
    """
    # pad the images 
    padsize = size//2
    
    if len(vid.shape) == 3:
        newvid = np.pad(vid, [(0,0), (padsize, padsize), (padsize, padsize)], mode=pad_mode)
    elif len(vid.shape) == 4:        
        newvid = np.pad(vid, [(0,0), (padsize, padsize), (padsize, padsize), (0,0)], mode=pad_mode)
    
    tracks_ = tracks + padsize
    n_superpixels, n_frames, _ = tracks_.shape
    
    img_seqs = []
    
    for i in range(n_superpixels):
        img_seq = []

        for j in range(n_frames):
            # box bounds.
            xmin = tracks_[i,j,1] - padsize
            xmax = xmin + size
            ymin = tracks_[i,j,0] - padsize
            ymax = ymin + size
            
            # crop the image patch.
            img_seq.append(newvid[j,ymin:ymax,xmin:xmax][None,:])
        img_seq = np.concatenate(img_seq, axis=0)
        img_seqs.append(img_seq[None,:])
        
    img_seqs = np.concatenate(img_seqs, axis=0)
    return img_seqs

    
def corr2d_spixel_vid_seq(spixelseq):
    """ computes the 2d image correlation between successive patches along a superpixel track

    Parameters
    ----------
    spixelseq : numpy array
        number of image patch sequences. (assumes all image patches are successive temporally)

    Returns
    -------
    corrs : numpy array
        (n_frames,) 2d correlation values for each frame

    """
    from skimage.feature import match_template
    
    corrs = []
    n_imgs = len(spixelseq)
    
    for i in range(n_imgs-1):
        im1 = spixelseq[i]
        im2 = spixelseq[i+1]

        corr = match_template(im1, im2) # do we pad the input? -> don't think we need?
        corrs.append(np.max(corr))
        
    corrs = np.hstack(corrs)
    return corrs

def corr2d_all_spixel_vid_seq(spixel_seqs):
    """ computes the 2d image correlation between successive patches along a superpixel track for each superpixel track

    Parameters
    ----------
    spixel_seqs : numpy array
        n_spixels x n_frames number of image patch sequences. (assumes all image patches are successive temporally per superpixel)

    Returns
    -------
    corr_patches_superpixels : numpy array
        (n_spixels x n_frames) 2d correlation values for each frame for each superpixel track.

    """
    corr_patches_superpixels = []
    for i in range(spixel_seqs.shape[0]):
        # output is a temporal 1d sequence.
        corr_patches_superpixels.append(corr2d_spixel_vid_seq(spixel_seqs[i]))
        
    corr_patches_superpixels = np.vstack(corr_patches_superpixels)
    
    return corr_patches_superpixels
    

def RMSD_track(meantracks, ref_time=0):
    """ computes the root mean squared displacement curve (RMSD) given a reference time.
    :math:`RSMD=\sum{|x(t)-x(t=t_0)|}` where :math:`t_0` is the reference time usually 0.

    Parameters
    ----------
    meantracks : numpy array
        (n_spixels x n_frames x 2) centroid superpixel positions
    ref_time : int 
        reference time given as the frame number to compute displacements relative to.

    Returns
    -------
    rmsd : numpy array
        (n_spixels x n_frames) root mean squared displacement.

    """
    disps = meantracks[:,1:] - meantracks[:,ref_time][:,None,:]
    disps_mag = disps[:,:,0]**2 + disps[:,:,1]**2
    disps_mag = np.mean(disps_mag, axis=0)
    rmsd = np.sqrt(disps_mag)
    
    return rmsd


def RMSD_track_axis(meantracks, ref_time=0, axis=1):
    """ computes the root mean squared displacement curve (RMSD) given a reference time.
    :math:`RSMD=\sum{|x_{axis}(t)-x_{axis}(t=t_0)|}` where :math:`t_0` is the reference time usually 0.

    Parameters
    ----------
    meantracks : numpy array
        (n_spixels x n_frames x 2) centroid superpixel positions
    ref_time : int 
        reference time given as the frame number to compute displacements relative to.
    axis : int 
        0 or 1 depending on y- or x- axis velocity

    Returns
    -------
    rmsd : numpy array
        (n_spixels x n_frames) root mean squared displacement (along the x- or y- axis)
        
    """
    disps = meantracks[:,1:] - meantracks[:,ref_time][:,None,:]
    disps_mag = disps[:,:,axis]**2
    disps_mag = np.mean(disps_mag, axis=0)
    rmsd = np.sqrt(disps_mag)

    return rmsd
    
def average_displacement_tracks(meantracks):
    """ computes the mean displacement vector over time of a set of tracks.
    
    Parameters
    ----------
    meantracks : numpy array
        (n_spixels x n_frames x 2) centroid superpixel positions

    Returns
    -------
    disp : numpy array
        (n_spixels x 2) mean (over time) displacement vector of each superpixel
        
    """
    disp = meantracks[:,1:] - meantracks[:,:-1]
    disp = np.mean(disp, axis=1)
     
    return disp
#==============================================================================
# Visualisation for the corr sequence ? 
#==============================================================================
#f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)
#ax1.bar(.5*(hist_bins_r[1:] + hist_bins_r[:-1]), hist_corrs_r, width=hist_bins_r[1]-hist_bins_r[0], color='r', alpha=0.4)               
#ax2.bar(.5*(hist_bins_g[1:] + hist_bins_g[:-1]), hist_corrs_g, width=hist_bins_g[1]-hist_bins_g[0], color='g', alpha=0.4)               
#plt.xlim([0,1])
#plt.ylim([0,0.5])
#ax1.set_ylabel('% of Superpixels')
#ax2.set_ylabel('% of Superpixels')
#plt.xlabel('Normalised Cross-Correlation')
##        f.savefig('hist_correlation-'+base_name, dpi=300, bbox_inches='tight', pad_inches=0)
#plt.show()
#
#fig, ax = plt.subplots()
#plt.imshow(video[0], alpha=0.5)
#plt.scatter(tracks_r[:,0,1], tracks_r[:,0,0], c=meancorrs_, cmap='coolwarm', vmin=0, vmax=1)
#plt.grid('off')
#plt.axis('off')
#plt.colorbar()
#plt.show()
