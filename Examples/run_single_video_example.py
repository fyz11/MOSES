#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 23:13:14 2017

@author: felix
@email: felixzhou1@gmail.com


This is an example file showing how to use the library to derive measurements for a single video.

"""

if __name__=="__main__":
    
    #==============================================================================
    #   Defining some default params, (these are recommended params)  
    #==============================================================================
    
    # optical flow tracking parameters (doesn't really matter a lot)
    optical_flow_params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    
    # number of superpixels
    n_spixels = 1000
    
    from scipy.misc import imsave
    import pylab as plt 
    import scipy.io as spio
    import numpy as np
    from Optical_Flow_Tracking.superpixel_track import compute_grayscale_vid_superpixel_tracks
    from Utility_Functions.file_io import read_multiimg_PIL
    from Visualisation_Tools.track_plotting import plot_tracks
    from Track_Filtering.filter_meantracks_superpixels import filter_red_green_tracks
    from Motion_Analysis.mesh_statistics_tools import construct_MOSES_mesh, from_neighbor_list_to_graph, compute_MOSES_mesh_strain_curve, construct_mesh_strain_vector, compute_motion_saliency_map, compute_boundary_formation_index, compute_MOSES_mesh_stability_index, compute_max_vccf_cells_before_after_gap, compute_spatial_correlation_function, compute_mesh_principal_strain_angle_ellipse, compute_mesh_order
    from Motion_Analysis.wound_close_sweepline_area_segmentation import wound_sweep_area_segmentation
    from Visualisation_Tools.mesh_visualisation import visualise_mesh, visualise_mesh_strain_ellipses
    from skimage.filters import gaussian

    """
    1. Specify input file 
    """
    infile = 'c_EPC2(G)_CP-A(R)_KSFM+RPMI_5_Fast GFP.tif_R-G.tif'
    

    """
    2. Read the file in as a vidstack.
    """
    print 'reading .tif video ...'
    vidstack = read_multiimg_PIL(infile)
    n_frames, n_rows, n_cols, _ = vidstack.shape
    
    print('Size of video: (%d,%d,%d)' %(n_frames,n_rows,n_cols))
    
    """
    3. Superpixel tracks with optical flow algorithm, compute 1 for each relevant color channel.
    
    *SLOW STEP*, wait a few minutes.
    
    """
    print 'extracting superpixel tracks from video, can be slow...'
    optflow_r, meantracks_r = compute_grayscale_vid_superpixel_tracks(vidstack[:,:,:,0], optical_flow_params, n_spixels)
    optflow_g, meantracks_g = compute_grayscale_vid_superpixel_tracks(vidstack[:,:,:,1], optical_flow_params, n_spixels)
    
    # save the output:
    savetracksmat = ('meantracks_'+infile).replace('.tif', '.mat')
    spio.savemat(savetracksmat, {'meantracks_r':meantracks_r,
                                 'meantracks_g':meantracks_g})

    # used to plot matplotlib figures with no boundaries.
    width = float(n_cols)
    height = float(n_rows)
    
    # visualize the tracks along with first frame, you might notice 'contamination' of tracks, we can clean with up with post track filtering
    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(vidstack[0])
    plot_tracks(meantracks_r, ax, color='r', lw=.2)
    plot_tracks(meantracks_g, ax, color='g', lw=.2)
    ax.set_xlim([0, n_cols])
    ax.set_ylim([n_rows, 0])
    ax.grid('off')
    ax.axis('off')
    fig.savefig(('tracksimg-no-filt_'+infile).replace('.tif', '.png'), dpi=height)
    plt.show()
    
    
    """
    4. Post Filtering of Superpixel Tracks.
    """
    print 'post filtering tracks for epithelial sheet migration videos, can be slow...'
    meantracks_r, meantracks_g = filter_red_green_tracks(meantracks_r, meantracks_g, img_shape=(n_rows, n_cols), frame2=1)
    
    
    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(vidstack[0])
    plot_tracks(meantracks_r, ax, color='r', lw=.2)
    plot_tracks(meantracks_g, ax, color='g', lw=.2)
    ax.set_xlim([0, n_cols])
    ax.set_ylim([n_rows, 0])
    ax.grid('off')
    ax.axis('off')
    fig.savefig(('tracksimg-filt_'+infile).replace('.tif', '.png'), dpi=height)
    plt.show()
    
#     save the improved tracks, now we only use these for all downstream analysis e.g. to compute statistics etc.
    savetracksmat = ('meantracks-filt_'+infile).replace('.tif', '.mat')
    spio.savemat(savetracksmat, {'meantracks_r':meantracks_r,
                                 'meantracks_g':meantracks_g})
    
    # example of loading up the saved .mat object 
    meantracks_r = spio.loadmat(('meantracks-filt_'+infile).replace('.tif', '.mat'))['meantracks_r']
    meantracks_g = spio.loadmat(('meantracks-filt_'+infile).replace('.tif', '.mat'))['meantracks_g']
    
    # get the average superpixel size. (this is effectively the resolution of the tracking and used again and again to set automatic thresholds.)
    spixel_size = meantracks_r[1,0,1] - meantracks_r[1,0,0]


    """
    5. MOSES mesh computation, we compute a mesh for each, and visualise the state of the mesh at frame 20 as example. 
    """
    print 'Computing MOSES Mesh ...'
    MOSES_mesh_strain_time_r, MOSES_mesh_neighborlist_r = construct_MOSES_mesh(meantracks_r, dist_thresh=1.2, spixel_size=spixel_size)
    MOSES_mesh_strain_time_g, MOSES_mesh_neighborlist_g = construct_MOSES_mesh(meantracks_g, dist_thresh=1.2, spixel_size=spixel_size)
    

    # visualise the two meshes independently by exploiting networkx for a chosen frame .e.g. 20, first turn the neighborlist into a networkx graph object.
    mesh_frame20_networkx_G_red = from_neighbor_list_to_graph(meantracks_r, MOSES_mesh_neighborlist_r, 20)
    mesh_frame20_networkx_G_green = from_neighbor_list_to_graph(meantracks_g, MOSES_mesh_neighborlist_g, 20)
    
    
    width = float(n_cols)
    height = float(n_rows)
    
    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(vidstack[20], alpha=0.7)
    visualise_mesh(mesh_frame20_networkx_G_red, meantracks_r[:,20,[1,0]], ax, node_size=.2, node_color='r')
    ax.set_ylim([n_rows,0])
    ax.set_xlim([0,n_cols])
    fig.savefig('mesh_frame20_red.png', dpi=height)
    
    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(vidstack[20], alpha=0.7)
    visualise_mesh(mesh_frame20_networkx_G_green, meantracks_g[:,20,[1,0]], ax, node_size=.2, node_color='g')
    ax.set_ylim([n_rows,0])
    ax.set_xlim([0,n_cols])
    fig.savefig('mesh_frame20_green.png', dpi=height)
    
    # we can plot the mesh strain in terms of the local strain ellipse based on the normalised direction of displacement/stretching at the same timepoint.
    eigenvalues_time, eigenangles_time, stretch_ratio_time, select_time = compute_mesh_principal_strain_angle_ellipse(meantracks_r, MOSES_mesh_neighborlist_r, point_thresh=8)
    
    # red ellipses show highly distorted regions (regions of high point density)
    fig, ax = plt.subplots()
    ax.imshow(vidstack[20], alpha=0.5)
    visualise_mesh_strain_ellipses(mesh_frame20_networkx_G_red, meantracks_r[:,20,[1,0]], 
                                   eigenvalues_time[20], eigenangles_time[20], spixel_size, ax, mask =select_time[20],
                                   node_size=10, width=.3, linewidths=.5, node_color='k')
    plt.show()
    
    
    """
    6. Normalised mesh strain, computes the mesh strain curve for each colour and combines them to produce the normalised mesh strain for the video which is used as a motion signature.
    """
    print 'Computing normalised MOSES Mesh strain curve...'
    mesh_strain_r = compute_MOSES_mesh_strain_curve(MOSES_mesh_strain_time_r, normalise=False)
    mesh_strain_g = compute_MOSES_mesh_strain_curve(MOSES_mesh_strain_time_g, normalise=False)
    
    mesh_strain_curve_video = .5*(mesh_strain_r+mesh_strain_g)
    normalised_mesh_strain_curve_video = mesh_strain_curve_video/ np.max(mesh_strain_curve_video)
    
    plt.figure()
    plt.plot(mesh_strain_r, 'r-', lw=2., label='red cells')
    plt.plot(mesh_strain_g, 'g-', lw=2., label='green cells')
    plt.xlabel('Frame Number')
    plt.ylabel('Mesh Strain')
    plt.legend(loc='best')
    plt.savefig('MOSES_mesh_strain_curve_red-green.png', dpi=80, bbox_inches='tight', pad_inches=0)
    plt.show()
    
    
    """
    7. Infer the gap closure frame, using image segmentation.
    """
    print 'Inferring the gap closure frame...'
    wound_closure_frame = wound_sweep_area_segmentation(vidstack, spixel_size, max_frame=50, n_sweeps=50, n_points_keep=1, n_clusters=2, p_als=0.001, to_plot=True)
    
    print 'predicted gap closure frame is: %d' %(wound_closure_frame)
    
    
    """
    8. Compute the Metrics Presented in the paper:
        a) Boundary formation index
        b) Mesh Stability Index
        c) Max. Velocity cross correlation Index
        d) Spatial Correlation 
        e) Mesh Order Curve/Index
    """
    #==============================================================================
    #     a) Boundary Formation Index.
    #==============================================================================
    final_saliency_map_r, spatial_time_saliency_map_r = compute_motion_saliency_map(meantracks_r, dist_thresh=5.*spixel_size, shape=(n_rows, n_cols), filt=1, filt_size=spixel_size)
    final_saliency_map_g, spatial_time_saliency_map_g = compute_motion_saliency_map(meantracks_g, dist_thresh=5.*spixel_size, shape=(n_rows, n_cols), filt=1, filt_size=spixel_size)
    
    boundary_formation_index, av_saliency_map = compute_boundary_formation_index(final_saliency_map_r, final_saliency_map_g, spixel_size, pad_multiple=3)
    
    
    fig = plt.figure()
    plt.title('Boundary Formation Index %.3f' %(boundary_formation_index))
    plt.imshow(av_saliency_map, cmap='coolwarm')
    plt.axis('off')
    plt.grid('off')
    fig.savefig('av_saliency_map.png', dpi=80, bbox_inches='tight', pad_inches=0)
    plt.show()
    
    #==============================================================================
    #     b) Motion Stability Index.
    #==============================================================================
    
    # returns the combined normalised mesh strain curve.
    mesh_stability_index, normalised_mesh_strain_curve = compute_MOSES_mesh_stability_index(MOSES_mesh_strain_time_r, MOSES_mesh_strain_time_g, last_frames=5)
    
    print 'mesh stability index: %.3f' %(mesh_stability_index)
    
    #==============================================================================
    #     c) Max. Velocity Cross-Correlation Index
    #==============================================================================
    (max_vccf_before, _), (max_vccf_after, _) = compute_max_vccf_cells_before_after_gap(meantracks_r, meantracks_g, wound_heal_frame=wound_closure_frame, err_frame=5)
    
    print 'max velocity cross-correlation before: %.3f' %(max_vccf_before)
    print 'max velocity cross-correlation after: %.3f' %(max_vccf_after)
    
    
    #==============================================================================
    #     d) Spatial Correlation. 
    #==============================================================================
    spatial_corr_curve, (spatial_corr_pred,  a_value, b_value, r_value) = compute_spatial_correlation_function(meantracks_r, wound_closure_frame, wound_heal_err=5, dist_range=np.arange(1,6,1))
    
    # plot the curve and the fitted curve to y=a*exp(-x/b) to get the (a,b) parameters. 
    plt.figure()
    plt.title('Fitted Spatial Correlation: a=%.3f, b=%.3f' %(a_value, b_value))
    plt.plot(np.arange(1,6,1), spatial_corr_curve, 'ko', label='measured')
    plt.plot(np.arange(1,6,1), spatial_corr_pred, 'g-', label='fitted')
    plt.xlabel('Distance (Number of Superpixels)')
    plt.ylabel('Spatial Correlation')
    plt.legend(loc='best')
    plt.show()

    
    #==============================================================================
    #     e) Mesh Order Curve + mesh strain vector
    #==============================================================================
    mesh_strain_vector_r = construct_mesh_strain_vector(meantracks_r, [MOSES_mesh_neighborlist_r]) # if to_plot, displays the mesh with ellipses. red ellipses is excluded points due to high density as set by the threshold, blue ellipses is included in calculation.
    mesh_strain_vector_g = construct_mesh_strain_vector(meantracks_g, [MOSES_mesh_neighborlist_g])

    mesh_order_curve_r = compute_mesh_order(mesh_strain_vector_r, remove_mean=False)
    mesh_order_curve_g = compute_mesh_order(mesh_strain_vector_g, remove_mean=False)
    
    print 'mean mesh order red: %.3f' %(np.nanmean(mesh_order_curve_r))
    print 'mean mesh order green: %.3f' %(np.nanmean(mesh_order_curve_g))
    
    
    plt.figure()
    plt.title('Mesh Order')
    plt.plot(mesh_order_curve_r, 'r', label='Red')
    plt.plot(mesh_order_curve_g, 'g', label='Green')
    plt.legend(loc='best')
    plt.xlabel('Frame Number')
    plt.ylabel('Mesh Order')
    plt.show()
    

    # we can also visualise the mesh strain vector on the mesh using matplotlib quiver plot 
    fig, ax = plt.subplots()
    ax.imshow(vidstack[20], alpha=0.5)
    visualise_mesh(mesh_frame20_networkx_G_red, meantracks_r[:,20,[1,0]], ax, node_size=.5, node_color='r')
    ax.quiver(meantracks_r[:,20,1], meantracks_r[:,20,0], mesh_strain_vector_r[20,:,1], mesh_strain_vector_r[20,:,0], color='r', scale_units='xy')
    # note the minus sign for the y vector velocity. This is to abide with image conventions used in python.
    ax.set_ylim([n_rows,0])
    ax.set_xlim([0,n_cols])
    plt.show()
    
    fig, ax = plt.subplots()
    ax.imshow(vidstack[20], alpha=0.5)
    visualise_mesh(mesh_frame20_networkx_G_green, meantracks_g[:,20,[1,0]], ax, node_size=.5, node_color='g')
    ax.quiver(meantracks_g[:,20,1], meantracks_g[:,20,0], mesh_strain_vector_g[20,:,1], mesh_strain_vector_g[20,:,0], color='g', scale_units='xy')
    # note the minus sign for the y vector velocity. This is to abide with image conventions used in python.
    ax.set_ylim([n_rows,0])
    ax.set_xlim([0,n_cols])
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    