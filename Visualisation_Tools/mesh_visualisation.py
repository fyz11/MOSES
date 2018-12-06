#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 16:48:04 2017

@author: felix
@email: felixzhou1@gmail.com

This script and all the code within is licensed under the Ludwig License.

see README.md for details on usage.
"""

def visualise_mesh(networkx_graph, node_positions, ax, node_size=10, width=.3, linewidths=.5, node_color='k'):
    
    """
    Plot on a matplotlib axis the graphical connections. 
    
    Inputs:
    -------
    networkx_graph: networkx format representation of the mesh connections betweeen superpixels. see mesh_statistics_tools.from_neighbor_list_to_graph() function.
    node_positions: n_superpixels x 2 array, (x,y) points of where to plot the individual superpixels/points.
    ax: matplotlib axis to plot onto.
    
    Outputs:
    --------
        nothing, it is a void function.
    
    """
    import networkx as nx 
    
    nx.draw(networkx_graph, ax=ax, pos = node_positions, node_size=node_size, node_color=node_color, width=width, linewidths=linewidths)
    
    return []

def visualise_mesh_strain_ellipses(networkx_graph, node_positions, principal_strain, principal_angs, spixel_size, ax, mask=None, node_size=10, width=.3, linewidths=.5, node_color='k'):
    
    """
    Plot on a matplotlib axis the graphical connections. 
    
    Inputs:
    -------
    networkx_graph: networkx format representation of the mesh connections betweeen superpixels. see mesh_statistics_tools.from_neighbor_list_to_graph() function.
    node_positions: n_superpixels x 2 array, (x,y) points of where to plot the individual superpixels/points.
    ax: matplotlib axis to plot onto.
    
    Outputs:
    --------
        nothing, it is a void function.
    
    """
    import networkx as nx 
    import matplotlib as mpl 
    
    nx.draw(networkx_graph, ax=ax, pos = node_positions, node_size=node_size, node_color=node_color, width=width, linewidths=linewidths)

    # draw ellipses to represent the strain.    
    for j in range(len(principal_angs)):
        
        ell = mpl.patches.Ellipse(xy=(node_positions[j,0], node_positions[j,1]),
                    width=spixel_size/2.*2*principal_strain[j,0], height=spixel_size/2.*2*principal_strain[j,1],
                    angle=principal_angs[j])
        
        if mask is not None:
            sel_ = mask[j]
        else:
            sel_ = True # always plot. 

        if sel_ == True:
            ell.set_facecolor('b')
        elif sel_ == False:
            ell.set_facecolor('r')
        ell.set_alpha(0.5)
        ax.add_artist(ell)

    return []
    
    
    
    