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
    networkx_graph: 
    node_positions: 
    ax: matplotlib axis to plot onto.
    
    Outputs:
    --------
        nothing, it is a void function.
    
    """
    import networkx as nx 
    
    nx.draw(networkx_graph, ax=ax, pos = node_positions, node_size=node_size, node_color=node_color, width=width, linewidths=linewidths)
    
    return []
    
    
    
    