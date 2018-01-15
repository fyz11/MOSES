# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 15:31:11 2014

@author: felix
@email: felixzhou1@gmail.com

This script and all the code within is licensed under the Ludwig License.

see README.md for details on usage.
    
"""

def Eval_dense_optic_flow(prev, present, params):
    
    """
    Input:
        prev: previous frame, m x n image
        present: current frame, m x n image
        params: a dict object to pass all algorithm parameters.
            fields are the same as that in opencv docs:
            recommended starting values:
            
            params['pyr_scale'] = 0.5
            params['levels'] = 3
            params['winsize'] = 15
            params['iterations'] = 3
            params['poly_n'] = 5
            params['poly_sigma'] = 1.2
            params['flags'] = 0
        
    Output:
        flow: finds the displacement field between frames, prev and present such that
                prev(y,x) = next(y+flow(y,x)[1], x+flow(y,x)[0])
        where (x,y) is the cartesian coordinates of the image.
    """
    
    import numpy as np 
    import warnings
    import cv2

    # Check version of opencv installed, if not 3.0.0 then issue alert.
#    if '3.0.0' in cv2.__version__ or '3.1.0' in cv2.__version__:
        # Make the image pixels into floats.
    prev = prev.astype(np.float)
    present = present.astype(np.float)

    if '3.' in cv2.__version__:
        flow = cv2.calcOpticalFlowFarneback(prev, present, None, params['pyr_scale'], params['levels'], params['winsize'], params['iterations'], params['poly_n'], params['poly_sigma'], params['flags']) 
    else:
        flow = cv2.calcOpticalFlowFarneback(prev, present, pyr_scale=params['pyr_scale'], levels=params['levels'], winsize=params['winsize'], iterations=params['iterations'], poly_n=params['poly_n'], poly_sigma=params['poly_sigma'], flags=params['flags']) 
    
    return flow
