# -*- coding: utf-8 -*-
import numpy as np

def smooth_1D(data, box_pts):
    """
    This function smooth 1D data based on a moving average box (by convolution).

    Parameters
    ----------
    data: np.Array
        1D-data to smooths.
        
    box_pts: Int
        Size of the box for the moving averege calcuation.


    Returns
    -------
    XY1: Array float64.
        Smooth Data

        
Created on Sat Jul 23 15:50:12 2022

@author: facundo.morici
"""
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(data, box, mode='same')
    return y_smooth