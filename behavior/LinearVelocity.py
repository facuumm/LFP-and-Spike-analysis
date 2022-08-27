# -*- coding: utf-8 -*-
import numpy as np
import scipy

def Smooth(data, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(data, box, mode='same')
        return y_smooth

def LinearVelocity(positions, smooth = True , sigma = 10):
    """
    This function smooth 1D data based on a moving average box (by convolution).

    Parameters
    ----------
    positions: pd.DataFrame
        Contains position in X and Y for the first session.
        1st column: posX, 2nd column: posY
        
    smooth: bool
        To define if the velocity should be smooth or not. Default: True

    sigma: Int
        Standard deviation for Gaussian kernel. Default: 2


    Returns
    -------
    velocity: Array float64.
        Linear Velocity


    Created on Sat Jul 23 18:17:16 2022

    @author: facundo.morici
    """

    x = np.abs(np.diff(positions['X'].to_numpy()))
    y = np.abs(np.diff(positions['Y'].to_numpy()))

    
    velocity = np.sqrt(np.sum(np.column_stack((x,y)),axis=1)/(1/30)**2)
    
    if smooth:
        velocity = scipy.ndimage.gaussian_filter1d(velocity, sigma)
    
    return velocity
    

