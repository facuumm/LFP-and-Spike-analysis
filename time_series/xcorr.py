# -*- coding: utf-8 -*-



import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

def xcorr(x, y, maxlags=125):
    """
    This function split the input data in time bins and then it calculate the
    mean for each bin.
    
    Parameters
    ----------
        x: np.Array
            first time series data to cross-correlate
            
        y: np.Array
            second time series data to cross-correlate
            
        maxlags: Int
            Size of time window to show.
    
    
    Returns
    -------
        corr: np.Array
            This variable stores the cross-correlation coeff
        
        lags: np.Array
            This variable stores the lags.
    Created on Mon Aug 22 15:37:11 2022

    @author: facundo.morici
    """

    Nx = len(x)
    if Nx != len(y):
        raise ValueError('x and y must be equal length')

    c = signal.correlate(x, y, mode='full')
    lags = signal.correlation_lags(maxlags+1, maxlags+1, mode="full")
    if maxlags is None:
        maxlags = Nx - 1

    if maxlags >= Nx or maxlags < 1:
        raise ValueError('maxlags must be None or strictly positive < %d' % Nx)

    corr = c[Nx - 1 - maxlags:Nx + maxlags]

    return corr, lags