import numpy as np
import pandas as pd
import math

def dataTS_binning(data, n):
    """
    This function split the input data in time bins and then it calculate the
    mean for each bin.
    
    Parameters
    ----------
        data: np.Array
            1D-data to split in temporal bins.
        
        n: Int
            Size of the bin in seconds.
    
    
    Returns
    -------
        v: DataFrame.
            This variable stores 'TimeStamps', 'Values' and 'Bins' from
            the data input
        
        bined_data_mean: pd.Series
            This Series store the mean of each time bin.
        
        Created on Sat Jul 23 16:41:47 2022

    @author: facundo.morici
    """   
    timestamps = np.linspace(1,len(data),len(data))*1/30
         
    v = pd.DataFrame({'TimeStamps': timestamps, 'Values': data})
        
    v["Bins"] = pd.cut(v["TimeStamps"],math.floor(v['TimeStamps'][len(v)-1]/n)) #downsampling by two orders of magnitude
    bined_data_mean = v.groupby("Bins").mean()["Values"]
     
    return v, bined_data_mean