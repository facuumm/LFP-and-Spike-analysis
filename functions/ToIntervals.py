## -- Serching for TTLs? Here is your code ;) -- ##
def ToIntervals(elements,t):
    """
    Detect the begining and end of each TTL
     
    Parameters
    ----------
         list_of_elements: Array 
             elements to dig in
             
        t: list
            Time vector to extrapolate index position.
             
         element: Int
             Element to detect. 1 or -1 are the beginign and ending respectively.
         
    Returns
    ----------
         TTLs_idx: pd.DataFrame
             Position in element for begining ('Start') and end ('End') of each TTL

         TTLs_time: pd.DataFrame
             Time for begining ('Start') and end ('End') of each TTL
             
    Created on Tue Jul 19 15:31:48 2022

    @author: facundo.morici     
    """    
    import sys, struct, math, os, time
    import pandas as pd
    import numpy as np
    import os
    import sys
    import glob
    from general.get_index_positions import get_index_positions

    print('Lets dig into the TTLs from your Channel')
    elements = np.diff(elements)
    elements = elements.tolist()
    

    
    print('Here comes the Start')
    idxI = get_index_positions(elements,1)
    print('Checking integrity of Start indexes')
    if 0 in idxI:
        idxI.remove(0)
    elif len(elements) in idxI:
        idxI.remove(len(elements))
        
    print('Here comes the End')    
    idxF = get_index_positions(elements,-1)
    print('Checking integrity of End indexes')
    if 0 in idxF:
        idxF.remove(0)
    elif len(elements) in idxF:
        idxF.remove(len(elements))
    
    TTLs_idx = pd.DataFrame()
    TTLs_idx['Start'] = idxI
    TTLs_idx['End'] = idxF[0:len(idxI)]    
    
    idxF = t[idxF] 
    idxI = t[idxI] 
    TTLs_time = pd.DataFrame()
    TTLs_time['Start'] = idxI
    TTLs_time['End'] = idxF
    print('Done')    
    return TTLs_idx, TTLs_time


# 