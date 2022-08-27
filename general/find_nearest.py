import sys, struct, math, os, time
import pandas as pd
import numpy as np

def find_nearest(array, value):

    """
     Find the nearest value inside an array
     
    Parameters
    ----------
         array: np.array
             Array containing the elements to be compared
             
         value: int, float
             Element to used as template
         
    Returns
    ----------
         idx: int
             index of the nearest element in the array
         
         nearest: float
             The element that is the closes to the template         
         
    Created on Thu Aug 18 18:25:57 2022

    @author: facundo.morici     
    """    
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx