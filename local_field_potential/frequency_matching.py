import sys, struct, math, os, time
import pandas as pd
import numpy as np


def frequency_matching(to_be_matched , template):
    """
     Find the nearest value inside an array
     
    Parameters
    ----------
         to_be_matched: np.array
             Array containing the elements to be compared
             
         template: np.array
             Array that will work as template.
             Needs to be shorter that to_be_matched array
         
    Returns
    ----------
         indx: array
             indexes of the nearest element in the array
         
         nearest: array
             The elements that is the closes to the template         
         
    Created on Thu Aug 18 18:25:57 2022

    @author: facundo.morici     
    """    

    to_be_matched = np.asarray(to_be_matched)
    template = np.asarray(template)
    nearest = []
    indx = []
    for i in range(0,len(template)):
        x,y = find_nearest(to_be_matched,template[i])
        nearest.append(x)
        indx.append(y)
        del x, y
    nearest = np.array(nearest)
    indx = np.array(indx)
    return indx, nearest
        
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx