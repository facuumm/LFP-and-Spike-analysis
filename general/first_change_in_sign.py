# -*- coding: utf-8 -*-

def first_change_in_sign(list_to_check,value,sign):
    """
    Returns the index and value of the first element with the input sign.
    
    Parameters
    ----------
         list_to_check: list 
             list to check the sign
             
         sign: Int
             It can be positive (1) or negative (-1)
         
    Returns
    ----------
         LFP: pd.DataFrame
             Contains all the LFP signals.
             Labels: ['time' , 'channel_0' , ... , 'channel_X']
             
    Created on Mon Jul 25 20:27:49 2022

    @author: facundo.morici
    """
    
    x = list_to_check - value
    if sign > 0:
        for i, e in enumerate(x):
            if e > 0:
                output = [i]
                break
    elif sign < 0:
        for i, e in enumerate(x):
            if e < 0:
                output = [i]
                break
    else:
        raise ValueError('Check if sign input is -1 or 1')   
        
    return output
    