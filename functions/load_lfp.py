import sys, struct, math, os, time
import pandas as pd
import numpy as np
import os
import sys
import glob
from load_intan.read_header import read_header
    # from load_intan.get_bytes_per_data_block import get_bytes_per_data_block
    # from load_intan.read_one_data_block import read_one_data_block
    # from load_intan.notch_filter import notch_filter
    # from load_intan.data_to_result import data_to_result
## -- Upload LFP file -- ##
def load_lfp(path,fs = 1250, save=False, factor=0.195, channels=np.array([])):
    """
     Reads Intan Technologies RHD2000 data and lfp.
     In the folder It is required to be info.rhd and .lfp files
     
    Parameters
    ----------
         path: String 
             Path where .rhd and .lfp files are located
             
         fs: Int
             LFP Sampling frequency of, used for time vector. Default: 1250
             
        save: bool
            If True, it will save the LFP into the introduced path.
            
        factor: float
            Factor to transform into microvolts. Default from RHD intan Tech
            
        channels = np.array
            Define the channels that you would like to export
         
    Returns
    ----------
         LFP: pd.DataFrame
             Contains all the LFP signals. IS NOT IN MICROVOLTS
             Labels: ['time' , 'channel_0' , ... , 'channel_X']
             
         LFP1: np.array
             Contains the LFP signals introduced in channels variable.
             First column is the time vector. 
       
         
    Created on Tue Jul 19 15:31:48 2022

    @author: facundo.morici     
    """    

    
    print('--- Reading the header ---')
    file = glob.glob(path+'\\info.rhd') #upload header
    fid = open(file[0], 'rb')
    header = read_header(fid)
    print('Done.')
    
    print('--- Loading your LFP file ---')
    file = glob.glob(path+'\\*.lfp') #upload lfp
    dividendo = (header['num_amplifier_channels'] + header['num_aux_input_channels']) #number of channels (including auxiliars, eg accelerometers)
    num_samples = int(os.path.getsize(file[0]) / (dividendo * 2))
    with open (file[0], 'rb') as fid:
            lfp = np.fromfile(fid, dtype=np.int16).reshape(-1,dividendo)
    
    LFP1 = np.empty([len(lfp),0])
    for i in channels:
        tmp = np.array(lfp[:,i-1], dtype=float)
        tmp *= factor
        LFP1 = np.column_stack([LFP1,tmp])
        del tmp

    
    lfp = lfp * factor
    dt = 1/fs
    t = np.linspace(dt,len(lfp)*dt,len(lfp))
    labels = [f'channel_{i}' for i in range(0,lfp.shape[1])]
    LFP = pd.DataFrame(data=lfp , columns=labels)
    LFP.insert(0,'time',t)
    
    LFP1 = np.insert(LFP1,0,t,axis=1)
    
    if save:
        LFP.to_csv(os.path.join(path,r'LFP.csv'))

    print('Done.')
    return LFP, LFP1