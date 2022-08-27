## -- Upload Digital Inputs file -- ##
def load_digital_inputs(path,fs = 20000, ch = [0,1,2,3]):
    """
     Reads Intan Technologies RHD2000 digital inputs.
     In the folder It is required to be info.rhd and digitalin.dat files
     
    Parameters
    ----------
         path: String 
             Path where .rhd and .dat files are located
             
         fs: Int
             DigitalIn Sampling frequency, used for time vector. Default: 20000
         
    Returns
    ----------
         camara = List bool values for channel 1
         shock = List bool values for channel 2
         left_valve = List bool values for channel 3
         right_valve = List bool values for channel 4
         time = time vector starting from dt=1/fs the the len(DI)*dt each dt
         
         
         
    Created on Tue Jul 19 15:31:48 2022

    @author: facundo.morici     
    """    
    import sys, struct, math, os, time
    import pandas as pd
    import numpy as np
    import os
    import sys
    import glob

    print('--- Lets load the TTLs from your Channels ---')
    
    # onlyfiles = [f for f in listdir(currentdir) if isfile(join(currentdir, f))]
    filename = '*digitalin.dat'
    file = [path+'\\'+filename]
    file = glob.glob(file[0])
    num_samples = int(os.path.getsize(file[0])/2) #intu16 = 2 bytes
    with open (file[0], 'rb') as fid:
            digitalin = np.fromfile(fid, dtype=np.uint16)
    print('DigitalIn file open')
    
    DI = [(1*((digitalin & a)>0)) for a in np.power(2, ch)] 
    print('DigitalIn data separated by channels')
    camara = DI[0]
    shock = DI[1]
    left_valve = DI[2]
    right_valve = DI[3]
    dt = 1/fs
    time = np.linspace(dt,len(digitalin)*dt,len(digitalin))   
    # time = time.tolist()
    del DI
    del digitalin

    print('Time vector generated')
    # plt.plot(time,DI[0])
    # plt.show()
    # time[-1]/60

    print('Camara, Shock, left and Right valves TTLs and time vector saved')
    return camara,shock,left_valve,right_valve,time