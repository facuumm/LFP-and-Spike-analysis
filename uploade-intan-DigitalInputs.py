%reset -f
## this script uploade the digital input from intan and split per channel
from os import listdir
import os
from os.path import isfile, join
import numpy as np

currentdir = 'Y:\Facundo\Rat37\Rat37-020721\Rat37-f_210702_172154'

## for info file
filename = 'info.rhd'
file = [currentdir+'\\'+filename]

## for time file
filename = 'time.dat'
file = [currentdir+'\\'+filename]
num_samples = int(os.path.getsize(file[0])/4) #int32 = 4 bytes
fid = open(file[0], 'rb') 
time = np.fromfile(fid, dtype=np.int32)
time = time/20000;

# onlyfiles = [f for f in listdir(currentdir) if isfile(join(currentdir, f))]
filename = 'digitalin.dat'
file = [currentdir+'\\'+filename]
num_samples = int(os.path.getsize(file[0])/4) #int32 = 4 bytes
fid = open(file[0], 'rb') 
digitalin = np.fromfile(fid, dtype=np.uint16)
ch = [np.bitwise_and(digitalin,a)>0 for a in np.power(2, range(16))]
#%%
import matplotlib.pyplot as plt


#%%
import numpy as np
for i in range(16):
    plt.figure(figsize = (12,4))
    a = plt.plot(ch[i])