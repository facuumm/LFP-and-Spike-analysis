#%% Clear console and variables
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

#%% Import libraries
import os
import numpy as np
import glob
import csv
import math
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

#%%Parameters
#Directory to analyse
currentdir = ('Y:\Facundo\Rat37');
folders = os.listdir(currentdir)

#Heatmaps
X = 60 # Number of bins to construct heatmaps
Y = 20 # Number of bins to construct heatmaps
mySmooth = 9 #sice of gaussian smooth for heatmaps
cmin = 0
cmax = 10

#Time
dt = 1/30 #1sec/30frames
tt = 120 #how many time befor the begining of each lap I want to plot

#Parameters for Gaussian
sigma = 20 # pick sigma value for the gaussian
#gaussFilter = gausswin(6*sigma + 1)
#gaussFilter = gaussFilter / sum(gaussFilter) # normalize

#Pixel/cm ratio (21cm = 100px)
pixcorrection = 21/100

#Platform limits to count the number of laps
xi = 20
xf = 170

xitot = 70
xftot = 100


store = [] # storage for neck position (x and y)
index = [] # storage for alternation session position (start and end)
v = [] # storage for velocity
vectormatrix = []
downsampled = []

for i in range(0,len(folders)):
    #print(folders[i])
    subfolder = os.listdir(currentdir + '\\' + folders[i])
    for ii in range(0,len(os.listdir(currentdir + '\\' + folders[i]))):
        #print(subfolder[ii]) #debug    
        files = glob.glob(currentdir + '\\' + folders[i] + '\\' + subfolder[ii] + '\\*.csv')
        files = files[0]
        print(files) #debug
        table = np.genfromtxt(files,dtype=float,skip_header=3,delimiter = ',')
        table = table[:,4:7]
        
        posx = table[:,0]     
        posy = table[:,1]     
        v = []
        t = np.arange(0,len(posx)*dt,dt)
        #calculate instantaneus velocity
        for b in range(0,len(posx)-1):
            v.append(math.sqrt(abs((posx[b+1]-posx[b])*pixcorrection/(t[b+1]-t[b])**2 + (posy[b+1]-posy[b])*pixcorrection/(t[b+1]-t[b])**2)))
        
        v = np.array(v)
        #v = gaussian_filter1d(v,1)
        plt.figure(figsize = (12,4))
        a = plt.plot(v)     
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             