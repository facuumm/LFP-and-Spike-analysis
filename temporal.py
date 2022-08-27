#general libraries
import sys, struct, math, os, time, glob, win32evtlog
sys.path.append('Y:\\Facundo\\codes\\python')
import numpy as np
import matplotlib.pyplot as plt
import pynapple as nap
from datetime import datetime, timedelta
from pymatch.Matcher import Matcher
import scipy
from scipy import fftpack
import math
import pandas as pd

##libraries for LFP analyses
import nitime
from nitime.timeseries import TimeSeries
from nitime import utils
import nitime.algorithms as alg
import nitime.viz
from nitime.viz import drawmatrix_channels
from nitime.analysis import CoherenceAnalyzer, MTCoherenceAnalyzer
from nitime.algorithms import spectral

##Custom functions
from functions.load_lfp import load_lfp
from functions.load_spks import load_spks
from functions.load_digital_inputs import load_digital_inputs
from functions.ToIntervals import ToIntervals
from functions.get_pos import get_pos
from general.smooth_1D import smooth_1D
from time_series.dataTS_binning import dataTS_binning
from time_series.xcorr import xcorr
from behavior.LinearVelocity import LinearVelocity
from behavior.get_matching_pairs import get_matching_pairs
from functions.selecting_events_TS import selecting_events_TS
from general.first_change_in_sign import first_change_in_sign
from general.find_nearest import find_nearest
from local_field_potential.frequency_matching import frequency_matching
from general.get_index_positions import get_index_positions


# def load_pos(path)
speed_l = 10  #minimal velocity to filter the data
speed_h = 30 #maximal velocity to filter the data    

subfolders = [ f.path for f in os.scandir('D:\\analysis') if f.is_dir() ]
sf = 1250 #sampling frequency

SxxA = []
SyyA = []
SxyA = []
CxyA = []
FA = []

SxxR = []
SyyR = []
SxyR = []
CxyR = []
FR = []

crossA = []
crossR = []

dHPC = [77,77,73,77,77,90,90,90]
vHPC = [57,49,49,49,49,8,8,8]

FreqAnalyses = False #to define if PSD, Cross-PSD and coherence will be done
AmpCrossCorr = True #to define if amplitude cross-Corr will be done
Spikes = False

#to define wich events will be uploaded
Events_to_upload = {'camara':True,'shock':False,'left_valve':False,'right_valve':False}

###### -- Main Code -- ######
for i in range(1,len(subfolders)-1):
    path = subfolders[i]
    print(f'\n--- Opening the following session: {path} ---')
    _ , lfp = load_lfp(path, channels = np.array([dHPC[i] , vHPC[i]]))
    # print(f'\n{lfp.shape[1]-1} LFP channels correctly uploaded')

    if Spikes:
        [spks_good, spks, K] = load_spks(path,64)
        del spks, K
        print('\nSpikes correclty loaded')

    Events = [f.path for f in os.scandir(path) if 'cat.evt' in f.path]#selecting cat.evt file
    Events = pd.read_csv(Events[0],header=None,sep=' ')
    Events = selecting_events_TS(Events,['aversive' , 'reward'])
    print(f'TimeStamps from  concatenated files selected')    

    [camara , shock , left_valve , right_valve , t] = load_digital_inputs(path)
    
    if Events_to_upload['camara']:
        print('\n--- Generating TTLs intervals for camara input ---')
        [camara_idx , camara_time] = ToIntervals(camara,t) 
    
    if Events_to_upload['shock']:
        print('\n--- Generating TTLs intervals for shock input ---')
        [shock_idx , shock_time] = ToIntervals(shock,t) 
    
    if Events_to_upload['left_valve']:
        print('\n--- Generating TTLs intervals for left_valve input ---')
        [left_valve_idx , left_valve_time] = ToIntervals(left_valve,t)
    
    if Events_to_upload['right_valve']:
        print('\n--- Generating TTLs intervals for right_valve input ---')
        [right_valve_idx , right_valve_time] = ToIntervals(right_valve,t)
    
    print('\n--- Let load the position of your behavioral sessions ---')
    XY1 , XY2 = get_pos(path,'orting','neck',[1 , 3],21/100)
    
    print('\n--- Calculating the velocity for both sessions ---')
    velocity1 = LinearVelocity(XY1 , smooth = True , sigma = 6)
    velocity2 = LinearVelocity(XY2 , smooth = True , sigma = 6)
    print('Done.')
    
    print('\n--- Preparing velocity data to be used ---')

    print('Binning the velocity')
    _ , binned1 = dataTS_binning(velocity1,1)
    _ , binned2 = dataTS_binning(velocity2,1)
    print('Done.')
    
    print(f'Filtering your Velocity bins beweetn {speed_l} and {speed_h}')
    binned1 = binned1[np.logical_and(binned1.values > speed_l,binned1.values < speed_h)]
    binned2 = binned2[np.logical_and(binned2.values > speed_l,binned2.values < speed_h)]
    print('Done.')
    print('Reseting index of the DataFrame')
    v1 = binned1.reset_index()
    v2 = binned2.reset_index()
    print('Done.')

    print('--- Matching bin-by-bin form speed data of both conditions ---')
    if v1.shape[0] < v2.shape[0]:
        template = v1
        to_be_matched = v2
        config = 1
    else:
        template = v2
        to_be_matched = v1
        config = 2
        
    matched , indexes = get_matching_pairs(template['Values'], to_be_matched['Values'])    
    to_be_matched = to_be_matched.loc[indexes]        
    print('Done.')

    print('Aligning bins to the bigining of each condition')
    # This loop is to find the closes positive value
    aversive_Start = first_change_in_sign(camara_time['Start'] , Events['aversive']['Start']/1000 , sign = 1)
    reward_Start = first_change_in_sign(camara_time['Start'] , Events['reward']['Start']/1000 , sign = 1)
    

    if config == 1:
        v1 = template
        v2 = to_be_matched
        del config
    else:
        v2 = template
        v1 = to_be_matched
        del config
    
    if aversive_Start < reward_Start:
        aversive = camara_time['Start'][aversive_Start].values
        tmp = v1['Bins'].values.tolist()
        vA = [tmp[j].right + aversive[0] for j in range(0,len(v1))]
        del aversive, tmp
        
        reward = camara_time['Start'][reward_Start].values
        tmp = v2['Bins'].values.tolist()
        vR = [tmp[j].right + reward[0] for j in range(0,len(v1))]
        del reward, tmp
    else:
        aversive = camara_time['Start'][aversive_Start].values
        tmp = v2['Bins'].values.tolist()
        vA = [tmp[j].right + aversive[0] for j in range(0,len(v1)-1)]
        del aversive, tmp
        
        reward = camara_time['Start'][reward_Start].values
        tmp = v1['Bins'].values.tolist()
        vR = [tmp[j].right + reward[0] for j in range(0,len(v1)-1)]
        del reward, tmp
    print ('Done')
    
    print('\n--- Restricting the LFP to the behavioral events ---')
      
    print('First, Aversive')
    print('Looking for the the selected bins into the time vector')
    idxA = [(np.abs(lfp[:,0] - vA[j])).argmin() for j in range(0,len(vA)-1)]
    print('Now, restricting the lfp in each one-sec bin')
    lfpA = np.empty((0,lfp.shape[1]))
    for j in range(0,len(idxA)):
        lfpA = np.append(lfpA,lfp[idxA[j]:idxA[j]+sf,:],axis=0)
    print('Done.')

    print('\nNow, Reward')
    print('Looking for the the selected bins into the time vector')
    idxR = [(np.abs(lfp[:,0] - vR[j])).argmin() for j in range(0,len(vR)-1)]
    lfpR = np.empty((0,lfp.shape[1]))
    for j in range(0,len(idxA)):
        lfpR = np.append(lfpR,lfp[idxR[j]:idxR[j]+sf,:],axis=0)
    print('Done.')
    
    if FreqAnalyses:
        print('\n--- Time Frequency analyses ---')
        print('First Aversive condition')
        to_cross = [lfpA[:,1],lfpA[:,2]]#,np.array(range(0,len(lfpA)))]
        to_cross = np.array(to_cross)
        f , xy = spectral.multi_taper_csd(to_cross, BW=1, Fs=sf, low_bias=False, NFFT=sf)
        x = xy[0,0,:]
        y = xy[1,1,:]
        cxy = np.abs(xy[0,1,:]) ** 2
        cxy = cxy / (x * y) 
        
        print('Saving results')
        SxxA.append(x)
        SyyA.append(y)
        SxyA.append(xy[0,1,:])    
        CxyA.append(cxy)   
        FA.append(f)
        print('Done')
        del x, y, xy, cxy, to_cross
        
        print('Now Reward condition')
        to_cross = [lfpR[:,1],lfpR[:,2]]#,np.array(range(0,len(lfpA)))]
        to_cross = np.array(to_cross)
        f , xy = spectral.multi_taper_csd(to_cross, BW=1, Fs=sf, low_bias= False, NFFT = sf)
        x = xy[0,0,:]
        y = xy[1,1,:]
        cxy = np.abs(xy[0,1,:]) ** 2
        cxy = cxy / (x * y) 
        
        print('Saving results')
        SxxR.append(x)
        SyyR.append(y)
        SxyR.append(xy[0,1,:])    
        CxyR.append(cxy)   
        FR.append(f)
        print('Done')
        del x, y, xy, cxy, to_cross  
    
    if AmpCrossCorr:
        print('\n--- Amplitude Cross-Correlation ---')
        print('First Aversive')
        b, a = scipy.signal.butter(4, [6,9], btype='bandpass', fs=sf)
        x = scipy.signal.filtfilt(b, a, lfpA[:,1])
        y = scipy.signal.filtfilt(b, a, lfpA[:,2])
        
        x = np.abs(scipy.signal.hilbert(x))
        y = np.abs(scipy.signal.hilbert(y))
        
        c , lags = xcorr(x,y,maxlags=125)
        
        # lags , c , _ , _ = plt.xcorr(x, y, normed=False, usevlines=False, maxlags=250)
        crossA.append(c)
        del a, b, x, y, c, lags
        
        print('Now Reward')
        b, a = scipy.signal.butter(4, [6,10], btype='bandpass', fs=sf)
        x = scipy.signal.filtfilt(b, a, lfpR[:,1])
        y = scipy.signal.filtfilt(b, a, lfpR[:,2])
        
        x = np.abs(scipy.signal.hilbert(x))
        y = np.abs(scipy.signal.hilbert(y))
        
        c , lags = xcorr(x,y,maxlags=125)
        # lags , c , _ , _ = plt.xcorr(x, y, normed=False, usevlines=False, maxlags=250)
        crossR.append(c)
        del a, b, x, y, c
        print('Done.')
    
    print('\n--- Reseting variables before starting the loop ---')
    del binned1, binned2, camara, camara_idx, camara_time, Events
    del lfpA, lfpR
    del right_valve
    del reward_Start, aversive_Start
    del t, template, to_be_matched, v1, v2, vA, vR
    del velocity1, velocity2, XY1, XY2
    
    
tmp = []
for j in range(0,i+1):
    tmp.append(len(FA[j]))

selected = FA[get_index_positions(tmp, min(tmp))[0]]
selected=np.arange(0,600,0.1)

freqs = []
xA = []
yA = []
xyA = []
cxyA = []
xR = []
yR = []
xyR = []
cxyR = []
for j in range(0,i+1):
    indx , _ = frequency_matching(FA[j],selected)
    freqs.append(FA[j][indx])
    xA.append(SxxA[j][indx])
    yA.append(SyyA[j][indx])
    xyA.append(SxyA[j][indx])
    cxyA.append(CxyA[j][indx])
    xR.append(SxxR[j][indx])
    yR.append(SyyR[j][indx])
    xyR.append(SxyR[j][indx])
    cxyR.append(CxyR[j][indx])
    del indx




SxxA1 = np.array(xA)
SyyA1 = np.array(yA)
SxyA1 = np.array(xyA)
CxyA1 = np.array(cxyA)

SxxR1 = np.array(xR)
SyyR1 = np.array(yR)
SxyR1 = np.array(xyR)
CxyR1 = np.array(cxyR)
f1 = np.array(freqs)

meanSxxA = np.mean(SxxA1,axis=0)
semSxxA = scipy.stats.sem(SxxA1,axis=0)
semSxxA = [meanSxxA-semSxxA , meanSxxA+semSxxA]

meanSxxR = np.mean(SxxR1,axis=0)
semSxxR = scipy.stats.sem(SxxR1,axis=0)
semSxxR = [meanSxxR-semSxxR , meanSxxR+semSxxR]


meanSyyA = np.mean(SyyA1,axis=0)
semSyyA = scipy.stats.sem(SyyA1,axis=0)
semSyyA = [meanSyyA-semSyyA , meanSyyA+semSyyA]

meanSyyR = np.mean(SyyR1,axis=0)
semSyyR = scipy.stats.sem(SyyR1,axis=0)
semSyyR = [meanSyyR-semSyyR , meanSyyR+semSyyR]

meanSxyA = np.mean(np.abs(SxyA1),axis=0)
semSxyA = scipy.stats.sem(np.abs(SxyA1),axis=0)
semSxyA = [meanSxyA-semSxyA , meanSxyA+semSxyA]

meanSxyR = np.mean(np.abs(SxyR1),axis=0)
semSxyR = scipy.stats.sem(np.abs(SxyR1),axis=0)
semSxyR = [meanSxyR-semSxyR , meanSxyR+semSxyR]

meanCxyA = np.mean(np.abs(CxyA1),axis=0)
semCxyA = scipy.stats.sem(np.abs(CxyA1),axis=0)
semCxyA = [meanCxyA-semCxyA , meanCxyA+semCxyA]

meanCxyR = np.mean(np.abs(CxyR1),axis=0)
semCxyR = scipy.stats.sem(np.abs(CxyR1),axis=0)
semCxyR = [meanCxyR-semCxyR , meanCxyR+semCxyR]

figure1, (ax1,ax2,ax3,ax4) = plt.subplots(4,1)
ax1.plot(selected,meanSxxA.T,'-r')
ax1.fill_between(np.array(selected),np.array(semSxxA[0]),np.array(semSxxA[1]),facecolor='red',alpha=0.2);ax1.set_xlim([0,20]);ax1.set_ylim([0,2500])

ax1.plot(selected,meanSxxR.T,'-b')
ax1.fill_between(np.array(selected),np.array(semSxxR[0]),np.array(semSxxR[1]),facecolor='blue',alpha=0.2);ax1.set_xlim([0,20]);ax1.set_ylim([0,2500])
ax1.set_ylabel('Power')

ax2.plot(selected,meanSyyA.T,'-r')
ax2.fill_between(np.array(selected),np.array(semSyyA[0]),np.array(semSyyA[1]),facecolor='red',alpha=0.2);ax2.set_xlim([0,20]);ax2.set_ylim([0,5000])

ax2.plot(selected,meanSyyR.T,'-b')
ax2.fill_between(np.array(selected),np.array(semSyyR[0]),np.array(semSyyR[1]),facecolor='blue',alpha=0.2);ax2.set_xlim([0,20]);ax2.set_ylim([0,5000])
ax2.set_ylabel('Power')

ax3.plot(selected,meanSxyA.T,'-r')
ax3.fill_between(np.array(selected),np.array(semSxyA[0]),np.array(semSxyA[1]),facecolor='red',alpha=0.2);ax3.set_xlim([0,20]);ax3.set_ylim([0,2500])

ax3.plot(selected,meanSxyR.T,'-b')
ax3.fill_between(np.array(selected),np.array(semSxyR[0]),np.array(semSxyR[1]),facecolor='blue',alpha=0.2);ax3.set_xlim([0,20]);ax3.set_ylim([0,2500])
ax3.set_ylabel('Power')

ax4.plot(selected,meanCxyA.T,'-r')
ax4.fill_between(np.array(selected),np.array(semCxyA[0]),np.array(semCxyA[1]),facecolor='red',alpha=0.2);ax4.set_xlim([0,20]);ax4.set_ylim([0,0.5])

ax4.plot(selected,meanCxyR.T,'-b')
ax4.fill_between(np.array(selected),np.array(semCxyR[0]),np.array(semCxyR[1]),facecolor='blue',alpha=0.2);ax4.set_xlim([0,20]);ax4.set_ylim([0,0.5])
ax4.set_ylabel('coherence'); ax4.set_xlabel('Frequency(Hz)')

del meanSxxA, semSxxA, meanSxxR, semSxxR, meanSyyA, semSyyA, meanSyyR, semSyyR
del meanSxyA, semSxyA, meanSxyR, semSxyR, meanCxyA, semCxyA, meanCxyR, semCxyR


crossA = np.array(crossA)
crossR = np.array(crossR)

tmp1 = []
tmp2 = []

for i in range(0,np.shape(crossA)[0]):
    x = crossA[i,:]-min(crossA[i,:])
    x /= max(x)
    tmp1.append(x)
    del x
    
    x = crossR[i,:]-min(crossR[i,:])
    x /= max(x)
    tmp2.append(x)
    del x
       

tmp1 = np.array(tmp1) 
tmp2 = np.array(tmp2) 

figure1, (ax1,ax2) = plt.subplots(1,2)
ax1.plot(lags*1/1250*1000,tmp1.T[:,1:-1],'r')
ax1.axvline(x=0, ymin=0, ymax=1,linestyle='dashed')
ax1.set_ylabel('Norm Corr Coeff'); ax1.set_xlabel('time (ms)')


ax2.plot(lags*1/1250*1000,tmp2.T[:,1:-1],'b')
ax2.axvline(x=0, ymin=0, ymax=1,linestyle='dashed')
ax2.set_xlabel('time (ms)')

