def load_spks(path,cutting):
    """
    Upload Spikes from SpikeSorting folder inside path

    Parameters
    ----------
    path: String
        Path of the Phy output

    cutting: Int
        Channel ID to split Spks from dorsal and ventral (starting from 0)        

    Returns
    -------
    spks_good: Array float64.
        Contains the timestamps in seconds of each spike in the first column
        and the Cluster ID inthe second one. Only from the good clusters
        
    spks: Array float64.
        Contains the timestamps in seconds of each spike in the first column
        and the Cluster ID inthe second one. All clusters, including MU and Noise    
        
    K: list
        Clusters ID + Category ('noise', 'mua', 'good')
        
        
   Created on Tue Jul 19 15:33:35 2022

   @author: facundo.morici
        
    """
    import pandas as pd
    import numpy as np
    import os
    
    print('--- Loading your Spikes ---')
    subfolders = [f.path for f in os.scandir(path) if f.is_dir() and ('orting' in f.path)]
    spks_time = np.load(subfolders[0]+'\\spike_times.npy').reshape(-1,1) #load spikes
    spks_clusters = np.load(subfolders[0]+'\\spike_clusters.npy').reshape(-1,1) #load clusters
    spks = np.hstack((spks_time , spks_clusters)) #concatenation of Spks and clusters
    K  = pd.read_csv(subfolders[0]+'\\cluster_group.tsv',sep='\t') #load info from each cluster
    Kinfo = pd.read_csv(subfolders[0]+'\\cluster_info.tsv',sep='\t')
    del spks_time, spks_clusters
    
    K = K.values.tolist()
    
    if K:
        good_ones = [i[0] for i in K if 'g' in i[1] and i[0] >= cutting] #selection of only good spikes
        tmp = [True if i in good_ones else False for i in spks[:,1]]
        spks_good = spks[tmp]
        print('Done.')
        return spks_good, spks, K
    else:
        return print('You do not have good SU in channels above your cutting parameter: {cutting}')