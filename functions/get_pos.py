def get_pos(path,string,bodypart,folders,pix_correction = 21/100):
    """
    Upload PosX and PosY from deeplabcut output and keep only those that are in
    the column of interest.

    Parameters
    ----------
    path: String
        Path of the DeepLabCut output
        
    string: str
        String that should not be part of the subfolders in the path to select.
        This is useful to exclude outputs from other processing steps.
        Example: 'orting' to exclude 'Spikesorting' folders

    bodypart: Str
        Choose the bodypart to load       
        Example: 'neck'
        
    folders: list
        Choose the position of the folders to keep.
        Example: If the folders I want to keep ar first and second: [0 , 1]
    
    pix_correction: Int
        Pixel/cm ratio. Default value, the one obtained in Girardeau lab Room2

    Returns
    -------
    XY1: Array float64.
        Contains position in X and Y for the first session.
        1st column: posX, 2nd column: posY
        
    XY2: Array float64.
        Contains position in X and Y for the second session.
        1st column: posX, 2nd column: posY

        
   Created on Tue Jul 19 15:33:35 2022

   @author: facundo.morici
        
    """
    import pandas as pd
    import numpy as np
    import os
    
    print('Loading your CSV files')
    if string:
        subfolders = [f.path for f in os.scandir(path) if f.is_dir() and (string not in f.path)]
    else:
        subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
    
    for i in range(0,len(subfolders)-1):
        if i in folders:
            files = [f.name for f in os.scandir(subfolders[i]) if ('filtered.csv' in f.name)]
            pos = pd.read_csv(subfolders[i] + os.sep + files[0], sep = ',', header = 1)
            x = pos[bodypart][1:-1].values #not the first value because it might be 'x' or 'y'
            X = [float(j) * pix_correction for j in x]
            del x
            y = pos[bodypart+'.1'][1:-1].values
            Y = [float(j) * pix_correction for j in y]
            del y
            
            if i == folders[0]:
                print('Saving positions of 1st Session')
                XY1 = pd.DataFrame(np.array((X,Y)).T , columns = ['X' , 'Y'])
                XY1.to_csv(path + os.sep + 'positions_in_linear_track_1.csv', sep = ',', index = False)
            else:
                print('Saving positions of 2nd Session')
                XY2 = pd.DataFrame(np.array((X,Y)).T , columns = ['X' , 'Y'])
                XY2.to_csv(path + os.sep + 'positions_in_linear_track_2.csv', sep = ',', index = False)
    print('Done')
    return XY1, XY2
