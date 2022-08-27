def selecting_events_TS(events,list1 = ['aversive' , 'reward']):
    """
    This function select the consecutive lines from the events input and just
    keep those that are part of list1. 

    Parameters
    ----------
    events: pd.DataFrame
        First column should contains the timestamp.
        Second column should contains the condition label (eg: 'aversive').
        Third column should contains the time label ('Start' and 'End').
        Following there is an example of this input
        
                -------------------------------
                0.0          sleep_1   Start
                8933619.2    sleep_1    End
                8933619.2    aversive   Start
                9848582.4    aversive   End
                9848582.4    sleep_2    Start
                18670649.6   sleep_2    End
                18670649.6   reward     Start
                19597292.8   reward     End
                19597292.8   sleep_3    Start
                28433024.0   sleep_3    End
                ------------------------------
        
        
    list1: list
        List that store the label of conditions to keep.
        example: ['aversive' , 'reward']


    Returns
    -------
    output: dict.
        Is a dictioray nested into another one. It use the labels inside list1
        to save the 'Start' and 'End' of each. Exalmple:
            
            {'aversive': {'Start': 8933619.2, 'End': 9848582.4},
             'reward': {'Start': 18670649.6, 'End': 19597292.8}}


    Created on Sat Jul 23 18:17:16 2022

    @author: facundo.morici
    """
    
    import numpy as np
    size1 = events.shape
    
    events = events.to_numpy()
    
    output = {}
    for index in np.arange(0,size1[0]-1,2):
        event_label = events[index , 1]
        event_start = events[index , 0]
        if events[index , 1] == events[index+1 , 1]:
            event_end = events[index+1 , 0]
        else:
            raise ValueError('Check the shape of your Events input. It might be missing one value')
            
        output[event_label] = {'Start' : event_start , 'End' : event_end}
        
        
    output_filtered = {k: output[k] for k in list1}
    
    return output_filtered