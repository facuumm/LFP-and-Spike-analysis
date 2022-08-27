from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

def get_matching_pairs(template, to_be_matched, scaler=False):
    """
    Generates 1:1 pair matching and sampling the to_be_matched array using the
    template array as a guide. It use NearestNeighbors from module sklearn to
    perform Unsupervised learner for implementing neighbor searches.
    
    * I used for matching 1-sec bins of velocity data from different behavioral
    conditions. But I guess it could be used for other stuffs as well.

    Parameters
    ----------
    template: Array
        Stores the values to guide the neighbor searching

    to_be_matched: Array
        Stores the values to be matched      
        
    scaler: bool
        It defins if the values are going to be normalized to mean and SD

    Returns
    -------
    matched: Array float64.
        Contains the matched values from to_be_matched input.
        
    matched: Array float64.
        Contains the indices of the matched values from to_be_matched array.
        
   Created on Tue Jul 25 12:24:35 2022

   @author: facundo.morici
        
    """
    template = template.values
    to_be_matched = to_be_matched.values
    
    template = template.reshape(-1,1)
    to_be_matched = to_be_matched.reshape(-1,1)
    
    if scaler == True:
        scaler = StandardScaler()

    if scaler:

        scaler.fit(template)
        template = scaler.transform(template)
        to_be_matched = scaler.transform(to_be_matched)

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(to_be_matched)
    distances, indices = nbrs.kneighbors(template)
    indices = indices.reshape(indices.shape[0])
    matched = to_be_matched[indices]
    return matched, indices
