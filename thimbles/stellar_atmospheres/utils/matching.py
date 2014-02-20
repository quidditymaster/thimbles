import numpy as np


def list_to_pairs(in_list):
    """ Used by get_matches to find pairs """
    pairs = []
    for i in xrange(len(in_list)):
        for j in xrange(i+1, len(in_list)):
            pairs.append(in_list[i],in_list[j])
    return pairs

def get_matches(data, tolerances, match_type = "max"):
    """
    Finds all the matching rows within the array data
    
    This uses dictionaries and rounding to create a lattice by with to find 
    matches. This algorithm works best for sparse sets where there are not 
    expected to be many matches. 
    
    Parameters
    ----------
    data : numpy.array with shape (N_observations, N_data_columns)
        This array has every observation as a row and each of the 
        parameters (which to match by) in the columns
    tolerances : numpy.array with shape (N_data_columns,)
        This array has the tolerances by with to match the parameter columns
    match_type : "max" or float
        if "max" the tolerances are interpreted as the maximum
        allowed differences between two rows in data to be counted as a match.

        if match_type is a floating point number it will be interpreted as the 
        order of the norm to use. e.g. if match_type is a number then two data rows 
        x and y will be considered a match if sum((x-y)**match_type) <= 1    
        
    Returns
    -------
    matches : list 
        This returns a list where all the values are tuples giving matches for
        indices (e.g. matches[0] = (i,j) where data[i] == data[j] within the 
        tolerances)
    
    Raises
    ------
    ValueError : If the tolerances.shape[0] != data.shape[1]
    
    
    Notes
    -----
    __1)__ This algorithm approaches O(N) if the matches have tolerances == 0
        because it uses the dictionary hashable type and approaches O(N**2) 
        if everything is found to be matches because it does a O(N**2) 
        operation to check all the possible matches
    __2)__ Matches are strictly less than in all dimensions
    
    Examples
    --------
    >>> data = np.array([[1.2, 2.3, 3.4],
                         [1.3, 4.3, 2.3],
                         [2.3, 2.3, 4.2],
                         [1.3, 2.1, 3.3]])
    >>> tolerances = np.array([0.11,0.3,0.11])
    >>> matches = get_matches(data,tolerances)
    >>> matches
    [(0, 3)]
    
    
    
    Modification History
    --------------------
    1, July 2013 : Tim Anderton
    20, July 2013 : Dylan Gregersen
        - modified doc string
    
    """
    _n_data, n_dim = data.shape
    
    n_dicts = 2**n_dim
    rounding_vecs = np.zeros((n_dicts, n_dim))

    for i in range(2**n_dim):
        binrep = bin(i)[2:]
        binrep = (n_dim-len(binrep))*"0" + binrep
        for j in range(n_dim):
            if binrep[j] == "1":
                rounding_vecs[i, j] = 1.0/3.0

    #dim_pairs = list_to_pairs(np.arange(n_dim))
    #n_dicts = len(dim_pairs) + n_dim + 1
    #for i in range(n_dim):
    #    rounding_vecs[i, i] = 1.0/3.0
    #for pidx in range(len(dim_pairs)):
    #    fidx, sidx = dim_pairs[pidx]
    #    rounding_vecs[pidx + n_dim, fidx] = 1.0/3.0
    #    rounding_vecs[pidx + n_dim, sidx] = 1.0/3.0
        
    match_dicts = [{} for i in range(n_dicts)]
    
    for data_idx in xrange(len(data)):
        for dict_idx in xrange(n_dicts):
            toround = data[data_idx]/(3.0*tolerances) + rounding_vecs[dict_idx]
            #toround = (data[data_idx]+rounding_vecs[dict_idx])/(3.0*tolerances)
            ctup = tuple(np.around(toround))
            match_dicts[dict_idx].setdefault(ctup,[]).append(data_idx)
         
    matches = set([])    
    for cdict in match_dicts:
        for value in cdict.values():
            if len(value) > 1:
                all_pairs = list_to_pairs(value)
                for pair in all_pairs:
                    matches.add(pair)
    
    #double check the matches
    normed_dat = data/(np.array(tolerances))
    checked_matches = []
    
    if match_type == "max":
        norm_func = lambda x: np.max(np.abs(x))
    else:
        match_type = float(match_type)
        norm_func = lambda x: np.sum(np.power(x, match_type))     
    
    for match in matches:
        fidx, sidx = match
        diff = normed_dat[fidx] - normed_dat[sidx]
        if norm_func(diff) <= 1.0:
            checked_matches.append(match)
    
    return checked_matches
