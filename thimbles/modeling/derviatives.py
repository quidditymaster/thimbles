import numpy as np
import scipy.sparse as sparse

from .paths import execute_path, extract_influence_graph, find_all_paths


class MaxDerivativeComplexityExceeded(Exception):
    pass

class BrokenDerivativeChain(Exception):
    pass


def deriv(response_params, input_params, combine_matrices=True):
    nrow = len(response_params)
    ncol = len(input_params)
    blocks = [[None for j in range(nrow)] for i in range(ncol)]
    
    influence_graph = extract_influence_graph(response_params, direction="upstream")
    for rp_idx, rp in enumerate(response_params):
        for ip_idx, ip in enumerate(input_params):
            connecting_paths = find_all_paths(ip, rp, influence_graph=influence_graph)
            for path in connecting_paths:
                cder_mat = path_deriv(path)
                prev_der = blocks[rp_idx][ip_idx]
                if prev_der is None:
                    blocks[rp_idx][ip_idx] = cder_mat
                else:
                    blocks[rp_idx][ip_idx] = prev_der + cder_mat
    if combine_matrices:
        dermat = scipy.sparse.bmat(blocks)
    else:
        dermat = blocks
    return dermat


def path_deriv(path, max_discrete_n=50, fallback_epsilon=1e-4):
    assert len(path) >= 3
    assert len(path) % 2 == 1
    params = path[::2]
    mods = path[1::2]
    
    assert all([isinstance(p, Parameter) for p in params])
    assert all([isinstance(m, Model) for m in mods])
    #try and use the fast_deriv model methods
    try:
        tot_deriv = mods[0].fast_deriv(params[0])
        if tot_deriv is None:
            raise BrokenDerivativeChain()
        for mod_idx in range(1, len(mods)):
            cur_deriv = mods[mod_idx].fast_deriv(params[mod_idx])
            if cur_deriv is None:
                raise BrokenDerivativeChain()
            tot_deriv = cur_deriv*tot_deriv
        return tot_deriv
    except BrokenDerivativeChain:
        cur_input_val = path[0].value
        
        central_input_val = np.atleast_1d(cur_input_val)
        ndim_input = len(cur_input_val)
        if ndim_input > max_discrete_n:
            raise MaxDerivativeComplexityExceeded()
        
        der_cols = []
        for dim_idx in range(ndim_input):
            delta_vec = central_input_val.copy()

            delta_vec[dim_idx] = central_input_val[dim_idx]+fallback_epsilon
            plus_d = execute_path(path, root_value=delta_vec)

            delta_vec[dim_idx] = central_input_val[dim_idx]-fallback_epsilon
            neg_d  = execute_path(path, root_value=delta_vec)
            
            dcol = (plus_d-neg_d)/(2.0*fallback_epsilon)
            der_cols.append(scipy.sparse.csc_matrix(deriv.reshape((-1, 1))))
        pder_mat = scipy.sparse.hstack(der_cols)
        return pder_mat

