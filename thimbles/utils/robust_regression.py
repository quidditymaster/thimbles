import numpy as np
import scipy.sparse

def lqal_factory(
        sigma,
        crossover = 2.5,
        asymptotic_slope = 2.5**2,
):
    """
    makes a locally quadratic asymptoticaly linear weighting function
    """
    sigma_weights = 1.0/sigma**2
    def wfunc(resids):
        z = np.abs(resids/sigma)
        #abs_z = np.clip(np.abs(z), z_min, z_max)
        rel_weights = np.where(z < crossover, 1.0, 1.0+(z-crossover)/asymptotic_slope)
        weights = sigma_weights/rel_weights
        return weights
    return wfunc


def irls(
        design_matrix,
        data,
        weighting_function,
        start_x=None,
        regularization_matrix=None,
        regularization_weighter=None,
        regularity_target=None,
        n_iter=5,
        #resid_delta_trhesh=1e-8,
        #x_delta_thresh=1e-8,
        #convergence_criterion="both",
):
    m_issparse = scipy.sparse.issparse(design_matrix)
    
    if start_x is None:
        start_x = np.zeros((design_matrix.shape[1],))
    start_x = np.asarray(start_x)
    last_x = start_x
    
    if weighting_function is None:
        weighting_function = lqal_factory(1.0)
    
    if regularization_weighter is None:
        regularization_weighter = lqal_factory(1.0)   
    
    if regularity_target is None:
        if len(data.shape) == 2:
            regularity_target = np.zeros((regularization_matrix.shape[0], data.shape[1]))
        else:
            regularity_target = np.zeros((regularization_matrix.shape[0]))
    
    if m_issparse:
        solver = lambda x, y: scipy.sparse.linalg.bicg(x, y)[0]
        dotter = lambda x, y: x*y
    else:
        design_matrix = np.asarray(design_matrix)
        solver = np.linalg.solve
        dotter = lambda x, y: np.dot(x, y)
    
    if regularization_matrix is None:
        target = data
    else:
        target = np.hstack([data, regularity_target])
    
    for iter_idx in range(n_iter):
        last_resids = data - dotter(design_matrix, last_x)
        resid_weights = weighting_function(last_resids)
        if regularization_matrix is None:
            dmat = design_matrix
            diag_weights = scipy.sparse.dia_matrix((resid_weights, 0), (len(weights), len(weights)))
        else:
            dmat = scipy.sparse.vstack([design_matrix, regularization_matrix])
            #reg_vec =regularization_matrix
            reg_vec = regularization_matrix*last_x
            regularization_weights = regularization_weighter(reg_vec)
            stacked_weights = np.hstack([resid_weights, regularization_weights])
            diag_weights = scipy.sparse.dia_matrix((stacked_weights, 0), (len(stacked_weights), len(stacked_weights)))
        
        dtrans = dmat.transpose()
        weighted_dmat = diag_weights*dmat
        #weighted_resids = weights*last_resids
        
        dtd = dtrans*weighted_dmat
        dtarg = dtrans*(diag_weights*target)
        delta_x = scipy.sparse.linalg.bicg(dtd, dtarg)[0]
        #dtd = dotter(dtrans, weighted_dmat)
        #dtarg = dotter(dtrans, diag_weights*target)
        #delta_x = solver(dtd, dtarg)
        last_x = delta_x
        #last_x = last_x + delta_x
    
    fit_dict = dict(
        data_weights=resid_weights,
        regularization_weights=regularization_weights,
        x=last_x,
    )
    
    return fit_dict


def low_rank_decomposition(
        data,
        rank,
        residual_weighter,
        row_damper,
        col_damper,
        n_iter = 5,
        init="pca",
):
    n_row, n_col = data.shape
    if init == "pca":
        print("carrying out pca")
        u, s, v = np.linalg.svd(data, full_matrices=False)
        init = (u[:, :rank]*s[:rank], v[:rank].transpose())
    
    row_coeffs, col_coeffs = init
    
    for iter_idx in range(n_iter):
        print("iter {}".format(iter_idx+1))
        for rank_idx in range(rank):
            model = np.dot(row_coeffs, col_coeffs.transpose())
            full_resids = data - model
            resid_weights = residual_weighter(full_resids)
            
            crow_vec = row_coeffs[:, rank_idx]
            ccol_vec = col_coeffs[:, rank_idx]
            
            #adjust row coefficients
            ccol_tres = ccol_vec.reshape((1, -1))*full_resids
            numerator = np.sum(ccol_tres*resid_weights, axis=1)
            row_damp = row_damper(crow_vec)
            denominator = np.sum((ccol_vec**2).reshape((1, -1))*resid_weights, axis=1) + row_damp
            row_delta = numerator/denominator
            row_coeffs[:, rank_idx] += row_delta

            #adjust column coefficients
            model = np.dot(row_coeffs, col_coeffs.transpose())
            full_resids = data - model
            resid_weights = residual_weighter(full_resids)
            crow_tres = full_resids*row_coeffs[:, rank_idx].reshape((-1, 1))
            numerator = np.sum(crow_tres*resid_weights, axis=0)
            col_damp = col_damper(ccol_vec)
            denominator = np.sum((row_coeffs[:, rank_idx]**2).reshape((-1, 1))*resid_weights, axis=0) + col_damp
            col_delta = numerator/denominator
            col_coeffs[:, rank_idx] += col_delta
    
    return row_coeffs, col_coeffs
