import numpy as np
import thimbles as tmb
import scipy.sparse
from piecewise_polynomial import powers_from_max_order
from piecewise_polynomial import MultiVariatePolynomial


def optimal_partition(data, cost_fn, minimize=True):
    """divides up data attempting to minimize(maximize) the sum of the 
    provided cost function over partition blocks.
    data should be a data structure that includes everything needed by 
    cost_fn to calculate the cost of a block.
    
    a list of separating indicies is returned in which the first index
    is included in its block. That is 
    data[indicies[0]:indicies[1]] gives the data of the first block, and
    data[indicies[-2]:indicies[-1]] gives the data of the last block.
    
    references: 
    Jackson & Scargle 2003 
    'An Algorithm for Optimal Partitioning of Data on an Interval'
    """
    sign = 1.0
    if minimize==False:
        sign = -1.0
    #import pdb; pdb.set_trace()
    npts = len(data)
    #an array for the optimal values for less than that index
    opt_val = np.zeros(npts+1)
    break_idxs = np.zeros(npts+1)
    for i in range(1, npts+1):
        cmin = float("inf")
        cidx = 0
        for j in range(i):
            tot_cost = sign*cost_fn(data[j:i]) + opt_val[j]
            if tot_cost < cmin:
                cmin = tot_cost
                cidx = j
        opt_val[i] = cmin
        break_idxs[i] = cidx
    partition = [break_idxs[-1]]
    while partition[-1] > 0:
        partition.append(break_idxs[partition[-1]-1])
    partition.reverse()
    partition.append(npts)
    return partition

def partitioned_polynomial_model(xvec, 
                                yvec, 
                                y_inv_var, 
                                poly_order,
                                grouping_column = 0, 
                                min_delta = 0,
                                max_delta = np.inf,
                                alpha=2.0, 
                                beta=2.0, 
                                beta_epsilon=0.01, 
                                gamma=2.5, 
                                y_mult=None):
    """
    fits optimal multivariate piecewise polynomials to the data with 
    dynamically chosen break points between the polynomials.
    The optimal partition is chosen so that the sum of the cost functions
    over the blocks in the partition is a minimum.
    
    cost = pseudo_huber_cost(residuals) + alpha*k + beta*k*(k+1)/max(beta_epsilon, npts_region - k -1)
    
        npts_region is the number of points with non-zero y errors included
        in the current partition block. 
        k is the number of terms used in the fit e.g. for a one dimensional
        piecewise quadratic k = poly_order + 1 = 3.
        the pseudo huber residual cost is a chi squared sum for small deltas
        and turns over to linear cost for large delta with the turnover
        controled by the input gamma parameter.
    
    parameters
    xvec: numpy.ndarray (n_data, n_variables)
        the independent variables
    yvec: numpy.ndarray (n_data,)
        the dependent variables
    y_inv_var: npumpy.ndarray (n_data,) 
        one over the variance associated to yvec
    poly_order: int or tuple 
        a tuple of the maximum order of the associated column
        of any term which contains a non-zero power of that variable. 
        for example if poly_order == (2, 1) call the first 
        variable(first column) x and the second variable(second column)
        y then we would allow the terms 1, x, x**2, y
        and if poly_order == (2, 2) we would allow 1, x, x**2, y, y**2, x*y
        x**2*y and y**2*x**2 etc, would not be allowed because their total
        order exceeds 2.
    grouping_column: int 
        the index of the column to be used to divide the data
        rows into blocks. Defaults to first column
    min_delta: float 
        The minimum difference in the values of the grouping column
        over which to allow a break. Row indicies are accumulated into 
        the elementary blocks from which the larger partitions are built
        until the difference in the grouping column from start to end 
        exceeds min_delta.
    max_delta: float
        the maximum difference in the values of the grouping column
        to allow before forcing a break.
    alpha: float
        The cost per extra term 2 corresponds to the AIC 
        log(npts_total) corresponds to using the BIC
    beta: float
        Cost multiplier to correct for small sample sizes, making picking samples
        which are not well constrained expensive.
    beta_epsilon: float
        beta*k*(k+1)/beta_epsilon is the cost associated to all
        not well constrained models (meaning npts_block <= (k + 1))
    gamma: float
        controls the turn over to linear part of pseudo-huber cost function
    y_mult: None or ndarray
        optional constant vector to multiply the polynomial by before comparing
        to the target y data.
    
    returns:
    partition: list of integers
     the indexes of rows of x representing the optimal partitioning.
    partition polys: list of MultiVariatePolynomial objects for each partition
    
    see also: partitioning.optimal_partition
    """
    assert len(xvec) == len(yvec)
    npts = len(xvec)
    assert len(y_inv_var) == npts
    
    if y_mult is None:
        y_mult = np.ones(npts)
    
    #keep a reshaped xvec so it is 2 dim
    if xvec.ndim == 2:
        xvec_2d = xvec
    elif xvec.ndim == 1:
        xvec_2d = xvec.reshape((-1, 1))
    else:
        raise Exception("can't handle input dimensions higher than 2!")
    n_cols = xvec_2d.shape[1]
    
    #create the elementary blocks
    gcol = xvec_2d[:, grouping_column]
    grouping_idxs = [0]
    last_g_val = gcol[0]
    for x_idx in range(1, npts):
        if np.abs(gcol[x_idx]-last_g_val) > min_delta:
            grouping_idxs.append(x_idx)
            last_g_val = gcol[x_idx]
    grouping_idxs.append(npts)
    
    #determine the center and scale of the x coordinates
    x_cent = np.mean(xvec_2d, axis = 0)
    x_scale = np.std(xvec_2d, axis = 0)
    
    #build up the polynomial terms to be used
    powers = powers_from_max_order(poly_order)
    n_terms = len(powers)
    
    #make an array of the monomial terms times the output modulation
    mvp = MultiVariatePolynomial(np.zeros(n_terms), powers, x_cent, x_scale)
    pofx = mvp.get_pofx(xvec_2d)*y_mult.reshape((-1, 1))
    
    #carry out the optimal partitioning algorithm.
    n_blocks = len(grouping_idxs)-1
    opt_val = np.zeros(n_blocks+1)
    break_idxs = np.zeros(n_blocks+1, dtype = int)
    opt_fit_params = np.zeros((n_blocks+1, n_terms))
    sigma_vec = 1.0/np.sqrt(y_inv_var)
    sigma_vec = np.where(y_inv_var == 0, 1e10, sigma_vec)
    gamma_vec = gamma*sigma_vec
    for i in range(1, n_blocks+1):
        cmin = float("inf")
        cidx = None
        last_opt_params = None
        for j in range(i):
            lb = grouping_idxs[j]
            ub = grouping_idxs[i]
            if np.abs(xvec_2d[ub-1, grouping_column]-xvec_2d[lb, grouping_column]) > max_delta:
                continue
            chop_sig=sigma_vec[lb:ub]
            chop_gam=gamma_vec[lb:ub]
            opt_params = tmb.utils.misc.pseudo_huber_irls(pofx[lb:ub], yvec[lb:ub], chop_sig, chop_gam)
            opt_fit_y = np.dot(pofx[lb:ub], opt_params)
            resids = opt_fit_y-yvec[lb:ub]
            #chi_sq = np.sum((opt_fit_y-yvec[lb:ub])**2*y_inv_var[lb:ub])
            ph_cost = tmb.utils.misc.pseudo_huber_cost(resids, chop_sig, chop_gam)
            n_local = ub-lb
            param_cost = alpha*n_terms
            param_cost += beta*n_terms*(n_terms+1)/max(beta_epsilon, n_local-n_terms-1)
            tot_cost = ph_cost + param_cost + opt_val[j]
            if tot_cost < cmin:
                cmin = tot_cost
                cidx = j
                last_opt_params = opt_params
        opt_val[i] = cmin
        break_idxs[i] = cidx
        opt_fit_params[i] = last_opt_params
    first_break = break_idxs[-1]
    partition = [first_break]
    while partition[-1] > 0:
        cbreak = break_idxs[partition[-1]-1]
        partition.append(cbreak)
    partition.insert(0, n_blocks)
    partition_mvps = []
    for pidx in range(len(partition)-1):
        coeffs = opt_fit_params[partition[pidx]]
        cmvp = MultiVariatePolynomial(coeffs, powers, x_cent, x_scale)
        partition_mvps.append(cmvp)
    orig_space_partition = []
    for part_idx in partition:
        orig_space_partition.append(grouping_idxs[part_idx])
    orig_space_partition.reverse()
    partition_mvps.reverse()
    return orig_space_partition, partition_mvps

    
def matrix_partition(
                     model_matrix, 
                     y,
                     y_weight, 
                     partitioning_ordinal=None,
                     min_delta = None,
                     max_delta = np.inf,
                     alpha=2.0, 
                     beta=2.0, 
                     beta_epsilon=0.01, 
                     gamma=2.5, 
                     ):
        npts = len(y)
        assert len(y_weight) == npts
        assert len(model_matrix)== npts
        #is_sorted = np.all(np.sort(partitioning_ordinal) == partitioning_ordinal)
        #if not is_sorted:
        #    raise ValueError("partitioning_ordinal must be sorted")
        
        if partitioning_ordinal is None:
            partitioning_ordinal = np.arange(npts)
        
        if min_delta is None:
            med_ord = np.median(partitioning_ordinal)
            min_delta = np.median(np.abs(partitioning_ordinal-med_ord))
            if min_delta == 0:
                min_delta = np.std(partitioning_ordinal)
            min_delta /= 10.0
        
        #create the elementary blocks        
        grouping_idxs = [0]
        last_g_val = partitioning_ordinal[0]
        for x_idx in range(1, npts):
            if np.abs(partitioning_ordinal[x_idx]-last_g_val) > min_delta:
                grouping_idxs.append(x_idx)
                last_g_val = partitioning_ordinal[x_idx]
        grouping_idxs.append(npts)
        
        n_cols = model_matrix.shape[1]     
        #carry out the optimal partitioning algorithm.
        n_blocks = len(grouping_idxs)-1
        opt_val = np.zeros(n_blocks+1)
        break_idxs = np.zeros(n_blocks+1, dtype = int)
        opt_fit_params = np.zeros((n_blocks+1, n_cols))
        sigma_vec = 1.0/np.sqrt(np.where(y_weight == 0, 1e-20, y_weight))
        gamma_vec = gamma*sigma_vec
        for i in range(1, n_blocks+1):
            print "optimizing block {} of {}".format(i, n_blocks)
            cmin = float("inf")
            cidx = None
            last_opt_params = None
            for j in range(i):
                lb = grouping_idxs[j] #j is the index of the lower block
                ub = grouping_idxs[i] #i is the index of the upper block
                if np.abs(partitioning_ordinal[ub-1]-partitioning_ordinal[lb]) > max_delta:
                    continue
                chop_sig=sigma_vec[lb:ub]
                chop_gam=gamma_vec[lb:ub]
                opt_params = tmb.utils.misc.pseudo_huber_irls(model_matrix[lb:ub], y[lb:ub], chop_sig, chop_gam)
                opt_fit_y = np.dot(model_matrix[lb:ub], opt_params)
                resids = opt_fit_y-y[lb:ub]
                #chi_sq = np.sum((opt_fit_y-yvec[lb:ub])**2*y_inv_var[lb:ub])
                ph_cost = tmb.utils.misc.pseudo_huber_cost(resids, chop_sig, chop_gam)
                n_local = ub-lb
                param_cost = alpha*n_cols
                param_cost += beta*n_cols*(n_cols+1)/max(beta_epsilon, n_local-n_cols-1)
                tot_cost = ph_cost + param_cost + opt_val[j]
                if tot_cost < cmin:
                    cmin = tot_cost
                    cidx = j
                    last_opt_params = opt_params
            opt_val[i] = cmin
            break_idxs[i] = cidx
            opt_fit_params[i] = last_opt_params
        first_break = break_idxs[-1]
        partition = [first_break]
        while partition[-1] > 0:
            cbreak = break_idxs[partition[-1]-1]
            partition.append(cbreak)
        partition.insert(0, n_blocks)
        partition_coeffs = []
        for pidx in range(len(partition)-1):
            coeffs = opt_fit_params[partition[pidx]]
            partition_coeffs.append(coeffs)
        partition_coeffs.reverse()
        orig_space_partition = []
        for part_idx in partition:
            orig_space_partition.append(grouping_idxs[part_idx])
        orig_space_partition.reverse()
        orig_space_partition = np.array(orig_space_partition)
        partition_coeffs = np.array(partition_coeffs)
        return orig_space_partition, partition_coeffs, opt_val


def block_constant_partition( 
                     y,
                     y_weight, 
                     partitioning_ordinal=None,
                     transform_matrix=None, 
                     min_delta = None,
                     max_delta = np.inf,
                     alpha=2.0, 
                     beta=2.0, 
                     gamma=2.5, 
                     ):
        npts = len(y)
        assert len(y_weight) == npts
        if transform_matrix is None:
            transform_matrix = scipy.sparse.identity(npts, dtype=float, format="csr")
        #assert len(model_matrix)== npts
        #is_sorted = np.all(np.sort(partitioning_ordinal) == partitioning_ordinal)
        #if not is_sorted:
        #    raise ValueError("partitioning_ordinal must be sorted")
        
        if partitioning_ordinal is None:
            partitioning_ordinal = np.arange(npts)
        
        if min_delta is None:
            med_ord = np.median(partitioning_ordinal)
            min_delta = np.median(np.abs(partitioning_ordinal-med_ord))
            if min_delta == 0:
                min_delta = np.std(partitioning_ordinal)
            min_delta /= 10.0
        
        #create the elementary blocks        
        grouping_idxs = [0]
        last_g_val = partitioning_ordinal[0]
        for x_idx in range(1, npts):
            if np.abs(partitioning_ordinal[x_idx]-last_g_val) > min_delta:
                grouping_idxs.append(x_idx)
                last_g_val = partitioning_ordinal[x_idx]
        grouping_idxs.append(npts)
        
        #carry out the optimal partitioning algorithm.
        n_blocks = len(grouping_idxs)-1
        opt_val = np.zeros(n_blocks+1)
        break_idxs = np.zeros(n_blocks+1, dtype = int)
        opt_fit_params = np.zeros((n_blocks+1,))
        sigma_vec = 1.0/np.sqrt(np.where(y_weight == 0, 1e-20, y_weight))
        gamma_vec = gamma*sigma_vec
        for i in range(1, n_blocks+1):
            print "optimizing block {} of {}".format(i, n_blocks)
            cmin = float("inf")
            cidx = None
            last_opt_params = None
            for j in range(i):
                lb = grouping_idxs[j] #j is the index of the lower block
                ub = grouping_idxs[i] #i is the index of the upper block
                if np.abs(partitioning_ordinal[ub-1]-partitioning_ordinal[lb]) > max_delta:
                    continue
                chop_sig=sigma_vec[lb:ub]
                chop_gam=gamma_vec[lb:ub]
                tmat = transform_matrix[lb:ub]
                collapsed_vec = tmat*np.ones(len(tmat))
                opt_params = tmb.utils.misc.pseudo_huber_irls(collapsed_vec.reshape((-1, 1)), y[lb:ub], chop_sig, chop_gam)
                opt_fit_y = collapsed_vec*opt_params
                resids = opt_fit_y-y[lb:ub]
                ph_cost = tmb.utils.misc.pseudo_huber_cost(resids, chop_sig, chop_gam)
                n_local = ub-lb
                param_cost = alpha
                param_cost += beta/float(n_local)
                tot_cost = ph_cost + param_cost + opt_val[j]
                if tot_cost < cmin:
                    cmin = tot_cost
                    cidx = j
                    last_opt_params = opt_params
            opt_val[i] = cmin
            break_idxs[i] = cidx
            opt_fit_params[i] = last_opt_params
        first_break = break_idxs[-1]
        partition = [first_break]
        while partition[-1] > 0:
            cbreak = break_idxs[partition[-1]-1]
            partition.append(cbreak)
        partition.insert(0, n_blocks)
        orig_space_partition = []
        for part_idx in partition:
            orig_space_partition.append(grouping_idxs[part_idx])
        orig_space_partition.reverse()
        return orig_space_partition, opt_val

def partition_average(data, partition):
    avg = []
    for part_idx in range(len(partition)-1):
        lb = partition[part_idx]
        ub = partition[part_idx+1]
        avg.append(np.mean(data[lb:ub]))
    return avg

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    def chi_sq(xvec):
        return np.sum((xvec-np.mean(xvec))**2)
    
    #Aikaike Information Criterion
    def AIC_cost_fn(xvec):
        return chi_sq(xvec) + 2
    
    def STN_cost_fn(xvec):
        return chi_sq(xvec) + 2.0/len(xvec)
    
    def AICc_cost_fn(xvec):
        return chi_sq(xvec) + 2.0 + 2.0/(len(xvec)-0.9)
    
    def MAD_cost_fn(xvec):
        med = np.median(xvec)
        return np.median(np.abs(xvec-med))/0.6457 + 1.0
    
    #testpoints = np.array([1, 1, 1, 3, 3], dtype = float)
    #part = optimal_partition(testpoints, AIC_cost_fn)
    
    n_samples = 400
    n_ord = 10
    
    n_bad_avg = 20
    bad_width = 20
    
    def ord_func(pixnum, ordnum):
        return 3*np.sin(0.01*pixnum) + 3.0*ordnum + ((ordnum-n_samples/2.0)/float(n_samples))**2*0.1*(ordnum)**2
    
    noise_level = 1.0
    noise = noise_level*np.random.normal(size = (n_ord*n_samples,))
    
    pix_in = np.arange(n_samples)
    
    #x coords first column pixels second column order
    x_in = np.zeros((n_samples*n_ord, 2))
    for ord_idx in range(n_ord):
        x_in[ord_idx::n_ord, 0] = pix_in
        x_in[ord_idx::n_ord, 1] = ord_idx
    
    good_mask = np.ones((n_ord*n_samples,), dtype=bool)
    for ord_idx in range(n_ord):
        c_num_bad = np.random.poisson(n_bad_avg)
        for bad_idx in range(c_num_bad):
            excl_lb = int(np.random.random()*n_samples)
            excl_ub = excl_lb + bad_width
            good_mask[ord_idx::n_ord][excl_lb:excl_ub] = False 

    true_points = ord_func(x_in[:, 0], x_in[:, 1]) 
    rpoints = true_points + noise
    
    in_inv_var = noise_level*np.ones(rpoints.shape)

    x_in_masked = x_in[good_mask]
    rpoints_masked = rpoints[good_mask]
    in_inv_var_masked = in_inv_var[good_mask]
    
    import time
    stime = time.time()    
    part, part_polys = partitioned_polynomial_model(x_in_masked, rpoints_masked, 
                                    in_inv_var_masked, poly_order=(2,2),
                                    grouping_column = 0, min_delta = 5, max_delta=50,
                                    alpha=2.0, beta=2.0, beta_epsilon=0.01)
    etime = time.time()
    print "partitioned in {} seconds".format(etime-stime)
    
    #part = optimal_partition(rpoints, cost_fn)
    #avgs = partition_average(rpoints, part)
    xval = x_in[good_mask]
    for ord_idx in range(n_ord):
        ord_mask = xval[:, 1] == ord_idx
        plt.plot(x_in[ord_idx::n_ord, 0], true_points[ord_idx::n_ord], "k")#, label="true")
        plt.scatter(x_in_masked[ord_mask, 0], rpoints_masked[ord_mask], c="g")#, label="observed")
        for i in range(len(part_polys)):
            print "part poly idx", i
            lb = part[i]
            ub = part[i+1]
            x_idxs = np.arange(len(rpoints_masked))
            bound_mask = (x_idxs >= lb)*(x_idxs < ub)
            mask = bound_mask*ord_mask
            if len(mask) > 0:
                plt.plot(x_in_masked[mask, 0], part_polys[i](x_in_masked[mask]) , c="r")
        plt.xlabel("pixel number")
        plt.ylabel("relative flux")
    plt.show()
