import numpy as np
import scipy
from scipy import ndimage
import scipy.sparse
import matplotlib.pyplot as plt

def running_ccorr(
        arr1, 
        arr2, 
        avg1=None, 
        avg2=None, 
        lags=10, 
        kernel_width=20,
        order=0,
        edge_mode="reflect",
):
    """calculate a running cross correlation between two vectors.
    """
    npts = len(arr1)
    assert npts == len(arr2)
    
    if hasattr(avg1, "__call__"):
        avg1 = avg1(arr1)
    if hasattr(avg2, "__call__"):
        avg2 = avg2(arr2)
    
    if not (avg1 is None):
        arr1 = arr1 - avg1
    if not (avg2 is None):
        arr2 = arr2-avg2
    
    if isinstance(lags, int):
        lags = np.arange(-lags, lags+1)
    lags = np.asarray(lags, dtype=int)
    
    n_lags = len(lags)
    corr_out = np.zeros((n_lags, npts))
    
    cc_filter = lambda x: ndimage.filters.gaussian_filter(x, sigma=kernel_width, mode=edge_mode, order=order)
    
    for lag_idx in range(n_lags):
        clag = lags[lag_idx]
        if clag == 0:
            acorr_out[lag_idx] = cc_filter(arr1*arr2)
        else:
            ccor = cc_filter(arr1[clag:]*arr2[:-clag])
            lb = clag//2
            ub = npts - clag//2
            if clag % 2 == 1:
                if clag % 4 == 1:
                    ub += 1
                else:
                    lb -= 1
            acorr_out[lag_idx, lb:ub] = ccor
    return acorr_out


def running_acorr(arr, avg=None, weights=None, window_sigma=5, ncorr=21, mode="reflect"):
    """
    generate a running auto-correlation estimate.
    
    parameters
    arr: numpy ndarray
      the data array to be autocorrelated  
    avg: function, numpy array, float or None
      The running mean of the data in arr.
      if avg is a callable it will be called on arr and the result treated as 
      the running mean. If avg is None the median value of arr will be used.
    window_sigma: float
      the width of the gaussian filter to use to obtain a local estimate
      of the correlation E[(arr-avg)_i*(arr-avg)_i+lag]
    """
    gauss_filter = ndimage.filters.gaussian_filter
    assert ncorr >= 1
    npts = len(arr)
    if avg is None:
        avg = np.median(arr)
    if hasattr(avg, "__call__"):
        avg = avg(arr)
    diff = arr - avg
    if weights is None:
        weights = np.ones(npts, dtype=float)
    acorr_out = np.zeros((npts, 2*ncorr+1))
    
    #make central weighted correlation estimate
    weighted_corr = gauss_filter(weights*diff**2, sigma=window_sigma, mode=mode)
    weight_correction = gauss_filter(weights, sigma=window_sigma, mode=mode)
    weight_correction = np.where(weight_correction > 0, weight_correction, 1.0)
    acorr_out[:, ncorr] = weighted_corr/weight_correction
    for i in range(1, ncorr+1):
        #geometric mean 1/(1/w1 + 1/w2) == w1*w2/(w1+w2)
        avg_weights = (weights[i:]*weights[:-i])/(weights[i:] + weights[:-i])
        weighted_diff_prod = gauss_filter(diff[i:]*diff[:-i]*avg_weights, sigma=window_sigma, mode=mode)
        weight_correction = gauss_filter(avg_weights, sigma=window_sigma, mode=mode)
        weight_correction = np.where(weight_correction > 0, weight_correction, 1.0)
        diff_prod = weighted_diff_prod/weight_correction
        
        lb_out = i//2
        ub_out = npts-i//2
        if i % 2 == 1:
            if (i-1) % 4 == 1:
                acorr_out[lb_out:ub_out-1, ncorr-i] = diff_prod
                acorr_out[lb_out+1:ub_out, ncorr+i] = diff_prod
            else:
                acorr_out[lb_out+1:ub_out, ncorr-i] = diff_prod 
                acorr_out[lb_out:ub_out-1, ncorr+i] = diff_prod
        else:
            acorr_out[lb_out:ub_out, ncorr-i] = diff_prod
            acorr_out[lb_out:ub_out, ncorr+i] = diff_prod
    return acorr_out


def extract_banded_part(A, band_width):
    k = band_width
    A = scipy.sparse.dia_matrix(A)
    offsets = A.offsets
    off_mask = (offsets >= -k)*(offsets <= k)
    clip_data = A.data[off_mask].copy()
    clip_offs = offsets[off_mask]
    return scipy.sparse.dia_matrix((clip_data, clip_offs), A.shape) 


def sparse_banded_approximate_inverse(A, extract_bands=True, band_width=None):
    """
    generate a sparse banded approximation to the inverse of an input banded sparse matrix.
    
    The method is based on the formulas presented in 
    'Approximating the inverse of banded matrices by banded matrices with applications to probability and statistics' Bickel and Linder 2010
    
    see equation 6
    
    which in turn is an application of Neumann series
    """
    npts = A.shape[0]
    assert A.shape[0] == A.shape[1]
    A = scipy.sparse.dia_matrix(A)
    if band_width is None:
        band_width = np.max(np.abs(A.offsets))
    
    #import pdb; pdb.set_trace()
    
    if extract_bands:
        A = extract_banded_part(A, band_width=band_width)
    
    spectral_radii = []
    for i in range(20):
        xrand = np.random.normal(size=npts)
        xrand /= np.sqrt(np.sum(xrand**2))
        yvec = A*xrand
        ynorm = np.sqrt(np.sum(yvec**2))
        spectral_radii.append(ynorm)
    gamma = np.median(spectral_radii)
    print("gamma", gamma)
    
    #rescale the matrix so that the effective gamma is 0.5
    gamma_eff = 0.5
    rescale_factor = gamma_eff/gamma
    
    A = rescale_factor*A
    
    power_mat = scipy.sparse.identity(A.shape[0]) - gamma_eff*A
    cur_inv_approx = power_mat
    last_power = scipy.sparse.identity(A.shape[0])
    for power_idx in range(1, band_width+1):
        last_power = last_power*power_mat
        cur_inv_approx = cur_inv_approx + last_power
    cur_inv_approx *= gamma_eff
    
    cur_inv_approx = rescale_factor*cur_inv_approx
    
    return cur_inv_approx


#@task(result_name="noise_estimate")
def estimate_spectrum_noise(
        spectrum,
        smoothing_scale=3.0,
        median_scale=200,
        apply_poisson=True,
        post_smooth=5.0,
        max_snr=1000.0,
        overwrite_noise=False,
):
    sm_var = smoothed_mad_error(spectrum.flux, smoothing_scale=smoothing_scale, median_scale=median_scale, apply_poisson=apply_poisson, post_smooth=post_smooth, max_snr=max_snr)
    if overwrite_noise:
        spectrum.var = sm_var
