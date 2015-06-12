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
        base_offset=0,
        lags=10, 
        window_sigma=10, 
        mode="reflect",
):
    """calculate a running cross correlation between two
    """
    
    if hasattr(avg1, "__call__"):
        avg1 = avg1(arr1)
    if hasattr(avg2, "__call__"):
        avg2 = avg2(arr2)
    if not (avg1 is None):
        arr1 = arr1 - avg1
    if not (avg2 is None):
        arr2 = arr2-avg2
    
    if isinstance(lags, int):
        lags = np.arange(lags+1)
    lags = np.asarray(lags, dtype=int)
    
    n_lags = len(lags)
    npts_out = min([len(arr1), len(arr2)]) + n_lags
    corr_out = np.zeros((npts_out, n_lags))
    offsets = lags + base_offset
    
    
    

def running_acorr(arr, avg=None, window_sigma=5, ncorr=21, mode="reflect"):
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
    acorr_out = np.zeros((npts, 2*ncorr+1))
    acorr_out[:, ncorr] = gauss_filter(diff**2, sigma=window_sigma, mode=mode)
    for i in range(1, ncorr+1):
        diff_prod = gauss_filter(diff[i:]*diff[:-i], sigma=window_sigma, mode=mode)
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
