#
# ########################################################################### #
# Standard Library
import os
from copy import deepcopy

# 3rd Party
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology
import scipy.fftpack as fftpack
import scipy.sparse
import scipy.sparse.linalg
from scipy.interpolate import interp1d
import numpy as np
from astropy import units
from astropy import coordinates
import astropy.io.fits as fits
try:
    import cvxopt
    import cvxopt.solvers
    with_cvxopt = True
except ImportError:
    with_cvxopt = False


# Internal
import thimbles
from thimbles import hydrogen
from . import resampling
from . import partitioning
from . import piecewise_polynomial

# ########################################################################### #

def AngleRA (angle,unit=units.hourangle,raise_errors=False):
    """
    An object which represents a right ascension angle
    
    see `astropy.coordinates.RA` for more extensive documentation

    The primary difference with astropy is that if the call to coordinates.RA
    errors you have the option to ignore it and return None for the angle
    i.e. when raise_errors=False
    
    """     
    try:
        result = coordinates.RA(angle,unit=unit)
    except Exception as e:
        result = None
        if raise_errors:
            raise e
    return result
     
def AngleDec (angle,unit=units.degree,raise_errors=False):
    """
    An object which represents a declination angle
    
    see `astropy.coordinates.Dec` for more extensive documentation
    
    The primary difference with astropy is that if the call to coordinates.RA
    errors you have the option to ignore it and return None for the angle
    i.e. when raise_errors=False
    
    """ 
    try:
        result = coordinates.Dec(angle,unit=unit)
    except Exception as e:
        result = None
        if raise_errors:
            raise e
    return result

def invert(arr):
    return np.where(arr > 0, 1.0/(arr+(arr==0)), np.inf)

def inv_var_2_var (inv_var):
    """
    Takes an inv_var and converts it to a variance. Carefully 
    handling zeros and inf values
    """
    fill = np.inf
    inv_var = np.asarray(inv_var).astype(float)
    zeros = (inv_var <= 0.0) # find the non_zeros
    inv_var[zeros] = -1.0
    var = 1.0/inv_var # inverse
    var[zeros] = fill
    return var

def var_2_inv_var (var):
    """
    Takes variance and converts to an inverse variance. Carefully
    handling zeros and inf values
    """
    fill = np.inf
    var = np.asarray(var).astype(float)
    zeros = (var <= 0.0) # find non zeros
    var[zeros] = -1.0
    inv_var = 1.0/var # inverse
    inv_var[zeros] = fill
    return inv_var

def clean_variances(variance, zero_ok=False, fill=np.inf):
    """takes input variances which may include nan's 0's and negative values
    and replaces those values with a fill value.
    
    variance: ndarray
      the array of variances array
    zero_ok: bool
      if True zero entries will be allowed to persist in the output
    fill: float
      the value to replace bad variances with
    """
    bad_mask = np.isnan(variance)
    if zero_ok:
        bad_mask += variance < 0
    else:
        bad_mask += variance <= 0
    new_variance = np.where(bad_mask, fill, variance)
    return new_variance

def clean_inverse_variances(inverse_variance):
    """takes input inverse variances which may include nan's, negative values
    and np.inf and replaces those values with 0
    """
    bad_mask = np.isnan(inverse_variance)
    bad_mask += inverse_variance < 0
    bad_mask += inverse_variance == np.inf
    new_variance = np.where(bad_mask, np.inf, inverse_variance)
    return new_variance

def reduce_output_shape (arr):
    shape = arr.shape
    new_shape = tuple()
    for i in xrange(len(shape)): 
        if shape[i] != 1: 
            new_shape += (shape[i],)
    return arr.reshape(new_shape)

#TODO: write this cross correlator in cython
def cross_corr(arr1, arr2, offset_number, overlap_adj=False):
    """cross correlate two arrays of the same size.
    correlating over a range of pixel offsets from -offset_number to 
    +offset_number and returning a 2*offset_number+1 length asarray. 
    To adjust for differences in the number of pixels overlapped if 
    overlap_adj==True we divide the result by the number of pixels
    included in the overlap of the two spectra.
    """
    assert len(arr1) == len(arr2)
    npts = len(arr1)
    offs = int(offset_number)
    cor_out = np.zeros(2*offs+1)
    offsets = range(-offs, offs+1)
    for offset_idx in range(len(offsets)):
        #print "offsets", offsets[offset_idx]
        #print "shapes", arr1.shape, arr2.shape
        coff = offsets[offset_idx]
        lb1, ub1 = max(0, coff), min(npts, npts+coff)
        lb2, ub2 = max(0, -coff), min(npts, npts-coff)
        cur_corr = np.sum(arr1[lb1:ub1]*arr2[lb2:ub2])
        if overlap_adj:
            n_overlap = min(ub1, ub2) - max(lb1, lb2)
            cur_corr /= float(n_overlap)
        cor_out[offset_idx] = cur_corr
    return cor_out


def local_gaussian_fit(yvalues, peak_idx=None, fit_width=2, xvalues=None):
    """fit a quadratic function taking pixels from 
    peak_idx - fit_width to peak_idx + fit_width, to the log of the yvalues
    passed in. Giving back the parameters of a best fit gaussian.
    
    inputs:
    values: numpy.ndarray 
      the asarray of input values to which the fit will be made
    peak_idx: int
      the rough pixel location of the maximum about which to fit.
      if None the global maximum is found and used.
    fit_width: float
      the width of the gaussian fit
    xvalues: numpy.ndarray
      if None then the parameters of the gaussian will be determined
      in terms of indicies (e.g. the center occurs at index=20.82) if given
      xvalues is interpreted as the corresponding x values for the yvalues
      asarray and the returned coefficients will be for that coordinate space
      instead.
    
    returns:
    center, sigma, peak_y_value
    """
    #import pdb; pdb.set_trace()
    if peak_idx is None:
        peak_idx = np.argmax(yvalues)
    lb = max(peak_idx - fit_width, 0)
    ub = min(peak_idx + fit_width, len(yvalues)-1)
    if xvalues == None:
        chopped_xvals = np.arange(lb, ub+1)
        peak_xval = peak_idx
    else:
        assert len(yvalues) == len(xvalues)
        chopped_xvals = xvalues[lb:ub+1]
        peak_xval = xvalues[peak_idx]
    xmatrix = np.ones((len(chopped_xvals), 3))
    delta = chopped_xvals-peak_xval
    xmatrix[:, 1] = delta
    xmatrix[:, 2] = delta**2
    chopped_yvalues = np.log(yvalues[lb:ub+1])
    #if the y values are negative 
    poly_coeffs = np.linalg.lstsq(xmatrix, chopped_yvalues)[0]
    offset = -poly_coeffs[1]/(2*poly_coeffs[2])
    center = peak_xval + offset
    sigma = 1.0/np.sqrt(2*np.abs(poly_coeffs[2])) 
    center_p_vec = np.asarray([1.0, offset, offset**2])
    peak_y_value = np.dot(center_p_vec, poly_coeffs)
    return center, sigma, np.exp(peak_y_value)


def get_local_maxima(arr):
    # http://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-asarray/3689710#3689710
    """
    Takes an asarray and detects the peaks using the local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    # define an connected neighborhood
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#generate_binary_structure
    neighborhood = morphology.generate_binary_structure(len(arr.shape),2)
    # apply the local maximum filter; all locations of maximum value 
    # in their neighborhood are set to 1
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.filters.html#minimum_filter
    local_max = (filters.maximum_filter(arr, footprint=neighborhood)==arr)
    # local_max is a mask that contains the peaks we are 
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.
    # 
    # we create the mask of the background
    background = (arr==0)
    # 
    # a little technicality: we must erode the background in order to 
    # successfully subtract it from local_max, otherwise a line will 
    # appear along the background border (artifact of the local maximum filter)
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#binary_erosion
    eroded_background = morphology.binary_erosion(
        background, structure=neighborhood, border_value=1)
    # 
    # we obtain the final mask, containing only peaks, 
    # by removing the background from the local_min mask
    detected_maxima = local_max - eroded_background
    return detected_maxima    

def wavelet_transform_fft(values, g_widths):
    """create a grid of gaussian line profiles and perform a wavelet transform
    over the grid. (a.k.a. simply convolve the data with all line profiles
    in the grid in an efficient way and then return an asarray that contains
    all of these convolutions.
    
    inputs:
    values: the asarray of values to transform
    g_widths: an asarray of gaussian widths (in pixels) to create profiles for 
    """
    n_pts ,= values.shape
    n_trans = 2**(int(np.log2(n_pts) + 1.5))
    n_g_widths = len(g_widths)
    val_fft = fftpack.fft(values, n=n_trans)
    out_dat = np.zeros((n_g_widths, n_pts), dtype=float)
    for g_width_idx in xrange(n_g_widths):
        c_g_width = g_widths[g_width_idx]
        l2norm = np.power(1.0/(np.pi*c_g_width), 0.25)
        deltas = np.arange(n_pts, dtype=float) - n_pts/2
        lprof = l2norm*np.exp(-(deltas/(2.0*c_g_width))**2)
        #we should normalize them via the L2 norm so we can see maxima effectively
        ltrans = np.abs(fftpack.fft(lprof, n=n_trans))
        wvtrans = fftpack.ifft(val_fft*ltrans)
        out_dat[g_width_idx] = wvtrans[:n_pts].real
    return out_dat

def wavelet_transform(values, g_widths, mask):
    """create a grid of gaussian line profiles and perform a wavelet transform
    over the grid. (a.k.a. simply convolve the data with all line profiles
    in the grid in an efficient way and then return an asarray that contains
    all of these convolutions.
    
    inputs:
    values: the asarray of values to transform
    g_widths: an asarray of gaussian widths (in pixels) to create profiles for 
    """
    n_pts ,= values.shape
    n_g_widths = len(g_widths)
    out_dat = np.zeros((n_g_widths, n_pts), dtype=float)
    max_delt = min((n_pts-1)/2, 5*np.max(g_widths))
    deltas = np.arange(2*max_delt+1) - max_delt
    if mask==None:
        mask = np.ones(values.shape)
    for g_width_idx in xrange(n_g_widths):
        c_g_width = g_widths[g_width_idx]
        lprof = np.exp(-0.5*(deltas/c_g_width)**2)
        conv_norm = np.convolve(lprof, mask, mode="same")
        conv_norm = np.where(conv_norm, conv_norm, 1.0)
        out_dat[g_width_idx] = np.convolve(values*mask, lprof, mode="same")/conv_norm
    return out_dat

def _incr_left(idx, maxima, bad_mask):
    if (idx - 1) < 0:
        return idx, True
    elif bad_mask[idx-1]:
        return idx, True
    elif maxima[idx-1]:
        return idx-1, True
    else:
        return idx-1, False

def _incr_right(idx, maxima, bad_mask):
    if (idx + 1) > len(maxima) -1:
        return idx, True
    elif bad_mask[idx+1]:
        return idx, True
    elif maxima[idx+1]:
        return idx+1, True
    else:
        return idx+1, False

def trough_bounds(maxima, start_idx, bad_mask=None):
    if bad_mask==None:
        bad_mask = np.zeros(maxima.shape, dtype=bool)
    left_blocked = False
    right_blocked = False
    left_idx = start_idx
    right_idx = start_idx
    while not left_blocked:
        left_idx, left_blocked = _incr_left(left_idx, maxima, bad_mask)
    while not right_blocked:
        right_idx, right_blocked = _incr_right(right_idx, maxima, bad_mask)
    return left_idx, right_idx

class MinimaStatistics:
    pass

def minima_statistics(values, variance, last_delta_fraction=1.0, max_sm_radius=0.5):
    #import pdb; pdb.set_trace()
    n_pts = len(values)
    bad_vals = np.isnan(variance)+(variance <= 0)+(variance > 1.0e40)
    sm_pix = int(max_sm_radius*12)
    sm_pix += sm_pix % 2
    sm_pix += 1
    window = np.exp(-np.linspace(-3, 3, sm_pix)**2)
    sm_norm = np.convolve(True-bad_vals, window, mode="same")
    sm_norm = np.where((sm_norm > 0), sm_norm, 1.0)
    smoothed_vals = np.convolve(values*(True-bad_vals), window, mode="same")/sm_norm
    maxima = get_local_maxima(smoothed_vals)
    #import pdb; pdb.set_trace()
    minima = get_local_maxima(-values)
    min_idxs ,= np.where(minima*(True-bad_vals))
    l_z = []
    r_z = []
    n_cons = []
    left_idxs = []
    right_idxs = []
    for min_idx in min_idxs:
        min_val = values[min_idx]
        left_idx, right_idx = trough_bounds(maxima, min_idx, bad_vals)
        left_delta_cor = np.abs(values[left_idx] - values[min(n_pts-1, left_idx+1)])
        left_h = (values[left_idx]-min_val)
        left_h -= (1.0-last_delta_fraction)*left_delta_cor
        left_z = left_h/np.sqrt(variance[left_idx] + variance[min_idx])
        right_delta_cor = np.abs(values[right_idx] - values[max(0, right_idx-1)])
        right_h = (values[right_idx]-min_val)
        right_h -= (1.0-last_delta_fraction)*right_delta_cor
        right_z = right_h/np.sqrt(variance[right_idx] + variance[min_idx])
        l_z.append(left_z)
        r_z.append(right_z)
        n_cons.append(right_idx-left_idx)
        left_idxs.append(left_idx)
        right_idxs.append(right_idx)
    l_z = np.asarray(l_z)
    r_z = np.asarray(r_z)
    n_cons = np.asarray(n_cons)
    left_idxs = np.asarray(left_idxs)
    right_idxs = np.asarray(right_idxs)
    ms = MinimaStatistics()
    ms.left_idxs = left_idxs
    ms.min_idxs = min_idxs
    ms.right_idxs = right_idxs
    ms.l_z = l_z
    ms.r_z = r_z
    ms.n_c = n_cons
    return ms

def detect_features(values, 
                    variance, 
                    reject_fraction=0.5,
                    last_delta_fraction=1.0, 
                    mask_back_off=-1,
                    max_sm_radius=0.5,
                    stat_func=lambda lz, rz, nc: np.sqrt(lz**2+rz**2+0.25*(nc-2)**2),
                    min_stats=None):
    """detect spectral features using an arbitrary statistic
    by default the stat_func is
    s = np.sqrt(l_z**2+r_z**2+0.25*(n_c-2)**2)
    where l_z is the z score of the difference between the minima
    and the closest local maximum on the left, r_z is the same for the right.
    n_c is the number of contiguous points around the minimum for which the
    value monotonically increases.
    
    inputs
    values: ndarray
        the flux values
    variance: ndarray
        the variance values associated to the flux values
    threshold: float
        the threshold to put on the statistic s for a detection
    last_delta_fraction: float 
        the fraction of the difference between the first
        maxima pixel found and the pixel value immediately prior to be used.
    mask_back_off: int
        the number of pixels to back off from the local maxima
        when creating the feature mask. a value of 0 would mask the maxima 
        a value of 1 would leave the maxima unmasked, a value of -1 masks 
        the maximum and one pixel out, a value of -2 masks the maxima and
        2 pixels out etc etc.
    max_sm_radius: float
        the pixel with a local maximum
    min_stats: MinimaStatistics 
        if the minima statistics have already been calculated you can pass it in
        and it will not be recalculated.

    returns:
    a tuple of 
    left_idxs, min_idxs, right_idxs, feature_mask
    left_idxs: the index of the left bounding maximum
    min_idxs: the index of the local minimum
    right_idxs: the index of the right bounding maximum
    feature_mask: a boolean asarray the shape of values which is true if there is
        no detected feature affecting the associated pixel.
    """
    #import matplotlib.pyplot as plt
    #import pdb; pdb.set_trace()
    if min_stats != None:
        msres = min_stats
    msres = minima_statistics(values, variance, last_delta_fraction)
    left_idxs, min_idxs, right_idxs = msres.left_idxs, msres.min_idxs, msres.right_idxs
    l_z, r_z, n_cons = msres.l_z, msres.r_z, msres.n_c
    s = stat_func(l_z, r_z, n_cons)
    sorted_s = np.sort(s)
    switch_idx = int(len(sorted_s)*(1.0-reject_fraction))
    threshold = np.mean(sorted_s[switch_idx:switch_idx+2])
    mask = s > threshold
    left_idxs = left_idxs[mask].copy()
    min_idxs = min_idxs[mask].copy()
    right_idxs = right_idxs[mask].copy()
    feature_mask = np.ones(values.shape, dtype=bool)
    for left_idx, right_idx in zip(left_idxs, right_idxs):
        feature_mask[max(0, left_idx+mask_back_off):right_idx-mask_back_off+1] = False 
    return msres, feature_mask


def smoothed_mad_error(spectrum, 
                       smoothing_scale=3.0, 
                       error_scale = 200,
                       apply_poisson=True, 
                       overwrite_error=False):
    cinv_var = spectrum.inv_var
    good_mask = (cinv_var > 0)*(spectrum.flux > 0)
    #detect and reject perfectly flat regions
    flux_der = scipy.gradient(spectrum.flux)
    flux_sec_der = scipy.gradient(flux_der)
    good_mask *= np.abs(flux_sec_der) > 0
    #smfl ,= wavelet_transform(spectrum.flux, [smoothing_scale], inv_mask)
    smfl = filters.gaussian_filter(spectrum.flux, smoothing_scale)
    eff_width = 2.0*smoothing_scale #effective filter width
    #need to correct for loss of variance from the averaging
    correction_factor = eff_width/max(1, (eff_width-1))
    diffs = (spectrum.flux - smfl)
    #use the running median change to protect ourselves from outliers
    mad = filters.median_filter(np.abs(diffs), error_scale)
    char_sigma = correction_factor*1.48*mad
    if apply_poisson:
        var = (smfl/np.median(smfl[good_mask])*char_sigma)**2
    else:
        var = char_sigma**2
    #put an upper limit on detected snr
    var += spectrum.flux*1e-3
    new_inv_var = np.where(good_mask, 1.0/var, 0.0)
    new_inv_var = clean_inverse_variances(new_inv_var)
    if overwrite_error:
        spectrum.inv_var = new_inv_var
    return new_inv_var

def min_delta_bins(x, min_delta, target_n=1, forced_breaks=None):
    """return a set of x values which partition x into bins.
    each bin must be at least min_delta wide and must contain at least target_n
    points in the x asarray (with an exception made when forced_breaks are used).
    The bins attempt to be the smallest that they can be while achieving both
    these constraints.
    Bins are simply built up from the left until the constraints are met 
    and then a new bin is begun, so the optimum bins do not necessarily result.
    
    inputs
    
    x: iterable
      the coordinates to choose a binning for
    min_delta: float
      the smallest allowed bin size
    target_n: int
      the desired number of objects per bin
    forced_breaks: list
      if forced_breaks is specified then the numbers in forced_breaks will 
      be included in the output bins. The bins will otherwise still attempt to
      meet the min_delta and target_n values.
    """
    sorted_x = np.sort(np.asarray(x))
    x_diff = np.abs(sorted_x[1:]-sorted_x[:-1])
    if forced_breaks == None:
        forced_breaks = []
    forced_breaks = sorted(forced_breaks)
    
    diff_sum = np.cumsum(x_diff)
    last_diff_sum = 0
    last_diff_idx = 0
    current_n = 0
    bins = [sorted_x[0]]
    last_bin_forced = True
    next_force_idx = 0
    for i in range(len(x_diff)):
        current_n += 1
        if len(forced_breaks) > next_force_idx:
            next_fb = forced_breaks[next_force_idx]
            if sorted_x[i+1] > next_fb:
                bins.append(next_fb)
                next_force_idx += 1
                last_diff_sum = diff_sum[i]
                below_target_n = current_n < target_n
                below_target_delta = next_fb - sorted_x[last_diff_idx]
                if below_target_n or below_target_delta: 
                    if not last_bin_forced:
                        bins.pop()
                last_bin_forced = True
                last_diff_idx = i
                last_diff_sum = diff_sum[i]
                current_n = 0
        if diff_sum[i]-last_diff_sum > min_delta and current_n >= target_n:
            avg_br = 0.5*(sorted_x[i] + sorted_x[i+1])
            bins.append(avg_br)
            last_diff_sum = diff_sum[i]
            last_diff_idx = i
            current_n = 0
            last_bin_forced = False
    if not last_bin_forced:
        bins.pop()
    if bins[-1] != sorted_x[-1]:
        bins.append(sorted_x[-1])
    return bins


def layered_median_mask(arr, n_layers=3, first_layer_width=31, last_layer_width=11, rejection_sigma=2.0):
    marr = np.asarray(arr)
    assert n_layers > 1
    assert first_layer_width >= 1
    assert last_layer_width >= 1
    layer_widths = np.asarray(np.linspace(first_layer_width, last_layer_width, n_layers), dtype=int)
    mask = np.ones(marr.shape, dtype=bool)
    for layer_idx in range(n_layers):
        lw = layer_widths[layer_idx]
        masked_arr = marr[mask]
        filtered = filters.median_filter(masked_arr, lw)
        local_mad = filters.median_filter(np.abs(masked_arr-filtered), lw)
        mask[mask] = masked_arr >= (filtered - rejection_sigma*1.4*local_mad)
    return mask

def smooth_ppol_fit(x, y, y_inv=None, order=3, mask=None, mult=None, partition="adaptive", partition_kwargs=None):
    if partition_kwargs == None:
        partition_kwargs = {}
    
    if y_inv == None:
        y_inv = np.ones(y.shape)
    
    if mult==None:
        mult = np.ones(y.shape, dtype=float)
    
    if mask == None:
        mask = np.ones(y.shape, dtype=bool)
 
    masked_x = x[mask]
    if partition == "adaptive":
        try:
            partition = min_delta_bins(masked_x, **partition_kwargs)[1:-1]
        except:
            partition = [np.median(masked_x)]
    
    pp_gen = piecewise_polynomial.RCPPB(poly_order=order, control_points=partition)
    ppol_basis = pp_gen.get_basis(masked_x).transpose()
    ppol_basis *= mult.reshape((-1, 1))
    in_sig = 1.0/np.sqrt(y_inv[mask])
    med_sig = np.median(in_sig)
    fit_coeffs = pseudo_huber_irls(ppol_basis, y, 
                      sigma=in_sig, 
                      gamma=2.0*med_sig, 
                      max_iter=5, conv_thresh=1e-4)
    n_polys = len(partition) + 1
    n_coeffs = order+1
    out_coeffs = np.zeros((n_polys, n_coeffs))
    for basis_idx in xrange(pp_gen.n_basis):
        c_coeffs = pp_gen.basis_coefficients[basis_idx].reshape((n_polys, n_coeffs))
        out_coeffs += c_coeffs*fit_coeffs[basis_idx]
    out_ppol = piecewise_polynomial.PiecewisePolynomial(out_coeffs, partition, centers=pp_gen.centers, scales=pp_gen.scales, bounds=pp_gen.bounds)
    return out_ppol
 
def echelle_normalize(spectra, masks="layered median", partition="adaptive", mask_kwargs=None, partition_kwargs=None):
    """normalize in a way that shares normalization shape accross nearby
    orders"""
    n_spec = len(spectra)
    if mask_kwargs == None:
        mask_kwargs = {}
    if partition_kwargs == None:
        partition_kwargs = {}
    
    if masks == "layered median":
        masks = []
        for spec_idx in range(n_spec):
            flux = spectra[spec_idx].flux
            mask = layered_median_mask(flux, **mask_kwargs)
            masks.append(mask)
    elif len(masks) == n_spec and len(masks[0]) == len(flux):
        masks = [np.asarray(mask, dtype=bool) for mask in masks]
    
    masked_pixels = np.hstack([np.arange(len(spectra[i].flux))[masks[i]] for i in range(n_spec)])
    masked_wvs = np.hstack([spectra[i].wv[masks[i]] for i in range(n_spec)])
    order_set = range(1, n_spec+1)
    order_nums = np.hstack([np.ones(len(spectra[i-1].flux), dtype=float)[masks[i-1]]*i for i in order_set])
    masked_fluxes = np.hstack([spectra[i].flux[masks[i]] for i in range(n_spec)])
    masked_inv_var = np.hstack([spectra[i].inv_var[masks[i]] for i in range(n_spec)])
    #break the model up into 3 models one for each of pixel, wv, order
    
    npts = len(masked_fluxes)
    
    #initialize the models with ones
    order_mod = np.ones(npts, dtype=float)
    pixel_mod = np.ones(npts, dtype=float)
    wv_mod = np.ones(npts, dtype=float)
    
    try:
        order_cp = min_delta_bins(order_set, 1, 1)[1:-1]
    except:
        order_cp = np.median(order_set)
    try:
        wv_cp = min_delta_bins(masked_wvs, 300, 1000)[1:-1]
    except:
        wv_cp = [np.median(masked_wvs)]
    max_pix = np.max(masked_pixels)
    try:
        pixel_cp = min_delta_bins(np.arange(0, max_pix), 100, 100)[1:-1]
    except:
        pixel_cp = [max_pix/2]
                                 
    #import matplotlib.pyplot as plt
    for iter_idx in range(3):
        #import pdb; pdb.set_trace()
        #fit order trends
        complement_mod = pixel_mod*wv_mod
        complement_mod = np.where(complement_mod > 0.1, complement_mod, 0.1) 
        order_poly = smooth_ppol_fit(order_nums, masked_fluxes, masked_inv_var, 2,
                                     mult = complement_mod,
                                     partition = order_cp,
                                     )
        order_mod = order_poly(order_nums)
        
        #fit pixel trends
        complement_mod = order_mod*wv_mod
        complement_mod = np.where(complement_mod > 0.1, complement_mod, 0.1) 
        pixel_poly = smooth_ppol_fit(masked_pixels, masked_fluxes, masked_inv_var, 2,
                                     mult = complement_mod,
                                     partition = pixel_cp,
                                     )
        pixel_mod = pixel_poly(masked_pixels)
        
        #fit wv_trends
        complement_mod = pixel_mod*order_mod
        complement_mod = np.where(complement_mod > 0.1, complement_mod, 0.1) 
        wv_poly = smooth_ppol_fit(masked_wvs, masked_fluxes, masked_inv_var, 2,
                                     mult = complement_mod,
                                     partition = wv_cp,
                                     )
        wv_mod = wv_poly(masked_wvs)
    
    norms = []
    for order_idx in range(1, len(spectra)+1):
        wvs = spectra[order_idx-1].wv
        order_nums = order_idx*np.ones(wvs.shape)
        pixels = np.arange(len(wvs))
        nmod = pixel_poly(pixels)*wv_poly(wvs)*order_poly(order_nums)
        norms.append(nmod)
    return norms

    
def approximate_normalization(spectrum,
                              partition_scale=500, 
                              poly_order=3,
                              mask_kwargs=None,
                              smart_partition=False, 
                              alpha=4.0,
                              beta=4.0,
                              H_mask_radius=2.0,
                              norm_hints=None,
                              overwrite=False,
                              min_stats=None,
                              ):
    """estimates a normalization curve for the input spectrum.
    spectrum: Spectrum
        a Spectrum object
    partition_scale: float
        a rough scale of the smallest allowable level of structure to be fit
        in pixels
    reject_fraction: float
        sets the feature detection cutoff at the value representing this fraction
        of all the feature troughs.
    poly_order: int
        the local order of spline to use
    mask_back_off: int
        a number of pixels to reduce the size of the feature mask by -1 enlarges
        the number of pixels excluded around a feature trough by 1 on each side
        and +1 would reduce by 1 on each side.
    smart_partition: bool
        if true attempt to use the optimal piecewise polynomial partitioning
        algorithm. If the smart partitioning fails we will automatically 
        switch to the quick way.
    alpha: float
        used only if smart_partition == True
        how expensive new partitions are
        see partitioning.partitioned_polynomial_model
    beta: float
        used only if smart_partition == True
        how expensive small block sizes are
        see partitioning.partitioned_polynomial_model
    norm_hints: None or tuple of arrays
        some suggestions for the norm which are entered into the fit.
        (but not the partitioning)
        wvs, continuum_fluxes, continuum_weights = norm_hints
    overwrite: bool
        if True the normalization object in the input spectrum is replaced
        with the calculated normalization estimate (it all goes into the efficiency)
    min_stats: MinimaStatistics
        if the minima statistics have already been you can pass them in here.
    
    returns
    ApproximateNorm object
    """
    #import matplotlib.pyplot as plt
    #import pdb; pdb.set_trace()
    wavelengths = spectrum.get_wvs()
    pix_delta = np.median(scipy.gradient(wavelengths))
    pscale = partition_scale*pix_delta
    flux = spectrum.flux
    variance = spectrum.get_var()
    inv_var = spectrum.get_inv_var()
    hmask = hydrogen.get_H_mask(wavelengths, H_mask_radius)
    good_mask = (inv_var > 0)*(flux > 0)*hmask
    #min_stats, fmask = detect_features(flux, variance, 
    #                                   reject_fraction=reject_fraction,
    #                                   mask_back_off=mask_back_off, 
    #                                   min_stats=min_stats)
    #fmask *= good_mask
    
    #generate a layered median mask.
    if mask_kwargs is None:
        mask_kwargs = {'n_layers':3, 'first_layer_width':201, 'last_layer_width':11, 'rejection_sigma':1.5} 
    fmask = layered_median_mask(flux, **mask_kwargs)*good_mask
    
    mwv = wavelengths[fmask].copy()
    mflux = flux[fmask].copy()
    minv = inv_var[fmask]
    if smart_partition:
        try:
            opt_part, mvps = partitioning.partitioned_polynomial_model(mwv, mflux, minv, 
                    poly_order=(poly_order,),
                    grouping_column=0, 
                    min_delta=pscale,
                    alpha=alpha, beta=beta, beta_epsilon=0.01)
            break_wvs = mwv[np.asarray(opt_part[1:-1], dtype=int)]
            use_simple_partition = False
        except:
            #the smart partitioning failed use the simple one instead
            use_simple_partition = True
    else:
        use_simple_partition = True
    if use_simple_partition:
        break_wvs = min_delta_bins(mwv, min_delta=pscale, target_n=20*(poly_order+1)+10)
        #roughly evenly space the break points partition_scale apart
    pp_gen = piecewise_polynomial.RCPPB(poly_order=poly_order, control_points=break_wvs)
    if norm_hints:
        hint_wvs, hint_flux, hint_inv_var = norm_hints
        mwv = np.hstack((mwv, hint_wvs))
        mflux = np.hstack((mflux, hint_flux))
        minv = np.hstack((minv, hint_inv_var))
    
    ppol_basis = pp_gen.get_basis(mwv).transpose()
    in_sig = 1.0/np.sqrt(minv)
    med_sig = np.median(in_sig)
    print med_sig, "med sig"
    fit_coeffs = pseudo_huber_irls(ppol_basis, mflux, 
                      sigma=in_sig, 
                      gamma=2.0*med_sig, 
                      max_iter=10, conv_thresh=1e-4)
    n_polys = len(break_wvs) + 1
    n_coeffs = poly_order+1
    out_coeffs = np.zeros((n_polys, n_coeffs))
    for basis_idx in xrange(pp_gen.n_basis):
        c_coeffs = pp_gen.basis_coefficients[basis_idx].reshape((n_polys, n_coeffs))
        out_coeffs += c_coeffs*fit_coeffs[basis_idx]
    continuum_ppol = piecewise_polynomial.PiecewisePolynomial(out_coeffs, break_wvs, centers=pp_gen.centers, scales=pp_gen.scales, bounds=pp_gen.bounds)
    approx_norm = continuum_ppol(wavelengths)
    if overwrite:
        spectrum.norm = approx_norm
        spectrum.feature_mask = fmask
    return approx_norm

def lad_fit(A, y):
    """finds the least absolute deviations fit for Ax = y
    inputs
    A  numpy asarray
    the matrix whose columns make up the basis vectors of the fit
    y numpy asarray
    the target output vector.
    """
    if not with_cvxopt:
        print "lad_fit function requires cvxopt but the import failed!"
    neg_ident_mat = -np.diag(np.ones(len(A)), 0)
    nrows, ncols = A.shape
    const_mat = np.empty((3*nrows, ncols+nrows))
    const_mat[:nrows, :ncols] = A
    const_mat[nrows:2*nrows, :ncols] = -A
    const_mat[2*nrows:, :ncols] = 0
    const_mat[:nrows, ncols:] = neg_ident_mat
    const_mat[nrows:2*nrows, ncols:] = neg_ident_mat
    const_mat[2*nrows:, ncols:] = neg_ident_mat
    bounds_vec = np.empty((3*nrows,))
    bounds_vec[:nrows] = y
    bounds_vec[nrows:2*nrows] = -y
    bounds_vec[2*nrows:] = 0
    objective_vec = np.ones(ncols+nrows)
    objective_vec[:ncols] = 0
    objective_vec = cvxopt.matrix(objective_vec.reshape(-1, 1)) 
    const_mat = cvxopt.matrix(const_mat)
    bounds_vec = cvxopt.matrix(bounds_vec.reshape(-1, 1))
    #print objective_vec, const_mat, bounds_vec
    opt_res = cvxopt.solvers.lp(objective_vec, const_mat, bounds_vec)
    return np.asarray(opt_res["x"]).reshape((-1,))[:ncols].copy(), opt_res

def vec_sort_lad(e, u):
    """determine the optimal scaling of source such that
    np.sum(np.abs(source*a - target)) is minimized using a sorting
    algorithm under the hood.
    """
    nz = u != 0
    unz = u[nz]
    signum = np.sign(unz)
    unz *= signum
    enz = np.asarray(signum*e[nz], dtype=float)
    ratios = enz/unz
    srat, sunz = zip(*sorted(zip(ratios, unz)))
    sunz = np.asarray(sunz)
    usum = -np.sum(sunz)
    usum_idx = 0
    while usum < 0:
        usum_idx += 1
        usum = np.sum(sunz[:usum_idx]) - np.sum(sunz[usum_idx:])
    opt_ratio = srat[usum_idx-1]
    return opt_ratio

def l1_factor(input_matrix, input_weights, rank=3, n_iter=3):
    rows_in, cols_in = input_matrix.shape
    w = np.random.random((rows_in, rank))-0.5
    h = np.linalg.lstsq(w, input_matrix)[0]
    for iter_idx in range(n_iter):
        "beginning matrix factor iteration %d" % (iter_idx+1)
        for rank_idx in range(rank):
            mod = np.dot(w, h)
            resid = input_matrix-mod
            for row_idx in range(rows_in):
                cresid = resid[row_idx] + w[row_idx, rank_idx]*h[rank_idx]
                opt_w = vec_sort_lad(cresid, h[rank_idx])
                w[row_idx, rank_idx] = opt_w
            mod = np.dot(w, h)
            resid = input_matrix-mod
            for col_idx in range(cols_in):
                cresid = resid[:, col_idx] + w[:, rank_idx]*h[rank_idx, col_idx]
                opt_h = vec_sort_lad(cresid, w[:, rank_idx])
                h[rank_idx, col_idx] = opt_h
    return w, h

def pseudo_huber_cost(resid_vec, sigma, gamma):
    sig4 = sigma**4
    gam4 = gamma**4 
    rat4 = sig4/gam4
    weights = 1.0/(gamma*np.sqrt(rat4 + resid_vec**2))
    return np.sum(resid_vec**2*weights)

def pseudo_huber_irls(A, b, sigma, gamma, max_iter=100, conv_thresh=1e-4):
    """
    fits for an optimal x such that
    Ax ~ b
    according to errors related to the pseudo-huber cost function
    attempting to minimize the cost function
    gamma*(sigma**4/gamma**4 + (Ax - b)**2)**(-1/2.0)
    taylor expanding this around zero we find that for small
    deltas this cost function reduces to the log probability for normal
    errors with variance sigma**2.
    For large errors where (Ax-b)**2 >> (sigma/gamma)**4
    it is easy to see that the cost becomes linear in the magnitude of the
    residual with cost |(Ax-b)|/gamma. This fit is carried out by 
    Iteratively Reweighted Least Squares (IRLS). Whereby we originally
    carry out the least squares fit and then iteratively reweight by a 
    function of the residuals in the hope of approximating the optimum
    of the desired cost function. 
    
    inputs
    
    A: numpy asarray or scipy.sparse matrix
    the fit matrix the results b will be fit as linear combinations of the 
    columns of A
    
    b: numpy asarray
    the data to be fit to
    
    sigma: numpy asarray or number
    the square root of the gaussian variance
    
    gamma: numpy asarray or number or None
    the linear cost scale. If you have some estimate of what scale of
    deviation is large enough to be a likely indicate an outlier use that.
    
    max_iter: integer
    the maximum number of reweighting and solving iterations to carry out
    before giving up on convergence and just returning the current solution.
    
    conv_thresh: float
    the result is considered to have converged if the average of the absolute
    value of the current residuals minus the previous iteration residuals 
    is less than conv_thresh
    """
    sig4 = sigma**4
    gam4 = gamma**4 
    rat4 = sig4/gam4
    
    if scipy.sparse.issparse(A):
        fit = scipy.sparse.linalg.lsqr(A, b)[0]
        last_deltas = A*fit
        
        for iter_idx in range(max_iter):
            mod = A*fit
            deltas = mod-b
            weights = 1.0/(gamma*np.sqrt(rat4 + deltas**2))
            ata_inv = A.transpose()*(weights*A)
            fit = scipy.sparse.linalg.lsqr(ata_inv, A.transpose()*(weights*b))[0]
            if np.mean(np.abs(last_deltas-deltas)) < conv_thresh:
                break
            last_deltas = deltas
    else:
        fit = np.linalg.lstsq(A, b)[0]
        last_deltas = np.dot(A, fit)
    
        for iter_idx in range(max_iter):
            mod = np.dot(A, fit)
            deltas = mod-b
            weights = 1.0/(gamma*np.sqrt(rat4 + deltas**2))
            ata_inv = np.dot(A.transpose()*weights, A)
            pinv = np.linalg.pinv(ata_inv)
            fit = np.dot(pinv, np.dot(A.transpose()*weights, b))
            if np.mean(np.abs(last_deltas-deltas)) < conv_thresh:
                break
            last_deltas = deltas
    return fit

def cache_decorator (cache,key):
    def outer (func):
        def inner (*args,**kwargs):
            if cache.get(key,None) is not None:
                return cache[key]
            ret = func(*args,**kwargs)
            cache[key] = ret
            return ret
        return inner
    return outer

def vac_to_air_sdss(vac_wvs):
    return vac_wvs/(1.0 + 2.735182e-4 + 131.4182 /vac_wvs**2 + 2.76249e8/vac_wvs**4)

def vac_to_air_apogee(vac_wvs, a=0.0, b1=5.792105e-2, b2=1.67917e-3, c1=238.0185, c2=57.362):
    """
    converts vaccuum wavelengths in angstroms to air wavelengths.
    the default constants are those suggested by APOGEE.
    """
    #assume that the vac_wvs are provided in angstroms
    #micron_wvs = vac_wvs/10**4
    #actually when I make plots doing the comparison it looks like they really
    #wanted angstroms as input anyway...
    micron_wvs = vac_wvs
    #the coefficients given by apogee are for wavelengths in microns so use that
    inv_sq_vac = micron_wvs**-2
    refractive_index = (1.0+a+b1/(c1-inv_sq_vac) + b2/(c2-inv_sq_vac))
    return vac_wvs/refractive_index

def load_star_fits(fname):
    hdul = fits.open(fname)
    wvs = np.asarray(hdul[1].data)
    flux = np.asarray(hdul[2].data)
    inv_var = np.asarray(hdul[3].data)
    #TODO: add proper handling of resolution on readin
    spec = thimbles.Spectrum(wvs, flux, inv_var)
    return spec
    
def write_star_fits(star ,fname):
    #make the HDU's
    if star.coadd == None:
        print "there is no coadd, cannot write out"
        return
    wvs = star.coadd.get_wvs()
    wv_hdr = fits.Header()
    wv_hdu = fits.ImageHDU(wvs, name="wavelengths")
    flux = star.coadd.flux
    flux_hdu = fits.ImageHDU(flux, name="flux")
    inv_var = star.coadd.get_inv_var()
    inv_var_hdu = fits.ImageHDU(inv_var, name="inverse variance")
    resolution = np.ones(len(wvs))
    res_hdr = fits.Header()
    res_hdr["dip_type"] = "gaussian"
    res_hdu = fits.ImageHDU(resolution, header=res_hdr, name="resolution")
    phdu = fits.PrimaryHDU()
    hdul = fits.HDUList([phdu, wv_hdu, flux_hdu, inv_var_hdu, res_hdu])
    hdul.writeto(fname)

hcoverk = 1.4387751297851e8
blconst = 1.191074728e24

def blam(wavelength, temperature):
    """blackbody intensity for wavelengths in angstroms note this does not
    take into account the sampling derivative"""
    return blconst*wavelength**-5.0/(np.exp(hcoverk/(temperature*wavelength))-1.0)


#wien_constant = 2.8977721(26)e-3 in units of M*K
wien_constant = 2.897772126e7 #in units of Angstrom*Kelvin
def blackbody_spectrum(sampling_wavelengths, temperature,  normalize = True):
    dlam_dx = scipy.gradient(sampling_wavelengths)
    bbspec = blam(sampling_wavelengths, temperature)/dlam_dx
    if normalize:
        peak_wv = wien_constant/temperature
        peak_val = blam(peak_wv, temperature)
        bbspec /= peak_val
    return bbspec
