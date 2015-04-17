
import numpy as np
import scipy
from scipy.special import erf, erfinv
from scipy import ndimage

def running_acorr(arr, avg=None, window_sigma=3, ncorr=21, mode="reflect"):
    """
    generate a running auto-correlation estimate.
    That is try to estimate 
    
    
    parameters
    arr: numpy ndarray
      the data array to be autocorrelated  
    avg: function, numpy array, float or None
      The running mean of the data in arr.
      if avg is a callable it will be called on arr and the result treated as 
      the running mean. If avg is None the median value of arr will be used.
    window_sigma: float
    
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

def max_norm(spectrum):
    return np.repeat(np.max(spectrum.flux), len(spectrum))

def mean_norm(spectrum):
    return np.repeat(np.mean(spectrum.flux), len(spectrum))

def median_norm(spectrum):
    return np.repeat(np.median(spectrum.flux), len(spectrum))

def sorting_norm(spectrum, min_frac=0.75, max_frac=0.97, n_samples=15, fit_plot=False):
    npts = len(spectrum)
    assert n_samples >= 2
    assert min_frac > 0.0
    assert max_frac < 1.0
    sortflux = np.sort(spectrum.flux)
    fractions = np.linspace(min_frac, max_frac, n_samples)
    sample_fluxes = np.zeros(n_samples)
    float_idxes = (npts-1)*fractions
    upper_idxes = np.ceil(float_idxes).astype(int)
    lower_idxes = np.floor(float_idxes).astype(int)
    alphas = float_idxes % 1.0
    vals = sortflux[upper_idxes]*alphas + sortflux[lower_idxes]*(1.0-alphas)
    fit_mat = np.ones((n_samples, 3))
    fit_mat[:, 1] = erfinv(2.0*(fractions-0.5))
    fit_mat[:, 2] = (1.0-fractions)**2
    res = np.linalg.lstsq(fit_mat, vals)[0]
    if res[2] > 0:
        fit_mat = fit_mat[:, :2]
        res = np.linalg.lstsq(fit_mat, vals)[0]
    if fit_plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2)
        axes[0].plot(fractions, vals, label="flux[quantile]")
        mod = np.dot(fit_mat, res)
        axes[0].plot(fractions, mod, label="best fit model")
        axes[0].set_xlabel("quantile") 
        axes[1].set_ylabel("flux")
        axes[1].plot(fractions, vals-mod)
        axes[1].set_xlabel("quantile")
        axes[1].set_ylabel("residuals")
        plt.show()
    #import pdb; pdb.set_trace()
    return np.repeat(res[0], npts)


def iterative_sorting_norm(
        spectrum, 
        init_min=0.1, 
        init_max=0.98, 
        degree=4,
        n_iter=4,
):
    x = spectrum.wv/spectrum.wv[-1]
    y = spectrum.flux
    pfit = np.polyfit(x, y, degree=degree)


def feature_masked_partitioned_lad_norm(
        spectrum,
        partition_scale=400, 
        poly_order=3,
        mask_kwargs=None,
        H_mask_radius=-3.5,
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
    break_wvs = min_delta_bins(mwv, min_delta=pscale, target_n=20*(poly_order+1)+10)
    pp_gen = piecewise_polynomial.RCPPB(poly_order=poly_order, control_points=break_wvs)
    if norm_hints:
        hint_wvs, hint_flux, hint_inv_var = norm_hints
        mwv = np.hstack((mwv, hint_wvs))
        mflux = np.hstack((mflux, hint_flux))
        minv = np.hstack((minv, hint_inv_var))
    
    ppol_basis = pp_gen.get_basis(mwv).transpose()
    in_sig = 1.0/np.sqrt(minv)
    med_sig = np.median(in_sig)
    print(med_sig, "med sig")
    fit_coeffs = pseudo_huber_irls(ppol_basis, mflux, 
                      sigma=in_sig, 
                      gamma=2.0*med_sig, 
                      max_iter=10, conv_thresh=1e-4)
    n_polys = len(break_wvs) + 1
    n_coeffs = poly_order+1
    out_coeffs = np.zeros((n_polys, n_coeffs))
    for basis_idx in range(pp_gen.n_basis):
        c_coeffs = pp_gen.basis_coefficients[basis_idx].reshape((n_polys, n_coeffs))
        out_coeffs += c_coeffs*fit_coeffs[basis_idx]
    continuum_ppol = piecewise_polynomial.PiecewisePolynomial(out_coeffs, break_wvs, centers=pp_gen.centers, scales=pp_gen.scales, bounds=pp_gen.bounds)
    approx_norm = continuum_ppol(wavelengths)
    if overwrite:
        spectrum.norm = approx_norm
        spectrum.feature_mask = fmask
    return approx_norm
