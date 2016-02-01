
import numpy as np
import scipy
from scipy.special import erf, erfinv
from scipy import ndimage
import thimbles as tmb

def max_norm(spectrum):
    return np.repeat(np.max(spectrum.flux), len(spectrum))

def mean_norm(spectrum):
    return np.repeat(np.mean(spectrum.flux), len(spectrum))

def median_norm(spectrum):
    return np.repeat(np.median(spectrum.flux), len(spectrum))


def adjusted_median(
        values,
        min_frac=0.6,
        max_frac=0.95,
        min_val=None,
        max_val=None,
        mask=None,
        n_samples=15,
        contaminant_sign=-1,
        return_fit_dict=False,
    ):
    """
    Estimate the median of a set of values which may suffer from 
    contaminants which push values systematically high or low.
    
    Instead of estimating the median from only the central values
    we fit for the median using the sorted values between any 
    specified minimum fraction and maximum fraction. Crucially
    These values do not have to include the median value but can
    use only the high or only low ends of the distribution to 
    estimate thus cutting out a majority of contaminants.
    
    The shape of the sorted values as a function of fraction is 
    assumed to be gaussian in shape.
    
    a quadratic tail of contaminants is fit for and removed if
    the sense of its deviation matches the expected sign of the 
    contaminants and if not this correction term is neglected.
    """
    
    
    if mask is None:
        mask = np.ones(len(values), dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    
    if not min_val is None:
        mask *= values < min_val
    if not max_val is None:
        mask *= values > max_val
    
    sort_vals = np.sort(values[mask])
    npts = len(sort_vals)
    if npts < 2:
        if npts == 1:
            return sort_vals[0]
        else:
            return np.nan
    
    fractions = np.linspace(min_frac, max_frac, n_samples)
    
    sample_vals = np.zeros(n_samples)
    float_idxes = (npts-1)*fractions
    upper_idxes = np.ceil(float_idxes).astype(int)
    lower_idxes = np.floor(float_idxes).astype(int)
    alphas = float_idxes % 1.0
    vals = sort_vals[upper_idxes]*alphas + sort_vals[lower_idxes]*(1.0-alphas)
    
    fit_mat = np.ones((n_samples, 3))
    fit_mat[:, 1] = erfinv(2.0*(fractions-0.5))
    fit_mat[:, 2] = (1.0-fractions)**2
    
    res = np.linalg.lstsq(fit_mat, vals)
    coeffs = res[0]
    #check if the contaminant term has the expected sign
    if coeffs[2]*contaminant_sign < 0:
        #leave out the quadratic contaminant term and refit
        res = np.linalg.lstsq(fit_mat[:, :2], vals)
        coeffs = res[0]
    
    if return_fit_dict:
        fd = dict(
            fractions=fractions,
            vals=vals,
            fit_mat=fit_mat,
            coefficients = coeffs,
            lstsq_info = res[1],
        )
        return coeffs[0], fd
    else:
        return coeffs[0]


def sorting_norm(
        spectrum, 
        min_frac=0.75, 
        max_frac=0.97,
        degree=None,
        n_split=None,
        n_samples=15,
        mask = None,
        h_mask_radius=-3.5,
        apply_h_mask = True,
        structure_vec=None,
        n_iter=3,
):
    npts = len(spectrum)
    assert n_samples >= 2
    assert min_frac > 0.0
    assert max_frac < 1.0
    
    if hasattr(spectrum, "flux"):
        flux = spectrum.flux
    else:
        flux = spectrum
    
    if mask is None:
        if hasattr(spectrum, "ivar"):
            mask = spectrum.ivar > 0
        else:
            mask = np.ones(npts, dtype=bool)
        
        secder = np.abs(scipy.gradient(scipy.gradient(flux)))
        secder_rank = secder.argsort().argsort()
        mask *= secder_rank < int(npts*0.75)+1
    
    if hasattr(spectrum, "wvs"):
        wavelengths = spectrum.wvs
        if apply_h_mask:
            mask*= tmb.hydrogen.get_H_mask(
                wavelengths, h_mask_radius)
        else:
            raise ValueError("A spectrum object must be supplied in order to apply an H mask!")
    
    npts_eff = int(np.sum(mask))
    if n_split is None:
        n_split = max(1, int(np.sqrt(npts)//4))
        
    pts_per = int(npts/(n_split+1))
    split_idxs = np.arange(pts_per, npts, pts_per)[:-1]
    
    split_fluxes = np.split(flux, split_idxs)
    split_masks = np.split(mask, split_idxs)
    
    split_medians = np.zeros(len(split_fluxes))
    n_regions = len(split_medians)
    
    for split_idx, split_flux in enumerate(split_fluxes):
        split_medians[split_idx] = adjusted_median(
            split_flux, 
            min_frac=min_frac, 
            max_frac=max_frac, 
            mask=split_masks[split_idx], 
            n_samples=n_samples, 
            contaminant_sign=-1
        )
    
    valid_med_mask = np.logical_not(np.isnan(split_medians))
    n_regions = int(np.sum(valid_med_mask))
    
    if degree is None:
        degree = int(np.sqrt(n_regions+1))-1
        degree = min(degree, 8)
    
    if degree > 1:
        full_x = np.linspace(-1, 1, npts)
        split_x = np.split(full_x, split_idxs)
        x_avgs = np.array([np.mean(x[m]) for x, m in zip(split_x, split_masks)])
        
        x_avgs = x_avgs[valid_med_mask]
        split_medians = split_medians[valid_med_mask]
        coeffs = np.polyfit(x_avgs, y=split_medians, deg=degree)
        pval = np.polyval(coeffs, full_x)
        return pval
    else:
        return np.repeat(np.mean(split_medians[valid_med_mask]), npts)


def median_bootstrap_norm(
        spectrum,
        degree=6,
        n_bootstrap=100,
        fraction=0.2,
        mask=None,
        reduce_func=None,
        return_realizations=False,
):
    npts = len(spectrum)
    if mask is None:
        if hasattr(spectrum, "ivar"):
            mask = spectrum.ivar > 0
        else:
            mask = np.ones(npts, dtype=bool)
    x = np.linspace(-1, 1, npts)
    if hasattr(spectrum, "flux"):
        flux = spectrum.flux
    else:
        flux = spectrum
    
    fits = np.zeros((n_bootstrap, npts))
    for idx in range(n_bootstrap):
        cmask = mask*(np.random.random(npts) < fraction)
        pc = np.polyfit(x[cmask], flux[cmask], deg=degree)
        fits[idx] = np.polyval(pc, x)
    
    if reduce_func is None:
        reduce_func = lambda x: np.median(x, axis=0)
    
    norm = reduce_func(fits)
    if return_realizations:
        return norm, fits
    else:
        return norm


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
    norm: numpy.ndarray
    """
    #import matplotlib.pyplot as plt
    #import pdb; pdb.set_trace()
    wavelengths = spectrum.wvs
    pix_delta = np.median(scipy.gradient(wavelengths))
    pscale = partition_scale*pix_delta
    flux = spectrum.flux
    variance = spectrum.var
    inv_var = spectrum.ivar
    hmask = tmb.hydrogen.get_H_mask(wavelengths, H_mask_radius)
    good_mask = (inv_var > 0)*(flux > 0)*hmask
    #min_stats, fmask = detect_features(flux, variance, 
    #                                   reject_fraction=reject_fraction,
    #                                   mask_back_off=mask_back_off, 
    #                                   min_stats=min_stats)
    #fmask *= good_mask
    
    #generate a layered median mask.
    if mask_kwargs is None:
        mask_kwargs = {'n_layers':3, 'first_layer_width':201, 'last_layer_width':11, 'rejection_sigma':1.5} 
    fmask = tmb.utils.misc.layered_median_mask(flux, **mask_kwargs)*good_mask
    
    mwv = wavelengths[fmask].copy()
    mflux = flux[fmask].copy()
    minv = inv_var[fmask]
    break_wvs = tmb.utils.misc.min_delta_bins(mwv, min_delta=pscale, target_n=20*(poly_order+1)+10)
    pp_gen = tmb.utils.piecewise_polynomial.RCPPB(poly_order=poly_order, control_points=break_wvs)
    if norm_hints:
        hint_wvs, hint_flux, hint_inv_var = norm_hints
        mwv = np.hstack((mwv, hint_wvs))
        mflux = np.hstack((mflux, hint_flux))
        minv = np.hstack((minv, hint_inv_var))
    
    ppol_basis = pp_gen.get_basis(mwv).transpose()
    in_sig = 1.0/np.sqrt(minv)
    med_sig = np.median(in_sig)
    print(med_sig, "med sig")
    fit_coeffs = tmb.utils.misc.irls(
        ppol_basis, mflux, 
        sigma=in_sig, 
        max_iter=10, 
        resid_delta_thresh=1e-2
    )
    n_polys = len(break_wvs) + 1
    n_coeffs = poly_order+1
    out_coeffs = np.zeros((n_polys, n_coeffs))
    for basis_idx in range(pp_gen.n_basis):
        c_coeffs = pp_gen.basis_coefficients[basis_idx].reshape((n_polys, n_coeffs))
        out_coeffs += c_coeffs*fit_coeffs[basis_idx]
    continuum_ppol = tmb.utils.piecewise_polynomial.PiecewisePolynomial(out_coeffs, break_wvs, centers=pp_gen.centers, scales=pp_gen.scales, bounds=pp_gen.bounds)
    approx_norm = continuum_ppol(wavelengths)
    if overwrite:
        spectrum.norm = approx_norm
        spectrum.feature_mask = fmask
    return approx_norm
