
import numpy as np
import scipy
import scipy.special
import scipy.optimize
import scipy.ndimage
import scipy.fftpack as fftpack
import thimbles as tmb
from thimbles import speed_of_light
from functools import reduce
sqrt2pi = np.sqrt(2*np.pi)
sqrt2 = np.sqrt(2)

profile_functions = {}

def gauss(wvs, center, g_width):
    return np.exp(-0.5*((wvs-center)/g_width)**2)/np.abs(sqrt2pi*g_width)

profile_functions["gaussian"] = gauss

def half_gauss(wvs, center, l_width, r_width):
    #pick the average of left and right norms
    avg_norm = 2.0/np.abs(sqrt2pi*(l_width + r_width))
    sig_vec = np.where(wvs< center, l_width, r_width)
    return avg_norm*np.exp(-(wvs-center)**2/(2*sig_vec**2))

profile_functions["half_gaussian"] = half_gauss


def voigt(wvs, center, g_width, l_width):
    "returns a voigt profile with gaussian sigma g_width and lorentzian width l_width."
    g_w = np.abs(g_width)
    l_w = np.abs(l_width)
    if l_w == 0:
        if g_w == 0:
            g_w = 1e-10
        return gauss(wvs, center, g_w)
    elif g_w == 0:
        g_w = 1e-5*l_w
    z = ((wvs-center)+l_w*1j)/(g_w*sqrt2)
    cplxv = scipy.special.wofz(z)/(g_w*sqrt2pi)
    return cplxv.real

profile_functions["voigt"] = voigt

#TODO: implement the holtsmark distribution
def nnstark(wvs, center, stark_width):
    """nearest neighbor approximation to a stark broadening profile"""
    delta_wvs = (np.abs(wvs-center) + 1e-10)/np.abs(stark_width) #1e-10 to regularize the limit delta_lam->0  
    return 3.0/(4.0*stark_width)*np.power(delta_wvs, -5.0/2.0)*np.exp(-np.power(delta_wvs, -3.0/2.0))

profile_functions["nearest_neighbor_stark"] = nnstark

def rotational(wvs, center, vsini, limb_dark = 0):
    "for wavelengths in angstroms and v*sin(i) in km/s"
    ml = np.abs(vsini/3e5*center)
    eps = limb_dark
    deltas = wvs-center
    deltas = deltas * (np.abs(deltas) <= ml)
    indep_part = 2*(1-eps)*np.power(1-np.power(deltas/ml, 2.0), 0.5)
    limb_part = 0.5*np.pi*eps*(1-np.power(deltas/ml, 2.0))
    norm = np.pi*ml*(1-eps/3.0)
    result = (np.abs(wvs-center) <= ml)*(indep_part + limb_part)/norm
    if np.sum(np.abs(result)) == 0: #rotation too small to detect at this resolution
        print("rot_too small", len(wvs))
        nwvs = len(wvs)
        result[int(nwvs/2)] = 1.0
    return result

profile_functions["rotational"] = rotational

def _calc_prototype_rt_profile(frac_width, n_freqs):
    eta_rt = frac_width * n_freqs
    fft_freqs = fftpack.fftfreq(n_freqs, 1.0)
    #replace zero frequency with very small non-zero freq
    fft_freqs[0] = 1.0/n_freqs**2
    macro_fft = 1.0-np.exp(-(np.pi*eta_rt*fft_freqs)**2)
    macro_fft /= (np.pi*eta_rt*fft_freqs)**2
    full_profile = (fftpack.ifft(macro_fft).real).copy()
    #reorder the fft to put the peak at the center
    ordering_indexes = np.argsort(fft_freqs)
    
    fft_freqs[0] = 0.0
    fft_freqs = fft_freqs[ordering_indexes]/frac_width
    return fft_freqs, full_profile[ordering_indexes].copy()


_prototype_rt_freqs, _prototype_rt_prof = _calc_prototype_rt_profile(0.05, 2048)
_prototype_rt_interper = scipy.interpolate.interp1d(_prototype_rt_freqs, _prototype_rt_prof, fill_value=0.0, bounds_error=False)

def radial_tangential_macroturbulence(wvs, center, eta_rt, n_freqs=1024):
    delta_lam = (eta_rt/speed_of_light)*center
    delta_freqs = (wvs - center)/delta_lam
    return _prototype_rt_interper(delta_freqs)

profile_functions["radial_tangential_macroturbulence"] = radial_tangential_macroturbulence

def convolved_stark(wvs, center, g_width, l_width, stark_width):
    """a numerically simple model for the line shape of a hydrogen feature.
    a Voigt profile convolved with a nearest-neighbor stark"""
    if len(wvs) % 2 == 1:
        array_central_wv = wvs[int(len(wvs)/2)]
    else:
        array_central_wv = 0.5*(wvs[int(len(wvs)/2)] + wvs[int(len(wvs)/2)-1])
    centered_stark = nnstark(wvs, array_central_wv, stark_width)
    voigt_prof = voigt(wvs, center, g_width, l_width)
    return scipy.ndimage.filters.convolve(centered_stark, voigt_prof)


def compound_profile(wvs, center, sigma, gamma, vsini, limb_dark, vmacro, convolution_mode="discrete", normalize=True):
    """a helper function for convolving together the voigt, rotational, 
    and radial tangential macroturbulent profiles.
    
    Note: the convolution procedure may induce shifts, warps and
    other unphysical artefacts so always examine the output.
    In particular problems will crop up if wvs[len(wvs)//2] != center
    
    parameters:
    wvs: ndarray
     the wavelengths at which to sample the profile
    center: float
     the central wavelength of the profile 
     (make sure it is also the central wavelength of wvs)
    sigma: float
     gaussian sigma width
    gamma: float
     lorentz width
    vsini: float
     projected rotational velocity [km/s]
    limb_dark: float
     limb darkening coefficient (between 0 and 1)
    vmacro: float
     radial tangential macroturbulent velocity [km/s]
    convolution_mode: string
     "fft" do the convolution by multiplying together fourier transforms
     "discrete" do the convolution explicitly in wv space.
     Because the fft implicitly assumes the function wraps around from low
     to high wavelengths and the discrete convolution assumes that the 
     function is zero outside the given wvs the fft convolution will
     tend to be too high in the wings and the discrete too low, 
     pick your poison.
    normalize: bool
     if True then nomalize the result to have sum=1
    """
    all_profs = []
    if not ((sigma ==0) and (gamma==0)):
        vprof = voigt(wvs, center, sigma, gamma)
        all_profs.append(vprof)
    if not (vsini == 0):
        rotprof = rotational(wvs, center, vsini, limb_dark)
        all_profs.append(rotprof)
    if not (vmacro == 0):
        macroprof = radial_tangential_macroturbulence(wvs, center, vmacro)
        all_profs.append(macroprof)
    
    if convolution_mode == "fft":
        ffts = [fftpack.fft(prof) for prof in all_profs]
        fft_prod = reduce(lambda x, y: x*y, ffts)
        print("WARNING compound profile computed via fft has bugs!")
        prof = fftpack.ifft(fft_prod).real.copy()
    elif convolution_mode == "discrete":
        prof = reduce(lambda x, y: np.convolve(x, y, mode="same"), all_profs)
    else:
        raise ValueError("convolution mode {} is not recognized".format(convolution_mode))
    if normalize:
        dlam = scipy.gradient(wvs)
        prof /= np.sum(prof*dlam)
    return prof


def compound_profile_derivatives(
        wvs,
        sigma,
        gamma,
        vsini,
        limb_dark,
        vmacro,
        oversample_ratio=100.0,
        eps_frac=None,
):
    min_wv = wvs[0]
    max_wv = wvs[-1]
    assert max_wv > min_wv
    npts_sample = int(oversample_ratio*len(wvs))
    npts_sample += (npts_sample % 2) - 1
    sample_wvs = np.exp(np.linspace(np.log(min_wv), np.log(max_wv), npts_sample))
    center = sample_wvs[npts_sample//2]
    kw_dict = dict(sigma=sigma, gamma=gamma, vsini=vsini, limb_dark=limb_dark, vmacro=vmacro)
    if eps_frac is None:
        eps_frac = 0.05
    
    sample_spec = tmb.Spectrum(sample_wvs, np.ones(npts_sample), np.ones(npts_sample))
    junk, rebin_mat = sample_spec.sample(wvs, mode="rebin", return_matrix=True)
    
    deriv_dict = {}
    for pname in kw_dict:
        kw_val = kw_dict[pname]
        if kw_val == 0.0:
            deriv_dict[pname] = np.zeros(len(wvs))
            continue
        minus_kw = {}
        minus_kw.update(kw_dict)
        minus_kw[pname] = kw_dict[pname]*(1.0-eps_frac)
        minus_prof = compound_profile(sample_wvs, center, **minus_kw)
        plus_kw = {}
        plus_kw.update(kw_dict)
        plus_kw[pname] = kw_dict[pname]*(1.0+eps_frac)
        plus_prof = compound_profile(sample_wvs, center, **plus_kw)
        
        eps_val = kw_dict[pname]*2*eps_frac
        oversampled_deriv = (plus_prof-minus_prof)/eps_val
        deriv_dict[pname] = rebin_mat*oversampled_deriv
    return deriv_dict


def make_derivative_decoupling_kernel(
        target_parameter,
        wvs,
        sigma,
        gamma,
        vsini,
        limb_dark,
        vmacro,
        profile_noise=None,
        eps_fracs=None,
        symmetrize=True,
        k_cutoff=0.98,
):
    assert (len(wvs) % 2) == 1
    center_wv = wvs[len(wvs)//2]
    
    if profile_noise is None:
        profile_noise = {}
    
    if eps_fracs is None:
        eps_fracs = [0.05, 0.2]
    
    collected_dvecs = []
    for eps_frac_idx, eps_frac in enumerate(eps_fracs):
        dvecs = compound_profile_derivatives(
            wvs=wvs, 
            sigma=sigma, 
            gamma=gamma, 
            vsini=vsini, 
            limb_dark=limb_dark, 
            vmacro=vmacro, 
            eps_frac=eps_frac
        )
        
        #normalize the derivatives to have sum of squares == 1
        for pname in dvecs:
            cur_d_vec = dvecs[pname]
            norm_sum = np.sqrt(np.sum(cur_d_vec**2))
            if norm_sum > 0:
                cur_d_vec /= norm_sum
            dvecs[pname] = cur_d_vec
        collected_dvecs.append(dvecs)
    
    #build the covariance matrix
    npts = len(wvs)
    covar = np.diag(np.repeat(max(0.01, (1-k_cutoff))/np.sqrt(npts), npts))
    
    #add in the derivative 'noise'
    dvec_stack = []
    for dvecs in collected_dvecs:
        for pname in dvecs:
            if not pname == target_parameter:
                noise_weight = profile_noise.get(pname, 1.0)
                dvec_stack.append(noise_weight*dvecs[pname])
    dvec_stack = np.array(dvec_stack)
    svd_res = np.linalg.svd(dvec_stack, full_matrices=False)
    
    cum_var_frac = np.cumsum(svd_res[1])
    cum_var_frac /= cum_var_frac[-1]
    k_keep = 0
    for i in range(len(cum_var_frac)):
        k_keep += 1
        if cum_var_frac[i] > k_cutoff:
            break
    
    for k in range(k_keep):
        pnoise_vec = svd_res[2][k]
        var_weight = svd_res[1][k]
        covar += var_weight*np.outer(pnoise_vec, pnoise_vec)
    
    inv_var = np.linalg.pinv(covar)
    kernel_vec = np.dot(inv_var, dvecs[target_parameter])
    
    if symmetrize:
        #symmetrize by averaging over the kernels own mirror image
        kernel_vec = 0.5*(kernel_vec[::-1] + kernel_vec)
    
    info_dict = dict(
        profile_derivatives=dvec_stack,
        k_keep=k_keep,
        k_cutoff=k_cutoff,
        covar=covar,
        icovar=inv_var,
        kernel=kernel_vec,
        profile_noise=profile_noise,
        gamma=gamma,
        sigma=sigma,
        vsini=vsini,
        limb_dark=limb_dark,
        vmacro=vmacro,
        eps_fracs=eps_fracs,
        wvs=wvs,
    )
    
    return kernel_vec, info_dict


def uniformly_sampled_profile_matrix(
        wvs, 
        sigma_min, 
        sigma_max,
        gamma_min,
        gamma_max,
        vsini_min=0.1,
        vsini_max=100.0,
        limb_dark_min = 0.0,
        limb_dark_max = 1.0,
        vmacro_min = 0.1,
        vmacro_max = 20.0,
        n_samples = 1000,
):
    center_wv = np.mean(wvs)
    out_mat = np.zeros((n_samples, len(wvs)))
    sigmas = np.random.uniform(sigma_min, sigma_max, size=(n_samples, 1))
    gammas = np.random.uniform(gamma_min, gamma_max, size=(n_samples, 1))
    vsinis = np.random.uniform(vsini_min, vsini_max, size=(n_samples, 1))
    limb_darks = np.random.uniform(limb_dark_min, limb_dark_max, (n_samples, 1))
    vmacros = np.random.uniform(vmacro_min, vmacro_max, (n_samples, 1))
    params = np.hstack([sigmas, gammas, vsinis, limb_darks, vmacros])
    for i in range(n_samples):
        sig, gam, vsini, limb_dark, vmacro = params[i]
        out_mat[i] = compound_profile(wvs, center_wv, sig, gam, vsini, limb_dark, vmacro)
    return out_mat, params

