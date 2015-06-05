
import numpy as np
import thimbles as tmb
from thimbles.tasks import task
import scipy

@task()
def make_wavelength_standard(
        spectra, 
        min_wv=None, 
        max_wv=None, 
        sampling_resolution=None
):
    """make a log-linear wavelength solution to use as a common wavelength solution for many spectra. 
    """
    if min_wv is None:
        min_wv = np.min([np.min(spec.wvs) for spec in spectra])
    if max_wv is None:
        max_wv = np.max([np.max(spec.wvs) for spec in spectra])
    if sampling_resolution is None:
        sampling_resolution = np.median([np.median(spec.wvs/scipy.gradient(spec.wvs)) for spec in spectra])
    npts_coadd = int(np.log(max_wv/min_wv)*sampling_resolution + 1)
    coadd_wvs = np.power(10.0, np.linspace(np.log10(min_wv), np.log10(max_wv), npts_coadd))
    print("coadd wvs", coadd_wvs)
    return coadd_wvs


@task()
def coadd_simple(spectra, coadd_wvs=None, sampling_mode="rebin", pre_normalize=False):
    if coadd_wvs is None:
        coadd_wvs = make_wavelength_standard(spectra)
    npts_coadd = len(coadd_wvs)
    coadd_flux=np.zeros(npts_coadd)
    weights=np.zeros(npts_coadd)
    for spec in spectra:
        if pre_normalize:
            spec = spec.normalized()
        sampled_spec=spec.sample(coadd_wvs, mode=sampling_mode)
        cweights = sampled_spec.ivar
        coadd_flux += sampled_spec.flux*cweights
        weights += cweights
    coadd_flux /= weights
    coadd_spec = tmb.Spectrum(coadd_wvs, coadd_flux)
    return coadd_spec
    
    
