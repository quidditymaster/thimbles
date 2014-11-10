# Standard Library
import time
import os
import warnings

# 3rd Party
import numpy as np
import h5py

# Internal
from spectrum import Spectrum
from utils.misc import cross_corr
from utils.misc import local_gaussian_fit
from spectrum import WavelengthSolution
from thimbles.tasks import task
# ########################################################################### #

from thimbles import resource_dir

__all__ = ["template_cc_rv"]

# ########################################################################### #

def try_load_template():
    rv_standard = None
    try:
        hf = h5py.File(os.path.join(resource_dir, "g2_mp_giant.h5"), "r")
        rv_standard = Spectrum(np.asarray(hf["wv"]), np.asarray(hf["flux"]))
        hf.close()
    except Exception as e:
        warnings.warn(str(e)+"\nthere was an exception loading the template file, trying again")
    return rv_standard

for i in range(3): #number of times to retry if load fails
    rv_standard = try_load_template()
    if rv_standard is None:
        time.sleep(0.01+0.05*np.random.random())
    else:
        break

speed_of_light = 299792.458

#TODO: make a general cross correlator task. one where we build a cross correlation stack and one where we do the whole spectrum at once

@task()
def template_cc_rv(spectra, template=rv_standard, max_velocity=500, pix_poly_width=2):
    """use a template spectrum to estimate a common rv shift for a collection
    of spectra.
    
    spectra: list of spectrum objects
      the spectra to cross correlate
    template: Spectrum 
      the spectrum to cross correlate against
      the template should be log-linear wavelength sampled.
    
    """
    log_delta = np.log10(template.wv[-1]/template.wv[0])/len(template)
    ccors = []
    for spec in spectra:
        wv_bnds = spec.wv[0], spec.wv[-1]
        wv_min = min(wv_bnds)
        wv_max = max(wv_bnds)
        spec_bounds = (wv_min, wv_max)
        bounded_template = template.bounded_sample(spec_bounds)
        normed = spec.normalized()
        rebinned = normed.rebin(bounded_template.wv)
        max_pix_off = int(np.log10(1.0+max_velocity/speed_of_light)/log_delta) + 1
        rebinned_med = np.median(rebinned.flux)
        template_med = np.median(bounded_template.flux)
        ccors.append(cross_corr(rebinned.flux-rebinned_med, bounded_template.flux-template_med, max_pix_off, overlap_adj=True))
    ccors = np.asarray(ccors)
    ccor_maxes = np.max(ccors, axis=1)
    normed_ccors = ccors/ccor_maxes.reshape((-1, 1))
    ccor_med = np.median(normed_ccors, axis=0)
    max_idx = np.argmax(ccor_med)
    frac_delta = np.power(10.0, (np.arange(len(ccor_med)) - len(ccor_med)/2.0)*log_delta)
    ccor_vels = (frac_delta-1.0)*speed_of_light
    rv, ccor_sig, ccor_peak = local_gaussian_fit(ccor_med, max_idx, pix_poly_width, xvalues=ccor_vels)
    return rv
