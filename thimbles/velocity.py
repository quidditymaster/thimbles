# Standard Library
import time
import os

# 3rd Party
import numpy as np
import h5py

# Internal
from spectrum import Spectrum
from utils.misc import cross_corr
from utils.misc import local_gaussian_fit
from spectrum import WavelengthSolution
# ########################################################################### #

__all__ = ["template_rv_estimate"]

# ########################################################################### #

resource_dir = os.path.join(os.path.dirname(__file__), "resources")

def try_load_template():
    rv_standard = None
    try:
        hf = h5py.File(os.path.join(resource_dir, "g2_mp_giant.h5"), "r")
        rv_standard = Spectrum(np.array(hf["wv"]), np.array(hf["flux"]))
        hf.close()
    except Exception as e:
        print e
        print "there was an exception loading the template file, trying again"
    return rv_standard

for i in range(3): #number of times to retry if load fails
    rv_standard = try_load_template()
    if rv_standard is None:
        time.sleep(0.01+0.1*np.random.random())
    else:
        break

speed_of_light = 299792.458

def template_rv_estimate(spectra, template=rv_standard, delta_max=500, pix_poly_width=2):
    """use a template spectrum to estimate a common rv shift for a collection
    of spectra.
    
    spectra: list of spectrum objects
    template: a template spectrum to use to do the shifting to
        (expects a log-linear wavelength sampled template)
    
    """
    log_delta = np.log10(template.wv[-1]/template.wv[0])/len(template)
    #spec_wv_max = np.max([np.max(s.wv) for s in spectra])
    #wv_delta_max = (delta_max/speed_of_light)*spec_wv_max
    ccors = []
    for spec in spectra:
        wv_bnds = spec.wv[0], spec.wv[-1]
        wv_min = min(wv_bnds)
        wv_max = max(wv_bnds)
        spec_bounds = (wv_min, wv_max) 
        bounded_template = template.bounded_sample(spec_bounds)
        normed = spec.normalized()
        rebinned = normed.rebin(bounded_template.wv)
        max_pix_off = int(np.log10(1.0+delta_max/speed_of_light)/log_delta) + 1
        rebinned_med = np.median(rebinned.flux)
        template_med = np.median(bounded_template.flux)
        ccors.append(cross_corr(rebinned.flux-rebinned_med, bounded_template.flux-template_med, max_pix_off, overlap_adj=True))
    ccors = np.array(ccors)
    ccor_maxes = np.max(ccors, axis=1)
    normed_ccors = ccors/ccor_maxes.reshape((-1, 1))
    ccor_med = np.median(normed_ccors, axis=0)
    max_idx = np.argmax(ccor_med)
    frac_delta = np.power(10.0, (np.arange(len(ccor_med)) - (len(ccor_med)-1.0)/2.0)*log_delta)
    ccor_vels = (frac_delta-1.0)*speed_of_light
    #import pdb; pdb.set_trace()
    rv, ccor_sig, ccor_peak = local_gaussian_fit(ccor_med, max_idx, pix_poly_width, xvalues=ccor_vels)
    return rv
