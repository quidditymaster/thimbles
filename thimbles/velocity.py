# Standard Library
import time
import os
import warnings

# 3rd Party
import numpy as np
import scipy.ndimage
import h5py

# Internal
import thimbles as tmb
from thimbles.utils.misc import local_gaussian_fit
from .modeling import Model, Parameter, Estimator
from .sqlaimports import *

from .tasks import task
from . import speed_of_light
from . import resource_dir

__all__ = ["template_cc_rv"]

#TODO: make a general cross correlator task. one where we build a cross correlation stack and one where we do the whole spectrum at once


def cross_corr(arr1, arr2, offset_number, overlap_adj=False):
    """cross correlate two arrays of the same size.
    correlating over a range of pixel offsets from -offset_number to 
    +offset_number and returning a 2*offset_number+1 length asarray. 
    To adjust for differences in the number of pixels overlapped if 
    overlap_adj==True we divide the result by the number of pixels
    included in the overlap of the two arrays.
    """
    assert len(arr1) == len(arr2)
    npts = len(arr1)
    offs = int(offset_number)
    cor_out = np.zeros(2*offs+1)
    offsets = list(range(-offs, offs+1))
    for offset_idx in range(len(offsets)):
        coff = offsets[offset_idx]
        lb1, ub1 = max(0, coff), min(npts, npts+coff)
        lb2, ub2 = max(0, -coff), min(npts, npts-coff)
        cur_corr = np.sum(arr1[lb1:ub1]*arr2[lb2:ub2])
        if overlap_adj:
            n_overlap = min(ub1, ub2) - max(lb1, lb2)
            cur_corr /= float(n_overlap)
        cor_out[offset_idx] = cur_corr
    return cor_out

_rv_standard = None

@task()
def template_cc_rv(
        spectra, 
        template=None, 
        max_velocity=250,
        avg_width=20, 
        normalize_diff=True, 
        pix_poly_width=2
):
    """use a template spectrum to estimate a common rv shift for a collection
    of spectra.
    
    spectra: list of spectrum objects
      the spectra to cross correlate
    template: Spectrum 
      the spectrum to cross correlate against
      the template should be log-linear wavelength sampled.
    
    """
    if template is None:
        global _rv_standard
        if _rv_standard is None:
            _rv_standard = try_load_template()
        template = _rv_standard
    log_delta = np.log10(template.wv[-1]/template.wv[0])/(len(template)-1)
    ccors = []
    for spec in spectra:
        wv_bnds = spec.wv[0], spec.wv[-1]
        wv_min = min(wv_bnds)
        wv_max = max(wv_bnds)
        spec_bounds = (wv_min, wv_max)
        bounded_template = template.sample(spec_bounds, mode="bounded")
        template_avg = scipy.ndimage.filters.gaussian_filter(bounded_template.flux, sigma=avg_width)
        template_diff = bounded_template.flux-template_avg
        interped_spec = spec.sample(bounded_template.wv, mode="interpolate")
        avg_interped = scipy.ndimage.filters.gaussian_filter(interped_spec.flux, sigma=avg_width)
        interped_diff = interped_spec.flux-avg_interped
        
        if normalize_diff:
            template_diff /= np.sqrt(np.sum(template_diff**2))
            interped_diff /= np.sqrt(np.sum(interped_diff**2))
        
        max_pix_off = int(np.log10(1.0+max_velocity/speed_of_light)/log_delta) + 1
        ccors.append(cross_corr(interped_diff, template_diff, max_pix_off, overlap_adj=True))
    ccors = np.asarray(ccors)
    ccor_maxes = np.max(ccors, axis=1)
    normed_ccors = ccors/ccor_maxes.reshape((-1, 1))
    ccor_med = np.median(normed_ccors, axis=0)
    frac_delta = np.power(10.0, (np.arange(len(ccor_med)) - len(ccor_med)/2.0)*log_delta)
    ccor_vels = (frac_delta-1.0)*speed_of_light
    rv, ccor_sig, ccor_peak = local_gaussian_fit(
        ccor_med, 
        fit_width=pix_poly_width, 
        xvalues=ccor_vels
    )
    return rv


class RVShiftModel(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"RVShiftModel",
    }
    
    def __init__(self, output_p, wvs_p, rv_params):
        self.output_p = output_p
        self.add_input("wvs", wvs_p)
        for rv_p in rv_params:
            self.add_input("rv_params", rv_p, is_compound=True)
    
    def __call__(self, vprep=None):
        vdict = self.get_vdict(vprep)
        wvs = vdict[self.inputs["wvs"]]
        rv_tot = sum([vdict[p] for p in self.inputs["rv_params"]])
        return wvs*(1.0-rv_tot/tmb.speed_of_light)


class CrossCorrelationRelativeVelocityEstimator(Estimator):
    _id = Column(Integer, ForeignKey("Estimator._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"CrossCorrelationRelativeVelocityEstimator",
    }
    
    def __init__(self, source,):
        rv_p = source.context["rv"]
        flux_params = []
        wvs_params = []
        delta_helio_params = []
        for spec in source.spectroscopy:
            pointing = spec.observation.pointing
            delta_helio_p = pointing.context["delta_helio"]
            delta_helio_params.append(delta_helio_p)
            flux_params.append(spec.flux_p)
            wvs_p = spec.context["obs_wvs"] = None
            wvs_params.append()
        
        Estimator.__init__(self)    
    
    def __call__(self):
        rv_p = template_cc_rv()


def try_load_template():
    rv_standard = None
    try:
        hf = h5py.File(os.path.join(resource_dir, "g2_mp_giant.h5"), "r")
        rv_standard = tmb.spectrum.Spectrum(np.asarray(hf["wv"]), np.asarray(hf["flux"]))
        hf.close()
        if tmb.opts["wavelength_medium"] == "vacuum":
            vac_wvs = tmb.utils.misc.air_to_vac(rv_standard.wvs)
            vac_spec = tmb.spectrum.Spectrum(vac_wvs, rv_standard.flux)
            log_lin_wvs = np.exp(np.linspace(np.log(vac_wvs[0]), np.log(vac_wvs[-1]), len(vac_spec)))
            rv_standard = vac_spec.sample(log_lin_wvs)
    except Exception as e:
        warnings.warn(str(e)+"\nthere was an exception loading the template file")
    return rv_standard
