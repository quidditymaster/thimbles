# Standard Library
import time
import os

# 3rd Party
import numpy as np
import h5py

# Internal
from utils.misc import cross_corr
from utils.misc import local_gaussian_fit
from spectrum import WavelengthSolution
# ########################################################################### #

__all__ = ["RVTemplates","SimpleRVD"]

# ########################################################################### #

rv_template_h5file = os.path.dirname(__file__)
rv_template_h5file += "/../resources/templates/phoenix_templates_small.h5"

if not os.path.isfile(rv_template_h5file):
    print "You don't have the rv_template file"
    
speed_of_light = 299792.458

# ########################################################################### #

class RVTemplates:
    
    def __init__(self, template_file):
        #import pdb; pdb.set_trace()
        self.template_data = h5py.File(template_file, "r")
        self.resolutions = np.array(self.template_data["resolutions"])
    
    def get_best_resolution_index(self, input_spectrum):
        in_wvs = input_spectrum.get_wvs(frame="telescope")
        in_diffs = in_wvs[1:]-in_wvs[:-1]
        approx_res = np.median(0.5*(in_wvs[1:]+in_wvs[:-1])/in_diffs)
        res_idx = np.argmin(np.abs(self.resolutions-approx_res))
        #choose the next higher resolution step.
        res_idx = max(0, res_idx-1)
        return res_idx
    
    def correlate(self, input_spectrum, resolution_index=None, 
                                detrender=None, detrender_params = None,
                                max_rv=800):
        """
        input_spectrum: spectrum to correlate against
        max_rv: maximum radial velocity to consider in km/s
        detrender: a function to pass both the input spectrum and templates
            to remove trends in the data and remove the mean,
            if None a default smoothing is used.
            Ideally the detrender would return (flux/continuum - 1)
        """
        #estimate resolution of input_spectrum
        in_wvs = input_spectrum.get_wvs(frame="telescope")
        if resolution_index == None:
            res_idx = self.get_best_resolution_index(input_spectrum)
        else:
            res_idx = resolution_index
        res_string = "R_%d"%self.resolutions[res_idx]
        min_wv_in = np.min(in_wvs)
        max_wv_in = np.max(in_wvs)
        template_wvs = self.template_data["%s_wavelengths" % res_string]
        min_wv_temp = template_wvs[0]
        max_wv_temp = template_wvs[-1]
        log_delta = self.template_data["log_deltas"][res_idx]
        corr_max_wv = max_wv_in*(1.0+max_rv/speed_of_light)
        corr_min_wv = min_wv_in*(1.0-max_rv/speed_of_light)
        min_log_diff = np.log10(corr_min_wv)-np.log10(min_wv_temp)
        max_log_diff = np.log10(corr_max_wv)-np.log10(min_wv_temp)
        corr_min_idx = max(0, int(min_log_diff/log_delta))
        corr_max_idx = min(len(template_wvs)-1, int(max_log_diff/log_delta))
        cwvs = template_wvs[corr_min_idx:corr_max_idx]
        wv_soln = WavelengthSolution(cwvs)
        resampled_input = input_spectrum.rebin(wv_soln, frame="telescope")
        fluxes_data = self.template_data["%s_fluxes" % res_string]
        n_templates, npts = fluxes_data.shape
        max_pix_off = int(np.log10(1.0+max_rv/speed_of_light)/log_delta) + 1
        cross_correlations = np.empty((n_templates, 2*max_pix_off+1))
        
        #cross correlating
        for template_idx in xrange(n_templates):
            temp_flux = np.array(fluxes_data[template_idx, corr_min_idx:corr_max_idx])
            detrended_temp = detrender(temp_flux, np.ones(temp_flux.shape), *detrender_params)
            #to make cross correlation peaks comparable accross templates we need to norm
            temp_norm = np.sqrt(np.sum(detrended_temp**2))
            detrended_temp /= temp_norm
            correlation = cross_corr(detrended_resampled, detrended_temp, max_pix_off)
            cross_correlations[template_idx] = correlation
        return cross_correlations, log_delta
        

    def rv_shift(self, spectra):
        cross_correlations = []
        #pick the next highest resolution to cross correlate with
        best_res_idx = int(np.median([self.get_best_resolution_index(spec) for spec in spectra]))
        for spec_idx in range(len(spectra)):
            cspec = spectra[spec_idx]
            correlations, log_sigma = self.correlate(cmeas, resolution_index=best_res_idx, max_rv=800)
            cross_correlations.append(correlations)
        cross_correlations = np.array(cross_correlations)
        #cross_correlations should now have shape (n_meas, n_templates, n_offsets)
        #take the median over measurement_list first
        med_corr = np.median(cross_correlations, axis = 0)
            #then the maximum over offsets
            peak_vals = np.max(med_corr, axis = 1)
            best_template_idx = np.argmax(peak_vals)
            #for the best template find the associated rv
            best_ccor = med_corr[best_template_idx]
            peak_ccor_idx = np.argmax(best_ccor)
            g_center, g_sigma, pval = local_gaussian_fit(best_ccor, peak_ccor_idx)
            pixel_shift  = g_center-(len(best_ccor)//2)
            radial_velocity  = (10.0**(pixel_shift*log_sigma)-1.0)*speed_of_light
            print "setting rv %5.2f" % radial_velocity
            #import pdb; pdb.set_trace()
            for meas_idx in meas_idxs:
                param_list = star.measurement_list[meas_idx].wv_soln.get_parameter_list()
                rvparam ,= param_list[tagger("radial velocity")]
                rvparam.set(self.value_priority, radial_velocity)
