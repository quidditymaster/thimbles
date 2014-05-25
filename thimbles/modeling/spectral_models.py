import numpy as np
import scipy
import scipy.optimize

import thimbles as tmb

from ..spectrum import Spectrum
from . import factor_models
from . import predictors
from predictors import GaussianPredictor

__all__ = ["fit_single_voigt", "quick_quadratic_fit"]

class WavelengthScaledGaussianPredictor(predictors.Predictor):
    """provides a predictor of the form mean*wavelength +/- wavelength*sigma
    """
    
    def __init__(self, mean, sigma):
        self.mean = mean
        self._sigma = sigma
    
    def predict(self, feature):
        wv = feature.wv
        return self.mean*wv
    
    def sigma(self, feature):
        wv = feature.wv
        return self._sigma*wv

def feature_predictors_from_ensemble(features, verbose=False):
    """generates a dictionary of the form
    {"offset":offset_predictor, "sigma":sigma_predictor, ...}
    
    where the predictors are generated from the center and spread statistics
    of the feature ensemble. 
    
    features: list
     the feature objects
    
    """
    lparams = np.asarray([f.profile.get_parameters() for f in features])
    cent_wvs = np.asarray([f.wv for f in features])
    rel_norms = np.asarray([f.relative_continuum for f in features])
    delta_wvs = np.asarray([np.mean(scipy.gradient(f.data_sample.wv)) for f in features])
    dwv_over_wv = delta_wvs/cent_wvs
    med_inv_r = np.median(dwv_over_wv)
    
    sig_over_wv = lparams[:, 1]/cent_wvs
    sig_med = np.median(sig_over_wv)
    sig_mad = np.median(np.abs(sig_over_wv-sig_med))
    if verbose:
        print "sigma median", sig_med, "sigma mad", sig_mad
    
    vel_offs = lparams[:, 0]/cent_wvs
    vel_med = np.median(vel_offs)
    vel_mad = np.median(np.abs(vel_offs - vel_med))
    if verbose:
        print "velocity median", vel_med, "velocity mad", vel_mad
    
    gam_med = np.median(np.abs(lparams[:, 2]))
    gam_mad = np.median(np.abs(lparams[:, 2]-gam_med))
    if verbose:
        print "gamma median", gam_med, "gamma mad", gam_mad
    
    rel_med = np.median(rel_norms)
    rel_mad = np.median(np.abs(rel_norms-rel_med))
    if verbose:
        print "rel_norm median", gam_med, "rel_norm mad", gam_mad
    
    predictors = {}
    
    offset_predictor = WavelengthScaledGaussianPredictor(vel_med, 1.4*vel_mad)
    sigma_predictor = WavelengthScaledGaussianPredictor(sig_med, 1.4*sig_mad + 0.5*med_inv_r)
    gamma_predictor = GaussianPredictor(gam_med, 1.4*gam_mad+0.1*np.median(delta_wvs))
    rel_norm_predictor = GaussianPredictor(1.0, 0.01)
    
    predictors["offset"] = offset_predictor
    predictors["sigma"] = sigma_predictor
    predictors["gamma"] = gamma_predictor
    predictors["rel_norm"] = rel_norm_predictor
    
    return predictors

def feature_residuals_factory(feature, predictors):
    def resids(pvec):
        bspec = feature.data_sample
        wvs = bspec.wv
        flux = bspec.flux
        ew, rel_norm, offset, sigma, gamma = pvec
        prof_params = pvec[2:]
        prof = feature.profile(wvs, prof_params)
        model = rel_norm*bspec.norm*(1.0-ew*prof)
        resids = flux-model
        pred_off = predictors["offset"].predict(feature)
        pred_sig = predictors["sigma"].predict(feature)
        pred_gam = predictors["gamma"].predict(feature)
        pred_rn  = predictors["rel_norm"].predict(feature)
        pred_vec = np.asarray([pred_off-offset, pred_sig-sigma, pred_gam-gamma, pred_rn-rel_norm])
        return np.hstack((resids ,pred_vec))
    return resids

def fit_single_voigt(feature, fit_width=None, other_features=None, predictors=None):
    """a convenience function for (re)fitting a single feature in a spectrum
    keeping the normalization and extracted flux from other features fixed.
    """
    
    lparams = feature.profile.get_parameters()
    offset, sigma, gamma = lparams
    cent_wv = feature.wv
    
    resid_func = feature_residuals_factory(feature, predictors=predictors)
    
    cwv = feature.wv
    fit_bounds = (cwv-fit_width, cwv+fit_width)
    bspec = feature.data_sample.bounded_sample(fit_bounds)
    start_p = feature.profile.get_parameters()
    start_p[1:] = np.abs(start_p[1:])
    guessv = np.hstack((feature.eq_width, 1.0, start_p))
    fit_res = scipy.optimize.leastsq(resid_func, guessv)
    fit = fit_res[0]
    fit[3:] = np.abs(fit[3:])
    feature.profile.set_parameters(fit[2:])
    feature.relative_continuum = fit[1]
    feature.set_eq_width(fit[0])
    return feature

def ensemble_feature_fit(features, fit_width=None):
    if fit_width is None:
        fit_width = 0.3
    #import pdb; pdb.set_trace()
    fpreds = feature_predictors_from_ensemble(features, verbose=True)
    for feature in features:
        fit_single_voigt(feature, fit_width, predictors=fpreds)

def quick_quadratic_fit(features, npix_expand=2):
    """does a fast approximate feature fit by expanding a quadratic polynomial
    around the nearest local minimum contained in the features.data_sample
    This function should be used only to provide quick first guess fits.
    Note that the values of the features passed in will be modified.
    """
    for feature in features:
        #import pdb; pdb.set_trace()
        bspec = feature.data_sample
        wvs = bspec.wv
        cent_wv = feature.wv
        flux = bspec.flux
        minima = tmb.utils.misc.get_local_maxima(-flux)
        if np.sum(minima) == 0:
            feature.set_eq_width(0.0)
            continue
        minima_idxs = np.where(minima)[0]
        minima_wvs = wvs[minima_idxs]
        best_minimum_idx = np.argmin(np.abs(minima_wvs-cent_wv))
        closest_idx = minima_idxs[best_minimum_idx]
        fit_kw={"peak_idx":closest_idx, "fit_width":npix_expand, "xvalues":wvs}
        gfit_res = tmb.utils.misc.local_gaussian_fit(flux, **fit_kw)
        fit_center, fit_sigma, fit_y = gfit_res
        norm_flux = bspec.norm[closest_idx]
        depth = (norm_flux - flux[closest_idx])/norm_flux
        offset = wvs[closest_idx]-cent_wv
        avg_delta = np.mean(scipy.gradient(wvs))
        new_sigma = max(0.5*avg_delta, np.abs(fit_sigma))
        new_sigma = min(5.0*avg_delta, new_sigma)
        new_params = [offset, new_sigma, 0.0]
        feature.profile.set_parameters(new_params)
        feature.set_depth(depth)
    return features

class FeatureModel(factor_models.LocallyLinearModel):
    
    def __init__(self):
        raise NotImplemented

class VoigtSumModel(FeatureModel):
    
    def __init__(self, sample_wvs, features):
        """represents the spectrum as a linear sum of voigt features
        subtracted from a vector of ones.
        """
        self.sample_wvs = sample_wvs
        self.features = features

class SpectrographModel(factor_models.LocallyLinearModel):
    
    def __init__(self):
        pass

class SamplingModel(factor_models.LocallyLinearModel):
    
    def __init__(self):
        pass
    
class ReddeningModel(factor_models.LocallyLinearModel):
    
    def __init__(self):
        pass

class BroadeningModel(factor_models.LocallyLinearModel):
    
    def __init__(self):
        pass

class ContinuumModel(factor_models.LocallyLinearModel):
    
    def __init__(self):
        pass

class SpectralModel(Spectrum, factor_models.LocallyLinearModel):
    
    def __init__(self, wvs, model_wvs, models, ):
        pass
