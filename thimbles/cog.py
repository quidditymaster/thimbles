import numpy as np
import scipy
import scipy.integrate as integrate

from thimbles.profiles import voigt
from thimbles.utils import piecewise_polynomial as ppol
from thimbles.sqlaimports import *

from thimbles.modeling import Model, Parameter

def voigt_saturation_curve(
        gamma_ratio=0.1,
        min_saturation=-1.5, 
        max_saturation=4.0,
        n_sat=105,
        n_integ=301,
        asymptotic_extend=True,
        n_segments=13,
):
    log_sats = np.linspace(min_saturation, max_saturation, n_sat)
    sats = np.power(10.0, log_sats)
    
    #build the integration points
    x = np.zeros(n_integ+1)
    x[1:] = np.power(10.0, np.linspace(-1.5, 3, n_integ))
    
    #build the opacity profile
    opac_profile = voigt(x, 0.0, 1.0, gamma_ratio)
    
    #carry out the saturated profile integrations
    log_rews = np.zeros(n_sat)
    for sat_idx in range(n_sat):
        flux_deltas = 1.0-np.exp(-sats[sat_idx]*opac_profile)
        cur_rew = 2.0*integrate.simps(flux_deltas, x=x)
        log_rews[sat_idx] = np.log10(cur_rew)
    
    if asymptotic_extend:
        #add slope 1 points at the bottom and slope 1/2 points at top
        n_extend = 25
        lower_deltas = np.linspace(3.0, 1.0, n_extend)
        low_sats = min_saturation - lower_deltas
        upper_deltas = np.linspace(1.0, 3.0, n_extend)
        high_sats = max_saturation + upper_deltas
        log_sats = np.hstack((low_sats, log_sats, high_sats))
        low_asymp = log_rews[0] - lower_deltas#np.ones(n_extend)
        high_asymp = log_rews[-1] + 0.5*upper_deltas#np.repeat(0.5, n_extend)
        #slope = np.hstack((low_asymp, slope, high_asymp))
        log_rews = np.hstack((low_asymp, log_rews, high_asymp))
        cpoints = np.linspace(min_saturation-0.2, max_saturation+0.2, n_segments)
    else:
        cpoints = np.linspace(min_saturation, max_saturation, n_segements+2)
        cpoints = cpoints[1:-1]
    
    constrained_ppol = ppol.RCPPB(poly_order=2, control_points=cpoints)
    linear_basis = constrained_ppol.get_basis(log_sats)
    n_polys = len(cpoints) + 1
    n_coeffs = 3
    out_coeffs = np.zeros((n_polys, n_coeffs))
    fit_coeffs = np.linalg.lstsq(linear_basis.transpose(), log_rews)[0]
    for basis_idx in range(constrained_ppol.n_basis):
        c_coeffs = constrained_ppol.basis_coefficients[basis_idx].reshape((n_polys, n_coeffs))
        out_coeffs += c_coeffs*fit_coeffs[basis_idx]
    
    fit_quad = ppol.PiecewisePolynomial(out_coeffs, control_points=cpoints)
    
    cog= ppol.InvertiblePiecewiseQuadratic(
        fit_quad.coefficients,
        fit_quad.control_points,
        centers=fit_quad.centers,
        scales=fit_quad.scales
    )
    return cog

class InterpolatedSCoG(object):
    
    def __init__(
        self,
        min_log_gamma_ratio=-4,
        max_log_gamma_ratio=1,
        n_curves=4,
    ):
        lgrats = np.linspace(min_log_gamma_ratio, max_log_gamma_ratio, n_curves)
        grats = np.power(10, lgrats)
        self.scogs = [voigt_saturation_curve(cgr) for cgr in grats]
        self.lgr_delta = lgrats[1]-lgrats[0]
        self.n_curves = n_curves
        self.min_log_gamma_ratio = min_log_gamma_ratio
        self.max_log_gamma_ratio = max_log_gamma_ratio
    
    def __call__(self, log_tau, gamma_ratio):
        log_gratio = np.log10(gamma_ratio)
        output = np.zeros(log_tau.shape)
        gratio_frac_idx = (log_gratio-self.min_log_gamma_ratio)/self.lgr_delta
        #deal with the gamma ratios off the small end
        mask = gratio_frac_idx <= 0
        output[mask] = self.scogs[0](log_tau[mask])
        for i in range(self.n_curves-1):
            mask = (gratio_frac_idx >= i) * (gratio_frac_idx < i+1)
            alphas = gratio_frac_idx[mask]-i
            cltau = log_tau[mask]
            output[mask] = (1-alphas)*self.scogs[i](cltau) + alphas*self.scogs[i+1](cltau)
        #deal with gamma ratios off the big end
        mask = gratio_frac_idx >= (self.n_curves - 1)
        lg_delta_mag = log_gratio[mask] - self.max_log_gamma_ratio
        gam_adjust = 0.5*lg_delta_mag/(1.0+np.power(10.0, -(log_tau[mask]-lg_delta_mag)))
        output[mask] = self.scogs[-1](log_tau[mask]) + gam_adjust
        return output
    
    def inverse(self, log_tau, gamma_ratio):
        log_gratio = np.log10(gamma_ratio)
        output = np.zeros(log_tau.shape)
        gratio_frac_idx = (log_gratio-self.min_log_gamma_ratio)/self.lgr_delta
        #deal with the gamma ratios off the small end
        mask = gratio_frac_idx <= 0
        output[mask] = self.scogs[0].inverse(log_tau[mask])
        for i in range(self.n_curves-1):
            mask = (gratio_frac_idx >= i) * (gratio_frac_idx < i+1)
            alphas = gratio_frac_idx[mask]-i
            cltau = log_tau[mask]
            output[mask] = (1-alphas)*self.scogs[i].inverse(cltau) + alphas*self.scogs[i+1].inverse(cltau)
        #deal with gamma ratios off the big end
        mask = gratio_frac_idx >= (self.n_curves - 1)
        lg_delta_mag = log_gratio[mask] - self.max_log_gamma_ratio
        gam_adjust = 0.5*lg_delta_mag/(1.0+np.power(10.0, -(log_tau[mask]-lg_delta_mag)))
        output[mask] = self.scogs[-1].inverse(log_tau[mask]- gam_adjust)
        return output



class PseudoStrengthModel(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"PseudoStrengthModel"
    }
    
    #class attributes
    fallback_correction = -3.0
    
    def __init__(
            self,
            output_p,
            transition_indexer,
            ion_correction,
            teff,
    ):
        self.output_p=output_p
        self.add_parameter("transition_indexer", transition_indexer)
        self.add_parameter("ion_correction", ion_correction)
        self.add_parameter("teff", teff)
    
    def __call__(self, override=None):
        vdict = self.get_vdict(override)
        transition_indexer = vdict[self.inputs["transition_indexer"]]
        transitions = transition_indexer.transitions
        ion_correction = vdict[self.inputs["ion_correction"]]
        teff = vdict[self.inputs["teff"]]
        adj_pst = []
        for t in transitions:
            uncor_pst = t.pseudo_strength(teff=teff)
            ion_cor = ion_correction.get(t.ion, self.fallback_correction)
            adj_pst.append(uncor_pst+ion_cor)
        return np.array(adj_pst)


class SaturationOffsetModel(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"SaturationOffsetModel",
    }
    
    def __init__(
            self,
            output_p,
            teff,
            coeff_dict,
    ):
        self.output_p=output_p
        self.add_parameter("teff", teff)
        self.add_parameter("coeff_dict", coeff_dict)
    
    def __call__(self, override=None):
        vdict = self.get_vdict(override)
        teff = vdict[self.inputs["teff"]]
        coeff_dict = vdict[self.inputs["coeff_dict"]]
        offset = coeff_dict["offset"]
        
        teff_coeffs = coeff_dict["teff"]        
        for i in range(len(teff_coeffs)):
            offset = teff_coeffs[i]*teff**(i+1)
        return offset


class SaturationModel(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"SaturationModel"
    }
    
    def __init__(
            self,
            output_p,
            offset,
            saturation_curve,
            pseudostrengths,
    ):
        self.output_p = output_p
        self.add_parameter("offset", offset)
        self.add_parameter("saturation_curve", saturation_curve)
        self.add_parameter("pseudostrengths", pseudostrengths)
    
    def __call__(self, override=None):
        vdict = self.get_vdict(override)
        pst = vdict[self.inputs["pseudostrengths"]]
        sat_curve = vdict[self.inputs["saturation_curve"]]
        offset = vdict[self.inputs["offset"]]
        saturations = sat_curve.inverse(sat_curve(pst) + offset)
        return saturations 


class SaturationCurveModel(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"SaturationCurveModel",
    }
    
    def __init__(self, output_p, sigmas, gammas):
        self.output_p = output_p
        self.add_parameter("sigmas", sigmas)
        self.add_parameter("gammas", gammas)
    
    def __call__(self, override=None):
        vdict = self.get_vdict(override)
        sigmas = vdict[self.inputs["sigmas"]]
        gammas = vdict[self.inputs["gammas"]]
        mean_gamma_ratio = np.mean(gammas/sigmas)
        return voigt_saturation_curve(mean_gamma_ratio)

class SimpleCog(object):
    
    def __init__(
            self,
            saturation_curve,
            squeeze_factor,
    ):
        self.sat_curve = saturation_curve
        self.squeeze_factor = squeeze_factor
    
    def __call__(
            self,
            pseudostrengths,
    ):
        return self.sat_curve(pseudostrengths) - self.squeeze_factor
    
    def inverse(
            self,
            log_rews,
    ):
        return self.sat_curve.inverse(log_rews+self.squeeze_factor)
    
    def deriv(
            self,
            *args,
            **kwargs
    ):
        return self.sat_curve.deriv(*args, **kwargs)


class SimpleCogModel(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"SimpleCogModel",
    }
    
    def __init__(
            self,
            output_p,
            gamma_ratio,
            squeeze_factor,
    ):
        self.output_p = output_p
        self.add_parameter("gamma_ratio", gamma_ratio)
        self.add_parameter("squeeze_factor", squeeze_factor)
    
    def __call__(self, override=None):
        vdict = self.get_vdict(override)
        gamma_ratio = vdict[self.inputs["gamma_ratio"]]
        squeeze_factor = vdict[self.inputs["squeeze_factor"]]

        return SimpleCog(
            saturation_curve=voigt_saturation_curve(gamma_ratio),
            squeeze_factor=squeeze_factor,
        )
