import numpy as np
import scipy
from thimbles.tasks import task
from thimbles.profiles import voigt
import scipy.integrate as integrate
from thimbles.utils import piecewise_polynomial as ppol

@task()
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
    x[1:] = np.power(10.0, np.linspace(-2, 4, n_integ))
    
    #build the opacity profile
    opac_profile = voigt(x, 0.0, 1.0, gamma_ratio)
    
    #carry out the saturated profile integrations
    log_rews = np.zeros(n_sat)
    for sat_idx in range(n_sat):
        flux_deltas = 1.0-np.exp(-sats[sat_idx]*opac_profile)
        cur_rew = 2.0*integrate.trapz(flux_deltas, x=x)
        log_rews[sat_idx] = np.log10(cur_rew)
    
    #find the derivative of the saturation curve
    #slope = scipy.gradient(log_rews)/scipy.gradient(log_sats)
    
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
        cpoints = cpoints[:-1]
    
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
        
    cog= ppol.InvertiblePiecewiseQuadratic(fit_quad.coefficients, fit_quad.control_points, centers=fit_quad.centers, scales=fit_quad.scales)
    return cog


def fit_offset(
        saturation_dict, 
        abundance_proxy=None, 
        saturation_curve=None
):
    if saturation_curve is None:
        saturation_curve = voigt_saturation_curve()
    if abundance_proxy is None:
        abundance_proxy = lambda t: t.loggf - t.ep
    for trans in saturation_dict:
        sat_val = ew_dict
        delta_rews = np.log10(exemplars.ew/exemplars.doppler_width)
        x_deltas = exemplars.x.values - self.cog.inverse(delta_rews.values)
        offset = np.sum(x_deltas)
        if np.isnan(offset):
            offset = fallback_offset
        self.fdat.ix[groups[species_key], "x_offset"] = offset