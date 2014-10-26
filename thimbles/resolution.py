"""testing testing
"""

import numpy as np
import scipy
from scipy.interpolate import interp1d
import thimbles as tmb
from thimbles.modeling.modeling import Model

n_delts = 1024
z_scores = np.linspace(-6, 6, n_delts)
cdf_vals = scipy.stats.norm.cdf(z_scores)
min_z = z_scores[0]
max_z = z_scores[-1]
z_delta = (z_scores[1]-z_scores[0])

def approximate_gaussian_cdf(zscore):
    if zscore > max_z-z_delta-1e-5:
        return 1.0
    elif zscore < min_z:
        return 0
    idx_val = (zscore-min_z)/z_delta
    base_idx = int(idx_val)
    alpha = idx_val-base_idx
    return cdf_vals[base_idx]*(1-alpha) + cdf_vals[base_idx+1]*alpha

pass
# =========================================================================== #

class LineSpreadFunction:
    """a class for describing line spread functions.
    Integrations are carried out over pixel space. 
    """
    
    def integrate(self, index, lb, ub):
        lower_val = self.get_integral(index, lb)
        upper_val = self.get_integral(index, ub)
        return upper_val-lower_val

class GaussianLSF(LineSpreadFunction):
    
    def __init__(self,widths, max_sigma = 5.0, wv_soln=None):
        self.wv_soln = wv_soln
        self.widths = widths
        self.max_sigma = max_sigma
    
    def get_integral(self, index, pix_coord):
        zscore = (pix_coord-index)/self.widths[index]
        return approximate_gaussian_cdf(zscore)
    
    def get_coordinate_density_range(self, index):
        lb = index - self.max_sigma*self.widths[index]
        ub = index + self.max_sigma*self.widths[index]
        return lb, ub
    
    def get_rms_width(self, index):
        return self.widths[index]

class BoxLSF(LineSpreadFunction):
    
    def __init__(self, wv_soln=None):
        self.wv_soln = wv_soln
    
    def get_integral(self, index, pix_coord):
        if pix_coord > index+0.5:
            return 1
        elif pix_coord < index-0.5:
            return 0
        else:
            return pix_coord - index + 0.5
    
    def get_coordinate_density_range(self, index):
        return index-1, index+1
    
    def get_rms_width(self, index):
        return 1.0/12.0

class DiracLSF(LineSpreadFunction):
    
    def __init__(self, wv_soln=None):
        self.wv_soln = None
        self.derivative_scale = 1e-12
    
    def get_integral(self, index, pix_coord):
        if pix_coord >= index:
            return 1.0
        else:
            return 0.0
    
    def get_coordinate_density_range(self, index):
        return self.centers[index]-self.derivative_scale, self.centers[index]+self.derivative_scale
    
    def get_rms_width(self, index):
        return 0.0

class LineSpreadFunctionModel(Model):
    
    def __init__(self, model_wvs, target_wvs, pixel_sigmas):
        super(LineSpreadFunctionModel, self).__init__()
        self.wv = model_wvs
        self.target_wvs = target_wvs
        self.target_sigmas = pixel_sigmas
        #TODO: use faster interpolation
        interper = interp1d(target_wvs, pixel_sigmas*scipy.gradient(target_wvs), bounds_error=False, fill_value=1.0)
        gdense = tmb.utils.resampling.GaussianDensity(model_wvs, interper(model_wvs))
        transform = tmb.utils.resampling.get_resampling_matrix(self.wv, self.target_wvs, preserve_normalization=True, pixel_density=gdense)    
        self._lin_op = transform
    
    def as_linear_op(self, input, **kwargs):
        return self._lin_op
    
    def __call__(self, input):
        return self._lin_op*input