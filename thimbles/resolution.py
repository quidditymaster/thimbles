
import numpy as np
import scipy
from scipy.interpolate import interp1d
import thimbles as tmb
from thimbles.modeling.modeling import Model
from .sqlaimports import *
from thimbles.thimblesdb import ThimblesTable

n_delts = 1024
min_z, max_z = -6, 6
z_scores = np.linspace(min_z, max_z, n_delts)
cdf_vals = scipy.stats.norm.cdf(z_scores)
z_delta = (z_scores[1]-z_scores[0])

@jit(double[:](double[:]))
def approximate_normal_cdf(zscore):
    cdf = np.zeros(zscore.shape)
    for idx in range(zscore.shape[0]):
        cur_score = zscore[idx]
        if cur_score > max_z-z_delta-1e-5:
            cdf[idx] = 1.0
        elif cur_score < min_z:
            cdf[idx] = 0.0
        else:
            z_idx = (cur_score-min_z)/z_delta
            base_idx = int(z_idx)
            alpha = z_idx-base_idx
            cdf[idx] = cdf_vals[base_idx]*(1-alpha) + cdf_vals[base_idx+1]*alpha
    return cdf

pass
# =========================================================================== #


