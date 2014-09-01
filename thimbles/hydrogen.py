import numpy as np
import pandas as pd
import os
import scipy.sparse
from thimbles import resource_dir
from profiles import hydrogen_profile
from spectrum import Spectrum

data_cols = np.loadtxt(os.path.join(resource_dir, "transition_data", "Hydrogen_lines.txt"), usecols=[0, 1, 2, 3, 5])
hlines = pd.DataFrame(data=dict(wv=data_cols[:, 0], n_lo=data_cols[:, 1], n_up=data_cols[:, 2], ep=data_cols[:, 3], loggf=data_cols[:, 4]))

def get_H_mask(wvs, masking_radius=10.0):
    """a mask to remove wavelengths close to hydrogen features"""
    min_wv = np.min(wvs)
    max_wv = np.max(wvs)
    mask = np.ones(wvs.shape, dtype=bool)
    for line_idx in range(len(hlines)):
        lwv = hlines.iloc[line_idx]["wv"]
        if lwv < (min_wv - masking_radius):
            continue
        if lwv > (max_wv + masking_radius):
            continue
        mask *= np.abs(wvs-lwv) > masking_radius
    return mask


class HydrogenForegroundModel(Spectrum):
    
    def __init__(self, wvs, strength, sigma, gamma, e0_width):
        super(HydrogenForegroundModel, self).__init__(wvs, np.ones(len(wvs)))
        self._e0_width = e0_width
        self._strength = strength
        self._sigma = sigma
        self._gamma = gamma
        self.npts_model = len(self.wv)
        min_wv = self.wv[0]
        max_wv = self.wv[-1]
        self.hdat = hlines[(hlines.wv > min_wv)*(hlines.wv < max_wv)]
        self.calc_h_opac()
        self.calc_transmission()
    
    def calc_h_opac(self):
        self._h_opac_profile = np.zeros(self.wv.shape)
        rel_strengths = np.power(10.0, self.hdat["loggf"])
        delta_width = self.sigma+self.gamma+self.e0_width
        for line_idx in range(len(self.hdat)):
            rel_strength = rel_strengths.iloc[line_idx]
            hwv = self.hdat.wv.iloc[line_idx]
            min_idx = self.get_index(hwv-50*delta_width, clip=True)
            max_idx = self.get_index(hwv+50*delta_width, clip=True)
            sigma = self.sigma*(hwv/6562.0)
            op_prof = rel_strength*hydrogen_profile(self.wv[min_idx:max_idx+1], hwv, g_width=self.sigma, l_width=self.gamma, stark_width=self.e0_width)
            self._h_opac_profile[min_idx:max_idx+1] += op_prof
    
    @property
    def sigma(self):
        return self._sigma
    
    @sigma.setter
    def sigma(self, value):
        self._sigma = value
        self.calc_h_opac()
        self.calc_transmission()
    
    @property
    def strength(self):
        return self._strength
    
    @strength.setter 
    def strength(self, value):
        self._strength = value
        self.calc_transmission()
    
    @property
    def e0_width(self):
        return self._e0_width
    
    @property
    def gamma(self):
        return self._gamma
    
    def parameter_expansion(self, input):
        str_delta = self.strength*0.01
        plus_trans = np.exp(-(self.strength+str_delta)*self._h_opac)
        minus_trans = np.exp(-(self.strength-str_delta)*self._h_opac)
        strength_deriv_vec = (plus_trans-minus_trans)/(2.0*str_delta)
        expansion_mat = scipy.sparse.csc_matrix(strength_deriv_vec.reshape((-1, 1)))
        return expansion_mat
    
    def calc_transmission(self):
        self.flux = np.exp(-self.strength*self._h_opac_profile)
    
    def as_linear_op(self, input):
        return scipy.sparse.dia_matrix((self.flux, 0), shape=(self.npts_model, self.npts_model))
    
    def __call__(self, input):
        return input*self.flux