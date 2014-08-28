import numpy as np
import pandas as pd
import h5py
import os
import scipy.sparse
import thimbles as tmb
from thimbles.hypercube_interpolator import HypercubeGridInterpolator
from thimbles import resource_dir
from profiles import convolved_stark
from spectrum import Spectrum

data_cols = np.loadtxt(os.path.join(resource_dir, "transition_data", "Hydrogen_lines.txt"), usecols=[0, 1, 2, 3, 5])
hlines = pd.DataFrame(data=dict(wv=data_cols[:, 0], 
                                nlow=np.array(data_cols[:, 1], dtype=int), 
                                nup=np.array(data_cols[:, 2], dtype=int), 
                                ep=data_cols[:, 3], 
                                loggf=data_cols[:, 4]),
                      )

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

class HydgrogenProfile(object):
    
    def __init__(self):
        pass

def try_load_lemke():
    try:
        hf = h5py.File(os.path.join(resource_dir, "transition_data", "lemke.h5"), "r")
        return hf
    except Exception as e:
        print e
        print "exception loading lemke hydrogen profile data trying again"
        return None

lemke_dat = try_load_lemke()

class HydrogenLineOpacity(tmb.profiles.LineProfile):
    
    def __init__(self, wv, nlow, nup):
        self.nlow = nlow
        self.nup = nup
        self.wv = wv
        low_str = "{}".format(int(nlow))
        up_str = "{}".format(int(nup))
        if low_str in lemke_dat.keys():
            if up_str in lemke_dat[low_str].keys():
                pass
        base_group = "{}/{}/".format(low_str, up_str)
        
        self.log_nes = np.array(lemke_dat[base_group+"log_ne"])
        self.log_ts = np.array(lemke_dat[base_group+"log_t"])
        self.alphas = np.array(lemke_dat[base_group+"alphas"])
        self.alpha_binner = tmb.binning.CoordinateBinning(self.alphas)
        profile_grid = np.array(lemke_dat[base_group+"profile"])
        pinterp = HypercubeGridInterpolator(coordinates=[self.log_ts, self.log_nes],
                                            grid_data=profile_grid)
        self.pinterp = pinterp
    
    def __call__(self, wvs, parameters):
        """evaluate the line opacity at the given wavelengths
        Log(Temperature), Log(electron density) = parameters
        """
        #alpha def  alpha = delta_wv/F0   F0 = 1.25e-9 * ne^(2/3)
        input_alpha = np.abs(wvs-self.wv)
        input_alpha /= np.power(10.0, (2.0/3.0)*parameters[1] -8.9030899869919438)
        input_alpha = np.clip(input_alpha,  self.alphas[0], self.alphas[-1])
        alpha_profile = self.pinterp(parameters)
        alpha_indicies = self.alpha_binner.coordinates_to_indicies(input_alpha)
        min_indexes = np.array(alpha_indicies, dtype=int)
        mixing_ratio = alpha_indicies - min_indexes
        interped_profile = alpha_profile[min_indexes]*(1-mixing_ratio)
        interped_profile += alpha_profile[min_indexes+1]*mixing_ratio
        return np.exp(interped_profile)

class HydrogenForegroundModel(Spectrum):
    
    def __init__(self, wvs, strength, temperature, electron_density, nup_max=22):
        super(HydrogenForegroundModel, self).__init__(wvs, np.ones(len(wvs)))
        self._strength = strength
        self._temperature = temperature
        self._electron_density = electron_density
        self.npts_model = len(self.wv)
        min_wv = self.wv[0]
        max_wv = self.wv[-1]
        self.hdat = hlines[(hlines.wv > min_wv)*(hlines.wv < max_wv)]
        self.initialize_lines()
        self.calc_h_opac()
        self.calc_transmission()
    
    @property
    def electron_density(self):
        return self._electron_density
        self.calc_h_opac()
        self.calc_transmission()
    
    @electron_density.setter
    def electron_density(self, value):
        self._electron_density = value
    
    @property
    def temperature(self):
        return self._temperature
    
    @temperature.setter
    def temperature(self, value):
        self._temperature = value
        self.calc_h_opac()
        self.calc_transmission()
    
    def initialize_lines(self):
        self.hprofiles = []
        for l_idx in range(len(self.hdat)):
            ldat = self.hdat.iloc[l_idx]
            cent_wv = ldat["wv"]
            nlow, nup = ldat["nlow"], ldat["nup"]
            self.hprofiles.append(HydrogenLineOpacity(cent_wv, nlow, nup))
    
    def calc_h_opac(self):
        self._h_opac_profile = np.zeros(self.wv.shape)
        rel_strengths = np.power(10.0, self.hdat["loggf"])
        for line_idx, line_profile in enumerate(self.hprofiles):
            rel_strength = rel_strengths.iloc[line_idx]
            self._h_opac_profile += line_profile(self.wv, [np.log10(self.temperature), np.log10(self.electron_density)])
    
    @property
    def strength(self):
        return self._strength
    
    @strength.setter 
    def strength(self, value):
        self._strength = value
        self.calc_transmission()
        
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
    
    def parameter_expansion(self, input):
        start_strength = self.strength
        self.strength = 0.99*start_strength
        low_vec = self(input)
        self.strength = 1.01*start_strength
        high_vec = self(input)
        str_delta_vec = ()
        return 