import numpy as np
import pandas as pd
import h5py
import os
import scipy.sparse
import warnings
import thimbles as tmb
from thimbles.modeling import Model
from thimbles import resource_dir
from .profiles import convolved_stark
from .spectrum import Spectrum
from thimbles.tasks import task
from thimbles.sqlaimports import *

data_cols = np.loadtxt(os.path.join(resource_dir, "transition_data", "Hydrogen_lines.txt"), usecols=[0, 1, 2, 3, 5])
hlines = pd.DataFrame(data=dict(wv=data_cols[:, 0], 
                                nlow=np.array(data_cols[:, 1], dtype=int), 
                                nup=np.array(data_cols[:, 2], dtype=int), 
                                ep=data_cols[:, 3], 
                                loggf=data_cols[:, 4]),
                      )

@task()
def get_H_mask(wvs, masking_radius=-3.0, nup_max=None):
    """generates a mask which is false close to hydrogen features
    
    inputs
    wvs: np.ndarray
      the wavelengths at which to evaluate the mask
    masking_radius: float
      the radius around each hydrogen line to exclude.
      if the radius is positive it is interpreted as a constant number of
      angstroms to exclude. If the number is negative it is interpreted as
      the base 10 logarithm of the fraction of the line center wavelength
      to exclude. That is for a line at wv lambda the radius of exclusion is
      (10**masking_radius)*lambda.
    nup_max: int
      if specified hydrogen lines with upper energy levels above nup_max will
      not contribute to the mask.
    """
    min_wv = np.min(wvs)
    max_wv = np.max(wvs)
    mask = np.ones(wvs.shape, dtype=bool)
    for line_idx in range(len(hlines)):
        line_dat = hlines.iloc[line_idx]
        lwv = line_dat["wv"]
        nup = line_dat["nup"]
        if nup > nup_max:
            continue
        if masking_radius < 0:
            mrad = np.power(10.0, masking_radius)*lwv
        else:
            mrad = masking_radius
        if lwv < (min_wv - mrad):
            continue
        if lwv > (max_wv + mrad):
            continue
        mask *= np.abs(wvs-lwv) > mrad
    return mask


def try_load_lemke():
    try:
        hf = h5py.File(os.path.join(resource_dir, "transition_data", "lemke.h5"), "r")
        return hf
    except Exception as e:
        warnings.warn(str(e)+"\nexception loading lemke hydrogen profile data trying again")
        return None

lemke_dat = try_load_lemke()

class HydrogenLineOpacity(tmb.profiles.LineProfile):
    
    def __init__(self, wv, nlow, nup):
        self.nlow = nlow
        self.nup = nup
        self.wv = wv
        low_str = "{}".format(int(nlow))
        up_str = "{}".format(int(nup))
        if low_str in list(lemke_dat.keys()):
            if up_str in list(lemke_dat[low_str].keys()):
                pass
        base_group = "{}/{}/".format(low_str, up_str)
        
        self.log_nes = np.array(lemke_dat[base_group+"log_ne"])
        self.log_ts = np.array(lemke_dat[base_group+"log_t"])
        self.alphas = np.array(lemke_dat[base_group+"alphas"])
        self.alpha_binner = tmb.coordinatization.as_coordinatization(self.alphas)
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
        interped_profile += alpha_profile[np.clip(min_indexes+1, 0, len(alpha_profile)-1)]*mixing_ratio
        return np.exp(interped_profile)

class HydrogenForegroundOpacityModel(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    __mapper_args__={"polymorphic_identity":"HydrogenForegroundOpacityModel"}
    
    def __init__(self, wvs, strength, temperature, electron_density):
        Model.__init__(self)
        Spectrum.__init__(self, wvs, np.ones(len(wvs)))
        self.max_delta_wv_frac = 0.01
        self._temperature = temperature
        self._electron_density = electron_density
        self.npts_model = len(self.wv)
        min_wv = self.wv[0]
        max_wv = self.wv[-1]
        self.hdat = hlines[(hlines.wv > min_wv)*(hlines.wv < max_wv)].copy()
        self.series_ids = np.unique(self.hdat.nlow.values)
        self.series_index = {self.series_ids[idx]:idx for idx in range(len(self.series_ids))}
        
        strength = np.atleast_1d(strength)
        if len(strength) == 1:
            strength = np.repeat(strength, len(self.series_ids))
        elif len(strength) != len(self.series_ids):
            raise ValueError("different number of strengths than there are available Hydrogen Series!")
        self._strength = strength
        self.initialize_lines()
        self.calc_h_opac()
        self.calc_opac()
    
    @property
    def electron_density(self):
        return self._electron_density
    
    @electron_density.setter
    def electron_density(self, value):
        self._electron_density = value
        self.calc_h_opac()
        self.calc_opac()
    
    @property
    def temperature(self):
        return self._temperature
    
    @temperature.setter
    def temperature(self, value):
        self._temperature = value
        self.calc_h_opac()
    
    def initialize_lines(self):
        self.hprofiles = [[] for i in range(len(self.series_ids))]
        for series_idx in range(len(self.series_ids)):
            series_id = self.series_ids[series_idx]
            series_dat = self.hdat[self.hdat.nlow == series_id]
            for l_idx in range(len(series_dat)):
                ldat = series_dat.iloc[l_idx]
                cent_wv = ldat["wv"]
                nlow, nup = ldat["nlow"], ldat["nup"]
                self.hprofiles[series_idx].append(HydrogenLineOpacity(cent_wv, nlow, nup))
    
    def calc_h_opac(self):
        #self._h_opac_profile = np.zeros(self.wv.shape)
        opac_vecs = [np.zeros(self.wv.shape) for i in range(len(self.series_ids))]
        theta = 5040.0/self.temperature
        for series_idx in range(len(self.series_ids)):
            series_id = self.series_ids[series_idx]
            series_dat = self.hdat[self.hdat.nlow == series_id]
            rel_strengths = np.power(10.0, series_dat["loggf"]-theta*(series_dat["ep"]))
            for line_idx, line_profile in enumerate(self.hprofiles[series_idx]):
                rel_strength = rel_strengths.iloc[line_idx]
                lb, ub = self.get_index(line_profile.wv*np.array([1.0-self.max_delta_wv_frac, 1.0+self.max_delta_wv_frac]), clip=True)
                opac_vecs[series_idx][lb:ub+1] += line_profile(self.wv[lb:ub+1], [np.log10(self.temperature), np.log10(self.electron_density)])
        self.opac_matrix = scipy.sparse.bmat(opac_vecs).transpose()
    
    @property
    def strength(self):
        return self._strength
    
    @strength.setter 
    def strength(self, value):
        self._strength = np.clip(value, 0.01, np.inf)
        self.calc_opac()
    
    #@parameter(free=True, min=0.01, max=5.0, scale=1.0, step_scale=0.01)
    def strength_p(self):
        return self.strength
    
    #@strength_p.setter
    def set_strength(self, value):
        self.strength = value
    
    #@strength_p.expander
    def strength_expansion(self, input):
        return self.opac_matrix
    
    def __call__(self, input):
        return input + self.opac_matrix*self.strength
    
