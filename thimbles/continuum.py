import numpy as np
import pandas as pd
import thimbles as tmb
import matplotlib.pyplot as plt
import scipy
import scipy.integrate as integrate
import scipy.sparse
from thimbles.utils.misc import smooth_ppol_fit
import thimbles.utils.piecewise_polynomial as ppol
from thimbles import logger
from spectrum import Spectrum

class BlackBodyContinuumModel(Spectrum):
    
    def __init__(self, model_wvs, teff):
        self.model_wvs = model_wvs
        self.npts_model = len(model_wvs)
        self.teff = teff
    
    def calc_bk_bod(self):
        self._bk_bod = tmb.utils.misc.blackbody_flux(self.model_wvs, self.teff, normalize=True)
        self._lin_op = scipy.sparse.dia_matrix((self._bk_bod, 0), shape=(self.npts_model, self.npts_model))
    
    @property
    def teff(self):
        return self._teff
    
    @teff.setter
    def teff(self, value):
        self._teff = value
        self.calc_bk_bod()
    
    def as_linear_op(self, input, **kwargs):
        return self._lin_op
    
    def __call__(self, input):
        return self._bk_bod*input
    