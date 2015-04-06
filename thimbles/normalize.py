import numpy as np
import scipy.ndimage as ndimage

from .utils import misc

def normalize(spectrum, 
              mask="layered median", 
              partition="adaptive",
              mask_kwargs=None, 
              partition_kwargs=None,
              fit_kwargs=None
          ):
    flux = spectrum.flux
    wvs = spectrum.wv
    inv_var=spectrum.ivar
    
    if mask_kwargs is None:
        mask_kwargs = {}
    if partition_kwargs is None:
        partition_kwargs = {}
    if fit_kwargs is None:
        fit_kwargs = {}
    
    if mask == "layered median":
        mask = layered_median_mask(flux, **mask_kwargs)
    elif len(mask) == len(flux):
        mask = np.asarray(mask, dtype=bool)
    
    if partition == "adaptive":
        partition = misc.min_delta_bins(wvs, **partition_kwargs)
    
    smooth_ppol_fit(wvs, flux, y_inv=inv_var, mask=mask)
    return

class Normalization(object):
    """An object to represent a normalization and all of the relevant 
    attendant data. 
    """
    
    def __init__(self, spectrum, start_norm=None, refit=True,
                 spec_mod="gaussian minima", spec_kwargs=None, spec_hints=None,
                 mask="layered median", mask_kwargs=None, mask_hints=None,
                 partition="adaptive", partition_kwargs=None, partition_hints=None,
                 fit_function="pseudo huber", fit_kwargs=None, fit_hints=None,
                 ):
        self.spectrum = spectrum
        if start_norm == None:
            #if no starting normalization is given make a quick guess
            temp_mask = misc.layered_median_mask(spectrum.flux)
            masked_fl = spectrum.flux[temp_mask]
            med = np.zeros(spectrum.flux.shape)
            med[temp_mask] = ndimage.filters.median_filter(masked_fl, 20)
            gwidth = 50
            med_sm = ndimage.gaussian_filter(med, gwidth)
            mask_sm = ndimage.gaussian_filter(temp_mask, gwidth)
            start_norm = med_sm/mask_sm
            start_norm = np.where(start_norm > 0, start_norm, 1.0)
        self._norm = np.asarray(start_norm)
        
        if spec_mod == "gaussian minima":
            raise NotImplemented
        else:
            try:
                spec_mod = np.asarray(spec_mod)
                assert len(spec_mod) == len(spectrum.flux)
            except Exception as e:
                print(e)
        pass
    
    def __getitem__(self, index):
        return self._norm[index]
    
    def __add__(self, other):
        return self._norm+other
    
    def __sub__(self, other):
        return self._norm-other
    
    def __mul__(self, other):
        return self._norm*other
    
    def __div__(self, other):
        return self._norm/other
    
    def __radd__(self, other):
        return other+self._norm
    
    def __rsub__(self, other):
        return other-self._norm
    
    def __rmul__(self, other):
        return other*self._norm
    
    def __rdiv__(self, other):
        return other/self._norm
