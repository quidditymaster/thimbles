#
#
# ########################################################################### #
import time

# 3rd Party
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt

# internal
import thimbles as tmb
from thimbles import resampling
#from thimbles.utils.misc import inv_var_2_var, var_2_inv_var, clean_variances, clean_inverse_variances
#from .metadata import MetaData
from thimbles.flags import SpectrumFlags
from thimbles import logger
#from thimbles.reference_frames import InertialFrame
#from thimbles.resolution import LineSpreadFunction, GaussianLSF, BoxLSF
#from thimbles.binning import CoordinateBinning
from scipy.interpolate import interp1d
from thimbles.thimblesdb import ThimblesTable, Base
from thimbles.modeling import Model, Parameter
from thimbles.modeling.distributions import NormalDistribution
from thimbles.coordinatization import Coordinatization, as_coordinatization
from thimbles.sqlaimports import *

# ########################################################################### #

__all__ = ["WavelengthSolution","Spectrum"]

# ########################################################################### #

speed_of_light = 299792.458 #speed of light in km/s

class WavelengthSolution(ThimblesTable, Base):
    _indexer_id = Column(Integer, ForeignKey("Coordinatization._id"))
    indexer = relationship("Coordinatization")
    lsf = Column(PickleType)
    
    def __init__(self, wavelengths, rv=None, vhelio=None, shifted=True, lsf=None):
        """a class that encapsulates the manner in which a spectrum is sampled
        
        wavelengths: np.ndarray
          the central wavelengths of each pixel, the radial velocity
          and heliocentric velocity shifts are assumed to have already
          been applied unless shifted==False.
        rv: float
          the projected velocity along the line of sight not including
          the contribution from the earths motion around the earth sun
          barycenter. [km/s]
        vhelio: float
          the velocity around the earth sun barycenter projected onto the
          line of sight. [km/s]
        shifted: bool
          if False then the passed in rv and vhelio will be applied to 
          the wavelengths via
          wavelengths = wavelengths*(1-(rv+vhelio)/c)
        lsf: ndarray
          the line spread function in units of pixel width.
        """
        if rv is None:
            rv = 0.0
        self._rv = rv
        
        if vhelio is None:
            vhelio = 0.0
        self._vhelio = vhelio
        
        if not shifted:
            wavelengths = wavelengths*(1.0-(rv+vhelio)/speed_of_light)
        
        self.indexer = as_coordinatization(wavelengths)
        
        if lsf is None:
            lsf = np.ones(len(self))
        self.lsf = lsf
    
    def __len__(self):
        return len(self.indexer)
    
    @property
    def rv(self):
        return self._rv
    
    @rv.setter
    def rv(self, value):
        new_wvs = self.indexer.coordinates*(1.0 - value/speed_of_light)
        self.indexer.coordinates = new_wvs
        self._rv = value
    
    def set_rv(self, rv):
        self.rv = rv
    
    def get_rv(self):
        return self.rv
    
    @property
    def vhelio(self):
        return self._vhelio
    
    @vhelio.setter
    def vhelio(self, value):
        new_wvs = self.indexer.coordinates*(1.0 - value/speed_of_light)
        self.indexer.coordinates = new_wvs
        self._vhelio = value
    
    def get_wvs(self, pixels=None, clip=False, snap=False):
        if pixels is None:
            pixels = np.arange(len(self.indexer))            
        return self.indexer.get_coord(pixels, clip=clip, snap=snap)
    
    def get_index(self, wvs=None, clip=False, snap=False):
        if wvs == None:
            return np.arange(len(self.indexer))
        return self.indexer.get_index(wvs, clip=clip, snap=snap)
    
    def interp_matrix(self, wv_soln, fill_mode="zeros"):
        """generate an interpolation matrix which will transform from
        an input wavelength solution to this wavelength solution.
        """
        #TODO:

def as_wavelength_solution(wavelengths):
    if isinstance(wavelengths, WavelengthSolution):
        return wavelengths
    else:
        return WavelengthSolution(wavelengths)


class FluxParameter(Parameter):
    _id = Column(Integer, ForeignKey("Parameter._id"), primary_key=True)
    _value = Column(PickleType)
    _wv_soln_id = Column(Integer, ForeignKey("WavelengthSolution._id"))
    wv_soln = relationship("WavelengthSolution")
    
    #class attributes
    name = "flux"
    
    def __init__(self, wvs, value, free=True, propagate=True):
        self.wv_soln = as_wavelength_solution(wvs)
        self._value=value
        self.free=free
        self.propagate=propagate


class Spectrum(ThimblesTable, Base):
    """A representation of a collection of relative flux measurements
    """
    _flux_p_id = Column(Integer, ForeignKey("FluxParameter._id"))
    flux_p = relationship("FluxParameter")
    _obs_prior_id = Column(Integer, ForeignKey("Distribution._id"))
    _obs_prior = relationship("Distribution")
    info = Column(PickleType)
    
    def __init__(self,
                 wavelengths, 
                 flux, 
                 ivar=None,
                 flags=None,
                 info=None,
             ):
        """makes a spectrum
        wavelengths: ndarray or WavelengthSolution
          the wavelengths at each pixel, if more specific information
          is required (e.g. specifying the lsf or the heliocentric vel
          create a WavelengthSolution object outside with the info and
          pass that in instead of an ndarray.
        flux: ndarray
          the measured flux in each pixel
        ivar: ndarray or Distribution
          the inverse variance of the noise in each pixel
        flags: an optional SpectrumFlags object
          see thimbles.flags for info
        """
        flux = np.asarray(flux)
        
        self.flux_p = FluxParameter(wavelengths, flux)
        #self.flux = np.asarray(flux)
        
        if ivar is None:
            ivar = tmb.utils.misc.smoothed_mad_error(flux)
        ivar = tmb.utils.misc.clean_inverse_variances(ivar)
        
        #treat the observed flux as a piror on the flux parameter values
        self._obs_prior = NormalDistribution(flux, ivar)
        
        if flags is None:
            flags = SpectrumFlags()
        else:
            flags = SpectrumFlags(int(flags))
        self.flags = flags
        
        if info is None:
            info = {}
        self.info = info
    
    @property
    def wv_soln(self):
        return self.flux_p.wv_soln
    
    @property
    def flux(self):
        return self._obs_prior.mean
    
    @flux.setter
    def flux(self, value):
        self._obs_prior.mean = value
    
    @flux.setter
    def flux(self, value):
        self.flux_p.value = value
    
    @property
    def ivar(self):
        return self._obs_prior.ivar
    
    @ivar.setter
    def ivar(self, value):
        self._obs_prior.ivar = value
    
    @property
    def var(self):
        return tmb.utils.misc.inv_var_2_var(self.ivar)
    
    @var.setter
    def var(self, value):
        self.ivar = tmb.utils.misc.inv_var_2_var(value)
    
    def sample(wavelengths,
               valuation_mode="interp", 
               fill_mode="nearest",
               ):
        other_wv_soln = as_wavelength_solution(wavelengths)
        interp_trans = self.wv_soln.interp_matrix(other_wv_soln)
        
        if mode == "interp":
            out_lsf = self._lsf
            trans = self.wv_soln.interp_matrix(other_wv_soln)
    
    def __add__(self, other):
        if isinstance(other, (float, int)):
            return Spectrum(self.wv_soln, self.flux + other, self.ivar)
        else:
            raise NotImplementedError()
    
    def __div__(self, other):
        if isinstance(other, (float, int)):
            return Spectrum(self.wv_soln, self.flux/other, self.ivar*other**2)
        else:
            raise NotImplementedError()
    
    def __len__(self):
        return len(self.flux)
    
    def __mul__(self, other):
        if isinstance(other, (float, int)):
            return Spectrum(self.wv_soln, self.flux*other, self.ivar/other**2)
        else:
            raise NotImplementedError()
    
    def __repr__(self):
        wvs = self.wv
        last_wv = wvs[-1]
        first_wv = wvs[0]
        return "<`thimbles.Spectrum` ({0:8.3f},{1:8.3f})>".format(first_wv, last_wv)
    
    def __sub__(self, other):
        if isinstance(other, (float, int)):
            return Spectrum(self.wv_soln, self.flux - other, self.ivar)
        else:
            raise NotImplementedError()
        
    @property
    def rv(self):
        return self.wv_soln.rv
    
    @property
    def wv(self):
        return self.wv_soln.get_wvs()
    
    def set_rv(self, rv):
        self.wv_soln.set_rv(rv)
    
    def get_rv(self):
        return self.wv_soln.get_rv()
    
    def get_wvs(self, pixels=None, clip=False, snap=False):
        return self.wv_soln.get_wvs(pixels, clip=clip, snap=snap)
    
    def get_index(self, wvs, clip=False, snap=False):
        return self.wv_soln.get_index(wvs, clip=clip, snap=snap)
    
    #def get_inv_var(self):
    #    return self.inv_var
    #
    #def get_var(self):
    #    #TODO deal with zeros appropriately
    #    return tmb.utils.misc.inv_var_2_var(self.inv_var)
    
    def normalize(self, **kwargs):
        #TODO: put extra controls in here
        norm_res = tmb.utils.misc.approximate_normalization(self, overwrite=True, **kwargs)    
        return norm_res
    
    def normalized(self):
        nspec = Spectrum(self.wv_soln, self.flux/self.norm, self.ivar*self.norm**2)
        nspec.flags["normalized"] = True
        return nspec
    
    def sample(self, wavelengths, kind="interp", fill=np.nan):
        if kind == "interp":
            indexes = self.get_index(wavelengths)
            int_part = np.around(indexes).astype(int)
            alphas = indexes - int_part
            return self.flux[int_part]*(1-alphas) + self.flux[int_part+1]*alphas
        if kind =="rebin":
            raise NotImplementedError()
        
    def lsf_sampling_matrix(self, model_wvs):
        wv_widths = scipy.gradient(self.wv)*self.wv_soln.lsf
        #TODO: use faster interpolation
        interper = interp1d(self.wv, wv_widths, bounds_error=False, fill_value=1.0)
        gdense = resampling.GaussianDensity(model_wvs, interper(model_wvs))
        transform = resampling.get_resampling_matrix(model_wvs, self.wv, preserve_normalization=True, pixel_density=gdense)    
        return transform
        
    def rebin(self, new_wv_soln, frame="emitter"):
        #check if we have the transform stored
        if self._last_rebin_wv_soln_id == id(new_wv_soln):
            transform = self._last_rebin_transform
        else:
            in_wvs = self.get_wvs(frame=frame)
            if not isinstance(new_wv_soln, WavelengthSolution):
                new_wv_soln = WavelengthSolution(new_wv_soln)
            out_wvs = new_wv_soln.get_wvs()
            transform = resampling.get_resampling_matrix(in_wvs, out_wvs, preserve_normalization=True)
            self._last_rebin_transform = transform
            self._last_rebin_wv_soln_id = id(new_wv_soln)
        out_flux = transform*self.flux
        var = self.get_var()
        #TODO make this take into account the existing lsfs
        covar = resampling.\
        get_transformed_covariances(transform, var)
        covar_shape = covar.shape
        #marginalize over the covariance
        out_inv_var  = 1.0/(covar*np.ones(covar_shape[0]))
        return Spectrum(new_wv_soln, out_flux, out_inv_var)
    
    def old_sample(self, wvs, frame="emitter"):
        """samples the spectrum at the provided wavelengths
        linear interpolation is carried out.
        
        returns: Spectrum
        """
        #shift the wavelengths to the observed frame
        index_vals = self.get_index(wvs, frame=frame)
        upper_index = np.array(np.ceil(index_vals), dtype=int)
        lower_index = np.array(np.floor(index_vals), dtype=int)
        alphas = index_vals - lower_index
        interp_vals =  self.flux[upper_index]*alphas
        interp_vals += self.flux[lower_index]*(1-alphas)
        var = self.get_var()
        sampled_var = var[upper_index]*alphas**2
        sampled_var += var[lower_index]*(1-alphas)**2
        return Spectrum(wvs, interp_vals, tmb.utils.misc.var_2_inv_var(sampled_var))
    
    def bounding_indexes(self, bounds, frame="emitter"):
        bvec = np.asarray(bounds)
        l_idx, u_idx = map(int, np.around(self.get_index(bvec, frame=frame)))
        l_idx = min(max(0, l_idx), len(self.flux)-1)
        u_idx = max(min(len(self.flux)-1, u_idx), 0)
        return l_idx, u_idx
    
    def bounded_sample(self, bounds, frame="emitter", copy=True):
        """returns the wavelengths and corresponding flux values of the 
        spectrum which are greater than bounds[0] and less than bounds[1]
        
        inputs:
        bounds: (lower_wv, upper_wv)
        frame: the frame of the bounds
        
        outputs:
        wvs, flux, inv_var
        """
        l_idx, u_idx = self.bounding_indexes(bounds, frame)
        if u_idx-l_idx < 1:
            return None
        out_wvs = self.get_wvs(np.arange(l_idx, u_idx+1), frame=frame)
        out_flux = self.flux[l_idx:u_idx+1]
        out_invvar = self.inv_var[l_idx:u_idx+1]
        out_norm = self.norm[l_idx:u_idx+1]
        if copy:
            out_flux = out_flux.copy()
            out_invvar = out_invvar.copy()
            out_norm = out_norm.copy()
        return Spectrum(out_wvs, out_flux, out_invvar, norm=out_norm)
    
    def plot(self, axes=None, **mpl_kwargs):
        plot_wvs = self.get_wvs(frame=frame)
        plot_flux = self.flux
        if axes == None:
            axes = plt.figure().add_subplot(111)
            xlabel = 'Wavelength'
            axes.set_xlabel(xlabel)
            axes.set_ylabel('Flux')
        l, = axes.plot(plot_wvs, plot_flux, **mpl_kwargs)
        return axes,l
    


