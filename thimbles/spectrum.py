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

from thimbles import speed_of_light

# ########################################################################### #

__all__ = ["WavelengthSolution","Spectrum"]

# ########################################################################### #

class DeltaHelioParameter(Parameter):
    """velocity correction for motion around the sun"""
    _id = Column(Integer, ForeignKey("Parameter._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"DeltaHelioParameter"
    }
    _value = Column(Float)  #helio centric velocity in km/s
    
    #class attributes
    name = "rv"
    
    def __init__(self, value):
        self._value = value

class RadialVelocityParameter(Parameter):
    """ Radial Velocity Parameter"""
    _id = Column(Integer, ForeignKey("Parameter._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"RadialVelocityParameter"
    }
    _value = Column(Float)  #helio centric velocity in km/s
    
    #class attribtues
    name="delta_helio"
    
    def __init__(self, value):
        self._value = value

class WavelengthsParameter(Parameter):
    """a parameter vector of wavelengths"""
    _id = Column(Integer, ForeignKey("Parameter._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"WavelengthsParameter"
    }
    _value = Column(Float)  #helio centric velocity in km/s
    
    #class attributes
    name = "wvs"
    
    def __init__(self, value):
        self._value = value

class WavelengthSolution(ThimblesTable, Base):
    _indexer_id = Column(Integer, ForeignKey("Coordinatization._id"))
    indexer = relationship("Coordinatization")
    
    #the parameters
    _rv_id = Column(Integer, ForeignKey("Parameter._id"))
    rv_p = relationship("RadialVelocityParameter", foreign_keys=_rv_id)
    _delta_helio_id = Column(Integer, ForeignKey("Parameter._id"))
    delta_helio_p = relationship("DeltaHelioParameter", foreign_keys=_delta_helio_id)
    
    #TODO: a polymorphic LSF table/class
    lsf = Column(PickleType)
    
    def __init__(self, wavelengths, rv=None, delta_helio=None, shifted=True, lsf=None):
        """a class that encapsulates the manner in which a spectrum is sampled
        
        wavelengths: np.ndarray
          the central wavelengths of each pixel, the radial velocity
          and heliocentric velocity shifts are assumed to have already
          been applied unless shifted==False.
        rv: float
          the projected velocity along the line of sight not including
          the contribution from the earths motion around the earth sun
          barycenter. [km/s]
        delta_helio: float
          the velocity around the earth sun barycenter projected onto the
          line of sight. [km/s]
        shifted: bool
          if False then the passed in rv and vhelio will be applied to 
          the wavelengths via
          wavelengths = wavelengths*(1-(rv+vhelio)/c)
        lsf: ndarray
          the line spread function in units of pixel width.
        """       
        
        #set up rv
        if rv is None:
            rv = 0.0
        if not isinstance(rv, Parameter):
            rv = RadialVelocityParameter(rv)
        self.rv_p = rv
        del rv
        
        #set up delta helio
        if delta_helio is None:
            delta_helio = 0.0
        if not isinstance(delta_helio, Parameter):
            delta_helio = DeltaHelioParameter(delta_helio)
        self.delta_helio_p = delta_helio
        del delta_helio
        
        wavelengths = np.asarray(wavelengths)
        if shifted:
            #remove external wavelength corrections
            correction = 1.0/(1.0 + self.fractional_shift)
            wavelengths = wavelengths * correction
        self.indexer = as_coordinatization(wavelengths) 
        
        if lsf is None:
            lsf = np.ones(len(self))
        lsf = np.asarray(lsf)
        self.lsf = lsf
    
    def __len__(self):
        return len(self.indexer)
    
    @property
    def rv(self):
        return self.rv_p.value
    
    @rv.setter
    def rv(self, value):
        self.rv_p.value = value
    
    def set_rv(self, rv):
        self.rv = rv
    
    def get_rv(self):
        return self.rv
    
    @property
    def delta_helio(self):
        return self.delta_helio_p.value
    
    @delta_helio.setter
    def delta_helio(self, value):
        self.delta_helio_p = value
    
    @property
    def fractional_shift(self):
        rv_val = self.rv_p.value
        dh_val = self.delta_helio_p.value
        return (rv_val + dh_val)/speed_of_light
    
    def get_wvs(self, pixels=None, clip=False, snap=False):
        """get object restframe wavelengths from pixel number"""
        if pixels is None:
            pixels = np.arange(len(self.indexer))
        return self.indexer.get_coord(pixels, clip=clip, snap=snap)
    
    def get_index(self, wvs=None, clip=False, snap=False):
        """get pixel number from object rest frame wavelengths"""
        if wvs is None:
            return np.arange(len(self.indexer))
        return self.indexer.get_index(wvs, clip=clip, snap=snap)
    
    def interpolant_matrix(self, wv_soln, fill_mode="zeros"):
        """generate an interpolation matrix which will transform from
        an input wavelength solution to this wavelength solution.
        """
        wv_soln = as_wavelength_solution(wv_soln)
        shifted_wvs = wv_soln.get_wvs()*(1.0 + self.fractional_shift)
        return self.indexer.interpolant_matrix(shifted_wvs)

def as_wavelength_solution(wavelengths):
    if isinstance(wavelengths, WavelengthSolution):
        return wavelengths
    else:
        return WavelengthSolution(wavelengths)


#TODO: models carrying wavelengths to observational indexes for each of the telescope frame, sun frame, and object frame

class FluxParameter(Parameter):
    _id = Column(Integer, ForeignKey("Parameter._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"FluxParameter",
    }
    #_value = Column(PickleType)
    _wv_soln_id = Column(Integer, ForeignKey("WavelengthSolution._id"))
    wv_soln = relationship("WavelengthSolution")
    start_index = Column(Integer)
    end_index = Column(Integer)
    
    #class attributes
    name = "flux"
    
    def __init__(self, wvs, value, start_index=None, end_index=None,):
        self.wv_soln = as_wavelength_solution(wvs)
        self._value=value
        if start_index is None:
            start_index = 0
        self.start_index = start_index
        if end_index is None:
            end_index = len(self.wv_soln)
        self.end_index=end_index
    
    @property
    def pixels(self):
        return np.arange(self.start_index, self.end_index)

class Spectrum(ThimblesTable, Base):
    """A representation of a collection of relative flux measurements
    """
    _flux_p_id = Column(Integer, ForeignKey("FluxParameter._id"))
    flux_p = relationship("FluxParameter")
    _obs_prior_id = Column(Integer, ForeignKey("Distribution._id"))
    obs_prior = relationship("Distribution")
    info = Column(PickleType)
    _source_id = Column(Integer, ForeignKey("Source._id"))
    
    def __init__(self,
                 wavelengths, 
                 flux, 
                 ivar=None,
                 flags=None,
                 info=None,
                 source=None,
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
        
        if isinstance(wavelengths, FluxParameter):
            self.flux_p = wavelengths
        else:
            self.flux_p = FluxParameter(wavelengths, flux)
        #self.flux = np.asarray(flux)
        
        if ivar is None:
            ivar = tmb.utils.misc.smoothed_mad_error(flux)
        ivar = tmb.utils.misc.clean_inverse_variances(ivar)
        
        #treat the observed flux as a piror on the flux parameter values
        self.obs_prior = NormalDistribution(flux, ivar)
        
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
        return self.obs_prior.mean
    
    @flux.setter
    def flux(self, value):
        self.obs_prior.mean = value
    
    @flux.setter
    def flux(self, value):
        self.flux_p.value = value
    
    @property
    def ivar(self):
        return self.obs_prior.ivar
    
    @ivar.setter
    def ivar(self, value):
        self.obs_prior.ivar = value
    
    @property
    def var(self):
        return tmb.utils.misc.inv_var_2_var(self.ivar)
    
    @var.setter
    def var(self, value):
        self.ivar = tmb.utils.misc.inv_var_2_var(value)
    
    def sample(self,
            wavelengths,
            mode="interpolate", 
    ):
        """generate a spectrum subsampled from this one.
        
        parameters
        
        wavelengths: ndarray
          the wavelengths to sample at
        mode: string
          the type of sampling to carry out.
          "interpolate"  linearly interpolate onto the given wv
          "bounded_view" return a wavelength sampled identically to 
                         the current spectrum but bounded by the 
                         first and last wavelengths given.
          "rebin"        apply a flux preserving rebinning procedure.
        """
        if mode=="bounded_view":
            bounds = sorted([wavelengths[0], wavelengths[-1]])
            l_idx, u_idx = self.get_index(bounds, snap=True)
            if u_idx-l_idx < 1:
                return None
            out_flux = self.flux[l_idx:u_idx+1]
            fparam = FluxParameter(self.wv_soln, out_flux, start_index=l_idx, end_index=u_idx)
            out_ivar = self.ivar[l_idx:u_idx+1]
            sampled_spec = Spectrum(fparam, out_flux, out_ivar)
        elif mode == "interpolate":
            tmat = self.wv_soln.interpolant_matrix(wavelengths)
            out_flux = tmat*self.flux
            out_var = (tmat*self.var)*tmat.transpose()
            out_var_collapse = out_var*np.ones(len(out_flux))
            out_ivar = tmb.utils.var_2_inv_var(out_var_collapse)
            sampled_spec = Spectrum(wavelengths, out_flux, out_ivar)
        elif mode =="rebin":
            #check if we have the transform stored
            in_wvs = self.wvs
            wavelengths = as_wavelength_solution(wavelengths)
            out_wvs = wavelengths.get_wvs()
            transform = resampling.get_resampling_matrix(in_wvs, out_wvs, preserve_normalization=True)
            out_flux = transform*self.flux
            var = self.get_var()
            #TODO make this take into account the existing lsfs
            covar = resampling.\
                    get_transformed_covariances(transform, var)
            covar_shape = covar.shape
            #marginalize over the covariance
            out_inv_var  = 1.0/(covar*np.ones(covar_shape[0]))
            sampled_spec = Spectrum(wavelengths, out_flux, out_inv_var)
        else:
            raise ValueError("mode {} not a valid sampling mode".format(mode))
        return sampled_spec
    
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
        wvs = self.wvs
        return "<`thimbles.Spectrum` ({0:8.3f},{1:8.3f})>".format(wvs[0], wvs[-1])
    
    def __sub__(self, other):
        if isinstance(other, (float, int)):
            return Spectrum(self.wv_soln, self.flux - other, self.ivar)
        else:
            raise NotImplementedError()
        
    @property
    def rv(self):
        return self.wv_soln.rv
    
    @property
    def pixels(self):
        return self.flux_p.pixels
    
    @property
    def wvs(self):
        return self.wv_soln.get_wvs(self.pixels)
    
    def set_rv(self, rv):
        self.wv_soln.set_rv(rv)
    
    def get_rv(self):
        return self.wv_soln.get_rv()
    
    def get_wvs(self, pixels=None, clip=False, snap=False):
        if pixels is None:
            pixels = self.flux_p.pixels
        return self.wv_soln.get_wvs(pixels, clip=clip, snap=snap)
    
    def get_index(self, wvs, clip=False, snap=False):
        return self.wv_soln.get_index(wvs, clip=clip, snap=snap)
    
    def pseudonorm(self, **kwargs):
        #TODO: put extra controls in here
        norm_res = tmb.utils.misc.approximate_normalization(self, **kwargs)    
        return norm_res
    
    def normalized(self, norm=None):
        if norm is None:
            norm = self.pseudonorm()
        nspec = Spectrum(self.wv_soln, self.flux/norm, self.ivar*norm**2)
        nspec.flags["normalized"] = True
        return nspec
    
    def lsf_sampling_matrix(self, model_wvs):
        wv_widths = scipy.gradient(self.wv)*self.wv_soln.lsf
        #TODO: use faster interpolation
        interper = interp1d(self.wv, wv_widths, bounds_error=False, fill_value=1.0)
        gdense = resampling.GaussianDensity(model_wvs, interper(model_wvs))
        transform = resampling.get_resampling_matrix(model_wvs, self.wv, preserve_normalization=True, pixel_density=gdense)    
        return transform
    
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
    
    def plot(self, ax=None, **mpl_kwargs):
        return tmb.charts.SpectrumChart(self, ax=ax)
    


