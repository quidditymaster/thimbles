#
#
# ########################################################################### #
import time
from copy import copy

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
#from thimbles.resolution import LineSpreadFunction, GaussianLSF, BoxLSF
#from thimbles.binning import CoordinateBinning
from scipy.interpolate import interp1d
from thimbles.thimblesdb import ThimblesTable, Base
from thimbles.modeling import Model, Parameter
from thimbles.modeling.factor_models import PickleParameter, FloatParameter
from thimbles.modeling.factor_models import IdentityMap
from thimbles.modeling.distributions import VectorNormalDistribution
from thimbles.resolution import PolynomialLSFModel
from thimbles.coordinatization import Coordinatization, as_coordinatization
from thimbles.sqlaimports import *
from sqlalchemy.orm.collections import attribute_mapped_collection

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
    
    def __init__(self, value):
        self._value = value


class RadialVelocityParameter(Parameter):
    """ Radial Velocity Parameter"""
    _id = Column(Integer, ForeignKey("Parameter._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"RadialVelocityParameter"
    }
    _value = Column(Float)  #helio centric velocity in km/s
    
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
    
    _lsf_id = Column(Integer, ForeignKey("Parameter._id"))
    lsf_p = relationship("Parameter", foreign_keys=_lsf_id)
    
    def __init__(self, wavelengths, rv=None, delta_helio=None, helio_shifted=True, rv_shifted=True, lsf=None, lsf_type="quadratic"):
        """a class that encapsulates the manner in which a spectrum is sampled
        
        wavelengths: np.ndarray
          the central wavelengths of each pixel, the radial velocity
          and heliocentric velocity shifts are assumed to have already
          been applied unless shifted==False.
        rv: float
          the projected velocity of the object along the line of 
          sight relative to the sun. [km/s]
        delta_helio: float
          the velocity of the earth around the sun projected onto the
          line of sight. [km/s]
        helio_shifted: bool
          whether or not delta_helio has already been applied.
          if False then delta_helio will be applied via
          wavelengths = wavelengths*(1-vhelio/c)
        rv_shifted: bool
          whether or not the radial velocity has already been applied.
          if False then the rv will be applied 
          wavelengths = wavlengths*(1-rv/c)
        lsf: ndarray
          the line spread function in units of pixel width.
        lsf_type: string
          how to treat the passed lsf parameter.
          'cached' store the passed array as a PickleParameter
          'quadratic' fit a quadratic to the lsf and generate a
             quadratic polynomial lsf model with this lsf as its output.
          'cubic'   same as quadratic but degree==3
          'quartic' same as quadratic but degree==4
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
        shift_to_apply = 0.0
        if not helio_shifted:
            shift_to_apply += self.delta_helio_p.value
        if not rv_shifted:
            shift_to_apply += self.rv_p.value
        fractional_shift = shift_to_apply/speed_of_light
        correction = 1.0/(1.0 + fractional_shift)
        wavelengths = wavelengths * correction
        self.indexer = as_coordinatization(wavelengths) 
        
        if lsf is None:
            lsf = np.ones(len(self))
        
        if lsf_type == "cached":
            lsf_p = PickleParameter(lsf)
        else:
            lsf_p = Parameter(lsf)
        self.lsf_p = lsf_p
        if lsf_type == "quadratic":
            lsf_mod = PolynomialLSFModel(lsf_p, degree=2)
        elif lsf_type == "cubic":
            lsf_mod = PolynomialLSFModel(lsf_p, degree=3)
        elif lsf_type == "quartic":
            lsf_mod = PolynomialLSFModel(lsf_p, degree=4)
        else:
            raise ValueError("lsf_type {} is not recognized")
    
    def __len__(self):
        return len(self.indexer)
    
    def __getitem__(self, index):
        return self.indexer[index]
    
    @property
    def lsf(self):
        return self.lsf_p.value
    
    @lsf.setter
    def lsf(self, value):
        self.lsf_p.set(value)
    
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
    
    def interpolant_sampling_matrix(self, wvs):
        """generate an interpolation matrix which will transform from
        an input wavelength solution to this wavelength solution.
        """
        wvs = as_wavelength_sample(wvs)
        return self.indexer.interpolant_sampling_matrix(wvs.wvs)


def as_wavelength_solution(wavelengths):
    if isinstance(wavelengths, WavelengthSolution):
        return wavelengths
    elif isinstance(wavelengths, WavelengthSample):
        return wavelengths.wv_soln
    else:
        return WavelengthSolution(wavelengths)


class WavelengthSample(ThimblesTable, Base):
    _wv_soln_id = Column(Integer, ForeignKey("WavelengthSolution._id"))
    wv_soln = relationship("WavelengthSolution")
    start = Column(Integer)
    end = Column(Integer)
    
    def __init__(self, wv_soln, start=None, end=None):
        wv_soln = as_wavelength_solution(wv_soln)
        self.wv_soln = wv_soln
        if start is None:
            start = 0
        if end is None:
            end = len(wv_soln)
        self.start = int(start)
        self.end = int(end)
    
    def __len__(self):
        return self.end-self.start
    
    @property
    def pixels(self):
        return np.arange(self.start, self.end)
    
    @property
    def wvs(self):
        return self.wv_soln.get_wvs(self.pixels)
    
    def interpolant_sampling_matrix(self, wavelengths):
        """calculate the sparse matrix which linearly interpolates"""
        full_mat = self.wv_soln.interpolant_sampling_matrix(wavelengths)
        part_mat = full_mat.tocsc()[:, self.start:self.end].copy()
        return part_mat

def as_wavelength_sample(wvs):
    if isinstance(wvs, WavelengthSample):
        return wvs
    else:
        return WavelengthSample(wvs)


class FluxParameter(Parameter):
    _id = Column(Integer, ForeignKey("Parameter._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"FluxParameter",
    }
    _wv_sample_id = Column(Integer, ForeignKey("WavelengthSample._id"))
    wv_sample = relationship("WavelengthSample")
    
    def __init__(self, wvs, flux=None):
        self.wv_sample = as_wavelength_sample(wvs)
        self._value = flux
    
    def __len__(self):
        return len(self.wv_sample)
    
    @property
    def pixels(self):
        return self.wv_sample.pixels


class Spectrum(ThimblesTable, Base):
    """A representation of a collection of relative flux measurements
    """
    #_flux_p_id = Column(Integer, ForeignKey("FluxParameter._id"))
    #flux_p = relationship("FluxParameter")
    #_obs_prior_id = Column(Integer, ForeignKey("Distribution._id"))
    #obs_prior = relationship("Distribution")
    
    info = Column(PickleType)
    _flag_id = Column(Integer, ForeignKey("SpectrumFlags._id"))
    flags = relationship("SpectrumFlags")
    _source_id = Column(Integer, ForeignKey("Source._id"))
    source = relationship("Source", backref="spectroscopy")
    
    def __init__(
            self,
            wavelengths, 
            flux, 
            ivar=None,
            lsf=None,
            flags=None,
            info=None,
            source=None,
            pseudonorm_func=None,
            pseudonorm_kwargs=None,
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
        if pseudonorm_func is None:
            pseudonorm_func = tmb.pseudonorms.sorting_norm
        self.pseudonorm_func=pseudonorm_func
        if pseudonorm_kwargs is None:
            pseudonorm_kwargs = {}
        self.pseudonorm_kwargs = pseudonorm_kwargs
        
        self.flux_p = FluxParameter(wavelengths)
        if not lsf is None:
            self.flux_p.wv_sample.wv_soln.lsf = lsf
        
        flux = np.asarray(flux)
        if ivar is None:
            ivar = tmb.utils.misc.smoothed_mad_error(flux)
        #treat the observed flux as a prior on the flux parameter values
        self.obs_prior = VectorNormalDistribution(flux, ivar)
        
        if flags is None:
            flags = SpectrumFlags()
        elif isinstance(flags, SpectrumFlags):
            flags = flags
        elif isinstance(flags, int):
            flags = SpectrumFlags(flags)
        elif isinstance(flags, dict):
            flags = SpectrumFlags(flags)
        self.flags = flags
        
        if info is None:
            info = {}
        self.info = info
        
        self.source = source
    
    @property
    def wv_sample(self):
        return self.flux_p.wv_sample
    
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
            mode="rebin", 
            return_matrix=None,
    ):
        """generate a spectrum subsampled from this one.
        
        parameters
        
        wavelengths: ndarray or WavelengthSolution 
          the wavelengths to sample at
        
        mode: string
         the type of sampling to carry out.
         
         choices
          "interpolate"  linearly interpolate onto the given wv
          "bounded"      ignore all but the first and last wavelengths 
                           given and sample in between in a manner 
                           identical to the current spectrum.
          "rebin"        apply a flux preserving rebinning procedure.
        
        return_matrix: bool
          if true return both a sampled version of the spectrum and the
          matrix used to generate it via multiplication into this spectrum's
          flux attribute.
        """
        if mode=="bounded":
            bounds = sorted([wavelengths[0], wavelengths[-1]])
            l_idx, u_idx = self.get_index(bounds, snap=True, clip=True)
            if u_idx-l_idx < 1:
                return None
            out_flux = self.flux[l_idx:u_idx+1]
            wv_sample = WavelengthSample(self.wv_sample.wv_soln, l_idx, u_idx+1)
            out_ivar = self.ivar[l_idx:u_idx+1]
            sampling_matrix = scipy.sparse.identity(len(out_flux))
            sampled_spec = Spectrum(wv_sample, out_flux, out_ivar)
        elif mode == "interpolate":
            tmat = self.wv_sample.interpolant_sampling_matrix(wavelengths)
            out_flux = tmat*self.flux
            out_var = tmat.transpose()*tmat*self.var
            out_ivar = tmb.utils.misc.var_2_inv_var(out_var)
            sampled_spec = Spectrum(wavelengths, out_flux, out_ivar)
            sampling_matrix = tmat
        elif mode =="rebin":
            in_wvs = self.wvs
            wavelengths = as_wavelength_sample(wavelengths)
            out_wvs = wavelengths.wvs
            transform = resampling.pixel_integrated_lsf(
                in_wvs, 
                out_wvs, 
                lsf=1.0,
                lsf_cdf=resampling.uniform_cdf,
                normalize="rows"
            )
            out_flux = transform*self.flux
            var = self.var
            covar = resampling.\
                    get_transformed_covariances(transform, var)
            covar_shape = covar.shape
            #marginalize over the covariance
            out_inv_var  = 1.0/(covar*np.ones(covar_shape[0]))
            sampled_spec = Spectrum(wavelengths, out_flux, out_inv_var)
            sampling_matrix = transform
        else:
            raise ValueError("mode {} not a valid sampling mode".format(mode))
        if not return_matrix:
            return sampled_spec
        else:
            return sampled_spec, sampling_matrix
    
    def __add__(self, other):
        if isinstance(other, (float, int)):
            return Spectrum(self.wv_soln, self.flux + other, self.ivar)
        else:
            return add_spectra(self, other, self.wv_sample)
    
    def __div__(self, other):
        if isinstance(other, (float, int)):
            return Spectrum(self.wv_soln, self.flux/other, self.ivar*other**2)
        else:
            return divide_spectra(self, other, self.wv_sample)
    
    def __len__(self):
        return len(self.flux)
    
    def __mul__(self, other):
        if isinstance(other, (float, int)):
            return Spectrum(self.wv_soln, self.flux*other, self.ivar/other**2)
        else:
            return multiply_spectra(self, other, self.wv_sample)
    
    def __repr__(self):
        wvs = self.wvs
        return "<`thimbles.Spectrum` ({0:8.3f},{1:8.3f})>".format(wvs[0], wvs[-1])
    
    def __sub__(self, other):
        if isinstance(other, (float, int)):
            return Spectrum(self.wv_soln, self.flux - other, self.ivar)
        else:
            return subtract_spectra(self, other, self.wv_sample)
    
    @property
    def rv(self):
        return self.wv_sample.wv_soln.rv
    
    @property
    def pixels(self):
        return self.wv_sample.pixels
    
    @property
    def wvs(self):
        return self.wv_sample.wv_soln.get_wvs(self.pixels)
    
    def set_rv(self, rv):
        self.wv_sample.wv_soln.set_rv(rv)
    
    def get_rv(self):
        return self.wv_sample.wv_soln.get_rv()
    
    def get_wvs(self, pixels=None, clip=False, snap=False):
        if pixels is None:
            pixels = self.wv_sample.pixels
        return self.wv_sample.wv_soln.get_wvs(pixels, clip=clip, snap=snap)
    
    def get_index(self, wvs, clip=False, snap=False, local=False):
        global_indexes = self.wv_sample.wv_soln.get_index(wvs, clip=clip, snap=snap)
        if local:
            return global_indexes - self.wv_sample.start
        else:
            return global_indexes
    
    def pseudonorm(self, **kwargs):
        kwdict = copy(self.pseudonorm_kwargs)
        kwdict.update(kwargs)
        return self.pseudonorm_func(self, **kwargs)
    
    def normalized(self, norm=None, **kwargs):
        if norm is None:
            if not self.flags["normalized"]:
                norm = self.pseudonorm(**kwargs)
            else:
                norm = np.ones(len(self))
        nspec = Spectrum(self.wv_sample, self.flux/norm, self.ivar*norm**2)
        flux_p_val = nspec.flux_p.value
        if not flux_p_val is None:
            nspec.flux_p.value = self.flux_p.value/norm
        nspec.flags["normalized"] = True
        return nspec
    
    def plot(self, ax=None, **mpl_kwargs):
        return tmb.charts.SpectrumChart(self, ax=ax)


def add_spectra(
        spectrum1,
        spectrum2,
        target_wvs=None, 
        sampling_mode="interpolate",
):
    if target_wvs is None:
        target_wvs = spectrum1.wv_sample
    else:
        target_wvs = as_wavelength_sample(target_wvs)
    spec1_samp = spectrum1.sample(target_wvs, mode=sampling_mode)
    spec2_samp = spectrum2.sample(target_wvs, mode=sampling_mode)
    
    out_flux = spec1_samp.flux + spec2_samp.flux
    out_var = spec1_samp.var + spec2_samp.var
    out_ivar = tmb.utils.misc.var_2_inv_var(out_var)
    return tmb.Spectrum(target_wvs, out_flux, out_ivar)

def subtract_spectra(
        spectrum1,
        spectrum2,
        target_wvs=None, 
        sampling_mode="interpolate",
):
    if target_wvs is None:
        target_wvs = spectrum1.wv_sample
    else:
        target_wvs = as_wavelength_sample(target_wvs)
    spec1_samp = spectrum1.sample(target_wvs, mode=sampling_mode)
    spec2_samp = spectrum2.sample(target_wvs, mode=sampling_mode)
    
    out_flux = spec1_samp.flux - spec2_samp.flux
    out_var = spec1_samp.var + spec2_samp.var
    out_ivar = tmb.utils.misc.var_2_inv_var(out_var)
    return tmb.Spectrum(target_wvs, out_flux, out_ivar)

def multiply_spectra(
        spectrum1,
        spectrum2,
        target_wvs=None, 
        sampling_mode="interpolate",
):
    if target_wvs is None:
        target_wvs = spectrum1.wv_sample
    else:
        target_wvs = as_wavelength_sample(target_wvs)
    spec1_samp = spectrum1.sample(target_wvs, mode=sampling_mode)
    spec2_samp = spectrum2.sample(target_wvs, mode=sampling_mode)
    
    out_flux = spec1_samp.flux * spec2_samp.flux
    out_var = spec1_samp.var*spec2_samp.flux**2 + spec1_samp.flux**2*spec2_samp.var
    out_ivar = tmb.utils.misc.var_2_inv_var(out_var)
    return tmb.Spectrum(target_wvs, out_flux, out_ivar)


def divide_spectra(
        spectrum1,
        spectrum2,
        target_wvs=None, 
        sampling_mode="interpolate",
):
    if target_wvs is None:
        target_wvs = spectrum1.wv_sample
    else:
        target_wvs = as_wavelength_sample(target_wvs)
    spec1_samp = spectrum1.sample(target_wvs, mode=sampling_mode)
    spec2_samp = spectrum2.sample(target_wvs, mode=sampling_mode)
    
    out_flux = spec1_samp.flux/spec2_samp.flux
    s2sq = spec2_samp.flux**2
    out_var = spec1_samp.var/s2sq + (spec1_samp.flux/(2.0*s2sq))**2*spec2_samp.var
    out_ivar = tmb.utils.misc.var_2_inv_var(out_var)
    return tmb.Spectrum(target_wvs, out_flux, out_ivar)


class RootSpectrumModel(tmb.modeling.Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"RootSpectrumModel",
    }
    
    def __init__(self, spectrum, model_wv_soln=None):
        self.output_p = spectrum.flux_p
        self.add_input("norm", FluxParameter(spec.wv_sample))
        self.add_input("sampling_matrix", Parameter())
        self.add_input("sky_add", FluxParameter(model_wv_soln))
        self.add_input("sky_mul", FluxParameter(model_wv_soln))
        self.add_input("broadening_matrix", Parameter())
        self.add_input("continuum", Parameter())
        self.add_input("features", FluxParameter(model_wv_soln))
    
    def __call__(self, vprep):
        vdict = self.get_vdict(vprep)
        norm = vdict[self.inputs["norm"]]
        samp_mat = vdict[self.inputs["sampling_matrix"]]
        sky_add = vdict[self.inputs["sky_add"]]
        sky_mul = vdict[self.inputs["sky_mul"]]
        bmat = vdict[self.inputs["broadening_matrix"]]
        continuum = vdict[self.inputs["continuum"]]
        features = vdict[self.inputs["features"]]
        
        pre_samp = sky_add*sky_mul*(bmat*(continuum*features))
        return norm*(samp_mat*(pre_samp))
        
