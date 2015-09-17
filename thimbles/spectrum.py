import time
from copy import copy

# 3rd Party
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt

# internal
import thimbles as tmb
from . import resampling
from .flags import SpectrumFlags
from .thimblesdb import ThimblesTable, Base
from .modeling import Model, Parameter
from .modeling.factor_models import PickleParameter, FloatParameter, PixelPolynomialModel
from .modeling.factor_models import IdentityMap
from .modeling.distributions import NormalDistribution
from .coordinatization import \
    Coordinatization, as_coordinatization, \
    ArbitraryCoordinatizationModel, LinearCoordinatizationModel, \
    LogLinearCoordinatizationModel, ArbitraryCoordinatization, \
    LinearCoordinatization, LogLinearCoordinatization
from .sqlaimports import *
from .modeling.associations import HasParameterContext, NamedParameters, ParameterAliasMixin
from sqlalchemy.orm.collections import attribute_mapped_collection

from . import speed_of_light
from .velocity import RVShiftModel


class SpectrumAlias(ParameterAliasMixin, ThimblesTable, Base):
    _context_id = Column(Integer, ForeignKey("Spectrum._id"))
    context= relationship("Spectrum", foreign_keys=_context_id, back_populates="context")

class Spectrum(ThimblesTable, Base, HasParameterContext):
    """A representation of a collection of relative flux measurements
    """
    context = relationship("SpectrumAlias", collection_class=NamedParameters)
    
    _slice_id = Column(Integer, ForeignKey("Slice._id"))
    slice = relationship("Slice", foreign_keys=_slice_id)
    _exposure_id = Column(Integer, ForeignKey("Exposure._id"))
    exposure = relationship("Exposure", foreign_keys=_exposure_id)
    _source_id = Column(Integer, ForeignKey("Source._id"))
    source = relationship("Source", foreign_keys=_source_id)
    
    info = Column(PickleType)
    _flag_id = Column(Integer, ForeignKey("SpectrumFlags._id"))
    flags = relationship("SpectrumFlags")
    
    cdf_type = Column(Integer)
    cdf_kwargs = Column(PickleType)
    cdf_options = {
        0: resampling.gaussian_cdf,
        1: resampling.uniform_cdf,
        2: resampling.box_convolved_gaussian_cdf,
    }
    
    pseudonorm = tmb.pseudonorms.sorting_norm
    
    def __init__(
            self,
            wavelengths, 
            flux,
            ivar=None,
            var=None,
            rv=None,
            rv_shifted=True,
            delta_helio=None,
            helio_shifted=True,
            lsf=None,
            lsf_degree=2,
            cdf_type=None,
            cdf_kwargs=None,
            flags=None,
            info=None,
            slice=None,
            exposure=None,
            source=None,
    ):
        """a representation of a spectrum object
        
        parameters
        
        wavelengths: ndarray
          the wavelengths at each pixel.
        flux: ndarray
          the measured flux in each pixel
        ivar: ndarray
          the inverse variance of the noise in each pixel.
        var: ndarray
          the variance of the noise in each pixel
          one of ivar and var must be None.
        rv: float
          The instantaneous radial velocity of the source object
        rv_shifted: bool
          whether or not a correction for the rv shift has already
          been applied.
        delta_helio: float
          The line of sight velocity correction due to the motion
          of the observer around the sun.
        helio_shifted:
          whether the helio centric velocity correction due to 
          delta_helio has already been applied.
        lsf: ndarray
          line spread function
        lsf_degree: integer
          the degree of the polynomial to fit to the lsf.
        cdf_type: integer
          determines the function to use as the cumulative distribution
          function (CDF) of the line spread function. defaults to 
          the cdf of the normal distribution.
             0: gaussian
             1: uniform box
             2: box convolved gaussian
        cdf_kwargs: dict
          optional arguments to pass to the CDF generating factory 
          function. e.g. the box width for the box convolved gaussian
        flags: an optional SpectrumFlags object
          see thimbles.flags.SpectrumFlags for info
        info: dict
          dictionary of relevant metadata (must be pickleable)
        source: Source or None
          the astrophysical light source
        observation: Observation
          the observation which resulted in this spectrum
        """
        HasParameterContext.__init__(self)
        wavelengths = as_coordinatization(wavelengths)
        flux = np.asarray(flux)
        if (not (var is None)) and (not (ivar is None)):
            raise ValueError("redundant specification: got a value for both ivar and var!")
        elif ivar is None:
            if var is None: #neither is specified
                var = tmb.utils.misc.smoothed_mad_error(flux)
        else: #ivar is specified
            var = tmb.utils.misc.inv_var_2_var(ivar)
        var = tmb.utils.misc.clean_variances(var)
        ivar = tmb.utils.misc.var_2_inv_var(var)
        
        if not (wavelengths.coordinates[-1] > wavelengths.coordinates[0]):
            raise ValueError("wavelengths must increase with increasing pixel index")
        
        #set up rv
        if rv is None:
            rv = 0.0
        if not isinstance(rv, Parameter):
            rv = FloatParameter(rv)
        rv_p = rv
        del rv
        self.add_parameter("rv", rv_p)
        
        #set up delta helio
        if delta_helio is None:
            delta_helio = 0.0
        if not isinstance(delta_helio, Parameter):
            delta_helio = FloatParameter(delta_helio)
        delta_helio_p = delta_helio
        del delta_helio
        self.add_parameter("delta_helio", delta_helio_p)
        
        applied_vel = 0.0
        if rv_shifted:
            applied_vel += rv_p.value
        if helio_shifted:
            applied_vel += delta_helio_p.value
        
        obs_correction = 1.0/(1.0-applied_vel/speed_of_light)
        obs_wvs = wavelengths*obs_correction
        obs_wvs_indexer = as_coordinatization(obs_wvs)
        
        obs_wvs_p = Parameter(obs_wvs_indexer)
        self.add_parameter("obs_wvs", obs_wvs_p)
        if isinstance(obs_wvs_indexer, ArbitraryCoordinatization):
            coord_param = PickleParameter(obs_wvs.coordinates)
            ArbitraryCoordinatizationModel(
                output_p=obs_wvs_p, 
                coords_p=coord_param,
            )
        elif isinstance(obs_wvs_indexer, (LinearCoordinatization, LogLinearCoordinatization)):
            min_p = FloatParameter(obs_wvs_indexer.min)
            max_p = FloatParameter(obs_wvs_indexer.max)
            LinearCoordinatizationModel(
                output_p=obs_wvs_p, 
                min_p=min_p, 
                max_p=max_p, 
                npts=obs_wvs_indexer.npts
            )
        
        rest_wvs_p = Parameter()        
        rv_shift_mod = RVShiftModel(
            output_p=rest_wvs_p, 
            wvs_p=obs_wvs_p,
            rv_params=[rv_p, delta_helio_p])
        self.add_parameter("rest_wvs", rest_wvs_p)
        
        flux_p = Parameter()
        #treat the observed flux as a prior on the flux parameter values
        obs_flux = NormalDistribution(mean=flux, ivar=ivar)#, parameters={"obs_flux":flux_p})
        #import pdb; pdb.set_trace()
        obs_flux.add_parameter("obs", flux_p) 
        self.add_parameter("obs_flux", flux_p)
        
        npts = len(flux)
        if lsf is None:
            lsf = np.ones(npts)
        if lsf_degree == "exact":
            lsf_p = PickleParameter(lsf)
        elif isinstance(lsf_degree, int):
            lsf_p = Parameter(lsf)
            lsf_mod = PixelPolynomialModel(output_p=lsf_p, autofit=True, degree=lsf_degree)
        self.add_parameter("lsf", lsf_p)
        
        if cdf_type is None:
            cdf_type = 0 #gaussian
        self.cdf_type=cdf_type
        if cdf_kwargs is None:
            cdf_kwargs = {}
        self.cdf_kwargs = cdf_kwargs
        
        if flags is None:
            flags = SpectrumFlags()
        elif isinstance(flags, SpectrumFlags):
            flags = flags
        elif isinstance(flags, (int, dict)):
            flags = SpectrumFlags(flags)
        self.flags = flags
        
        if info is None:
            info = {}
        self.info = info
        
        self.slice = slice
        self.exposure = exposure
        self.source = source
    
    def add_parameter(self, name, parameter, is_compound=False):
        SpectrumAlias(name=name, context=self, parameter=parameter, is_compound=is_compound)
    
    @property
    def flux_p(self):
        return self.context["obs_flux"]
    
    @property
    def lsf_p(self):
        return self.context["lsf"]
    
    @property
    def lsf(self):
        return self.lsf_p.value
    
    @property
    def cdf(self):
        return self.cdf_options[self.cdf_type]
    
    @property
    def obs_flux(self):
        return self.flux_p.distributions["obs"]
    
    @property
    def flux(self):
        return self.obs_flux.mean
    
    @flux.setter
    def flux(self, value):
        self.obs_flux.mean = value
    
    @property
    def var(self):
        return self.obs_flux.var
    
    @var.setter
    def var(self, value):
        self.obs_flux.var = value
    
    @property
    def ivar(self):
        return self.obs_flux.ivar
    
    @ivar.setter
    def ivar(self, value):
        self.obs_flux.ivar = value
    
    def sample(self,
            sampling,
            mode="rebin", 
            trim_edges=False,
            return_matrix=False,
    ):
        """generate a spectrum subsampled from this one.
        
        parameters
        
        sampling: ndarry or Spectrum 
          the wavelengths to sample at or the spectrum to sample
          similarly to.
        
        mode: string
         the type of sampling to carry out.
         
         choices
          "interpolate"  linearly interpolate onto the given wv
          "bounded"      ignore all but the first and last wavelengths
                         given and return the current spectrum between
                         those wavelength bounds.
          "rebin"        apply a flux preserving rebinning procedure.
        
        trim_edges: bool
          if True instead of returning the flux at the wavelengths
          specified in the sampling argument instead only return
          the overlap of those wavelengths this spectrum.
        
        return_matrix: bool
          if true return both a sampled version of the spectrum and the
          matrix used to generate it via multiplication into this spectrum's
          flux attribute.
        """
        if isinstance(sampling, Spectrum):
            wavelengths = sampling.wvs
        else:
            wavelengths = np.asarray(sampling)
        
        if trim_edges:
            l_wv, u_wv = self.wvs[0], self.wvs[-1]
            wmask = (l_wv <= wavelengths)*(u_wv >= wavelengths)
            wavelengths = wavelengths[wmask]
        
        if mode=="bounded":
            bounds = sorted([wavelengths[0], wavelengths[-1]])
            l_idx, u_idx = self.get_index(bounds, snap=True, clip=True)
            if u_idx-l_idx < 1:
                return None
            out_wvs = self.wvs[l_idx:u_idx+1].copy()
            out_flux = self.flux[l_idx:u_idx+1].copy()
            out_ivar = self.ivar[l_idx:u_idx+1].copy()
            if return_matrix:
                npts = len(self)
                nz_vals = np.zeros(npts, dtype=float)
                nz_vals[l_idx:u_idx+1] = 1.0
                sampling_matrix = scipy.sparse.dia_matrix((nz_vals, 0), shape=(npts, npts))
            sampled_spec = Spectrum(out_wvs, out_flux, out_ivar)
        elif mode == "interpolate":
            tmat = self.wv_soln.interpolant_sampling_matrix(wavelengths)
            out_flux = tmat*self.flux
            out_var = tmat.transpose()*tmat*self.var
            out_ivar = tmb.utils.misc.var_2_inv_var(out_var)
            sampled_spec = Spectrum(wavelengths, out_flux, out_ivar)
            sampling_matrix = tmat
        elif mode =="rebin":
            in_wvs = self.wvs
            out_wvs = np.asarray(wavelengths)
            transform = resampling.resampling_matrix(
                in_wvs,
                out_wvs, 
                lsf_in = 1.0,
                lsf_out= 1.0,
                lsf_cdf=resampling.uniform_cdf,
                lsf_cut=2.0,
                normalize="rows",
                lsf_units="pixel",
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
        elif mode == "lsf-convolved":
            if not isinstance(sampling, Spectrum):
                raise ValueError("lsf-convolved sampling mode requires a Spectrum object as first argument.")
            in_wvs = self.wvs
            out_wvs = sampling.wvs
            out_lsf = sampling.lsf
            lsf_cdf = sampling.cdf
            cdf_type = sampling.cdf_type
            cdf_kwargs = sampling.cdf_kwargs
            transform = resampling.resampling_matrix(
                in_wvs,
                out_wvs,
                lsf_in = self.lsf,
                lsf_out = out_lsf,
                lsf_cdf = lsf_cdf,
                cdf_kwargs=cdf_kwargs,
                lsf_cut = 5.0,
                reverse_broadening=False,
                normalize="rows",
                lsf_units="pixel",
            )
            out_flux = transform*self.flux
            var = self.var
            covar = resampling.\
                    get_transformed_covariances(transform, var)
            covar_shape = covar.shape
            out_inv_var = 1.0/(covar*np.ones(covar_shape[0]))
            sampled_spec = Spectrum(
                out_wvs, 
                out_flux, 
                out_inv_var, 
                cdf_type=cdf_type, 
                cdf_kwargs=copy(cdf_kwargs),
                lsf=out_lsf,
            )
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
            return add_spectra(self, other, self.wvs)
    
    def __truediv__(self, other):
        if isinstance(other, (float, int)):
            return Spectrum(self.wv_soln, self.flux/other, self.ivar*other**2)
        else:
            return divide_spectra(self, other, self.wvs)
    
    def __len__(self):
        return len(self.flux)
    
    def __mul__(self, other):
        if isinstance(other, (float, int)):
            return Spectrum(self.wv_soln, self.flux*other, self.ivar/other**2)
        else:
            return multiply_spectra(self, other, self.wvs)
    
    def __repr__(self):
        wvs = self.wvs
        return "<`thimbles.Spectrum` ({0:8.3f},{1:8.3f})>".format(wvs[0], wvs[-1])
    
    def __sub__(self, other):
        if isinstance(other, (float, int)):
            return Spectrum(self.wv_soln, self.flux - other, self.ivar)
        else:
            return subtract_spectra(self, other, self.wvs)
    
    @property
    def rv(self):
        return self.get_rv()
    
    @rv.setter
    def rv(self, value):
        self.set_rv(value)
    
    @property
    def wv_soln(self):
        return self.context["rest_wvs"].value
    
    @property
    def wvs(self):
        return self.context["rest_wvs"].value.coordinates
    
    @property
    def wv(self):
        return self.context["rest_wvs"].value.coordinates
    
    def set_rv(self, rv):
        self.context["rv"].value = rv
    
    def get_rv(self):
        return self.context["rv"].value
    
    def get_wvs(self, pixels, clip=False, snap=False):
        return self.wv_soln.get_coord(pixels, clip=clip, snap=snap)
    
    def get_index(self, wvs, clip=False, snap=False, local=False):
        return self.wv_soln.get_index(wvs, clip=clip, snap=snap)
    
    def normalized(self, norm=None, **kwargs):
        if norm is None:
            if not self.flags["normalized"]:
                norm = self.pseudonorm(**kwargs)
            else:
                norm = np.ones(len(self))
        nspec = Spectrum(self.wv_soln, self.flux/norm, self.ivar*norm**2)
        flux_p_val = nspec.flux_p.value
        if not flux_p_val is None:
            nspec.flux_p.value = self.flux_p.value/norm
        nspec.flags["normalized"] = True
        return nspec
    
    def plot(self, ax=None, **mpl_kwargs):
        return tmb.charts.SpectrumChart(self, ax=ax, **mpl_kwargs)


def add_spectra(
        spectrum1,
        spectrum2,
        target_wvs=None, 
        sampling_mode="rebin",
):
    if target_wvs is None:
        target_wvs = spectrum1.wvs
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
        sampling_mode="rebin",
):
    if target_wvs is None:
        target_wvs = spectrum1.wvs
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
        sampling_mode="rebin",
):
    if target_wvs is None:
        target_wvs = spectrum1.wvs
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
        sampling_mode="rebin",
):
    if target_wvs is None:
        target_wvs = spectrum1.wvs
    spec1_samp = spectrum1.sample(target_wvs, mode=sampling_mode)
    spec2_samp = spectrum2.sample(target_wvs, mode=sampling_mode)
    
    out_flux = spec1_samp.flux/spec2_samp.flux
    s2sq = spec2_samp.flux**2
    out_var = spec1_samp.var/s2sq + (spec1_samp.flux/(2.0*s2sq))**2*spec2_samp.var
    out_ivar = tmb.utils.misc.var_2_inv_var(out_var)
    return tmb.Spectrum(target_wvs, out_flux, out_ivar)
