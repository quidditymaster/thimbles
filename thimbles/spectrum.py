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
from thimbles.utils import resampling
#from thimbles.utils.misc import inv_var_2_var, var_2_inv_var, clean_variances, clean_inverse_variances
from .metadata import MetaData
from .flags import SpectrumFlags
from . import verbosity
from .reference_frames import InertialFrame
from .resolution import LineSpreadFunction, GaussianLSF, BoxLSF
from .binning import CoordinateBinning
from scipy.interpolate import interp1d

# ########################################################################### #

__all__ = ["WavelengthSolution","Spectrum"]

# ########################################################################### #

speed_of_light = 299792.458 #speed of light in km/s

class WavelengthSolution(CoordinateBinning):
    
    def __init__(self, obs_wavelengths, rv=None, barycenter_vel=None, lsf=None):
        """a class that encapsulates the manner in which a spectrum is sampled
        obs_wavelengths: np.ndarray
        emitter_frame: InertialFrame or Float
          the frame of motion of the emitting object relative to the Earth
          sun Barycenter. A float value will be interpreted as the radial
          velocity of the frame. velocity is in Km/S away is positive.
        observer_frame: InertialFrame or Float
          the frame of motion of the observer along the line of sight
          relative to the earth sun barycenter. A float value will be 
          interpreted as the radial velocity of the frame. velocity is in 
          Km/S away is positive.
        """
        super(WavelengthSolution, self).__init__(obs_wavelengths)
        
        if rv is None:
            rv = 0.0
        self._rv = rv
        
        if barycenter_vel is None:
            barycenter_vel = 0.0
        self._barycenter_vel = barycenter_vel
        
        if lsf is None:
            lsf = np.ones(len(obs_wavelengths))
        self.lsf = lsf
    
    @property
    def rv(self):
        return self._rv
    
    @rv.setter
    def rv(self, value):
        self._rv = value
    
    def set_rv(self, rv):
        self.rv = rv
    
    def get_rv(self):
        return self.rv
    
    @property
    def barycenter_vel(self):
        return self._barycenter_vel
    
    @barycenter_vel.setter
    def barycenter_vel(self, value):
        self._varycenter_vel = value
    
    def get_wvs(self, pixels=None, frame="emitter"):
        if pixels == None:
            obs_wvs = self.coordinates
        else:
            obs_wvs = self.indicies_to_coordinates(pixels)
        return self.observer_to_frame(obs_wvs, frame=frame)
    
    def get_index(self, wvs, frame="emitter", clip=False):
        shift_wvs = self.frame_to_observer(wvs, frame="emitter")
        if clip:
            return self.coordinates_to_indicies(shift_wvs, extrapolation="nearest")
        else:
            return self.coordinates_to_indicies(shift_wvs, extrapolation="linear")
    
    def observer_to_frame(self, observer_wvs, frame="emitter"):
        if frame == "emitter":
            delta_v = self.rv - self.barycenter_vel
        elif frame == "observer":
            delta_v = 0.0
        else:
            delta_v = frame - self.rv
        return observer_wvs*(1.0-delta_v/speed_of_light)
    
    def frame_to_observer(self, frame_wvs, frame="emitter"):
        if frame == "emitter":
            delta_v = self.rv - self.barycenter_vel
        elif frame == "observer":
            delta_v = 0.0
        else:
            delta_v = frame - self.barycenter_vel
        return frame_wvs*(1.0+delta_v/speed_of_light)
    
    def frame_conversion(self, wvs, wv_frame, target_frame):
        rest_wvs = self.frame_to_observer(wvs, frame=wv_frame)
        return self.observer_to_frame(rest_wvs, frame=target_frame)
    
    @property
    def dx_dlam(self):
        """find the gradient in pixel space which corresponds to
        a progression of wavelengths.
        """
        return 1.0/self.dlam_dx
    
    @property
    def dlam_dx(self):
        """find the gradients in wavelength space corresponding to 
        a progression of pixels.
        """
        return scipy.gradient(self.wv)    


class SpectralQuantity(object):
    
    def __init__(self, 
                 wavelength_solution, 
                 values, 
                 variances=None,
                 valuation_type="interpolate",
                 fill_value=0.0,
                 fill_var=np.inf):
        """a wavelength dependent quantity 
        
        wavelength_solution: WavelengthSolution
          the conversion from pixel to wavelength
        values: numpy.ndarray
          the value at each pixel
        variances: numpy.ndarray
          the associated uncertainty of the values
        valuation_type: string
          options 'interpolate', 'rebin'
        fill_value: float
          the value to return for wavelengths outside of the solution
        fill_var: float
          the variance for points outside of the wavelength solution.
        """
        if not isinstance(wavelength_solution, WavelengthSolution):
            wavelength_solution = WavelengthSolution
        self._wvsol = wavelength_solution
        self.values = np.asarray(values)
        self.set_variance()
        self.valuation_type = valuation_type
        self.fill_value = fill_value
        self.fill_var = fill_var
        
    
    def set_variance(self, variances):
        if not variances is None:
            variances = tmb.utils.misc.clean_variances(variances)
            self._var = variances
            self._inv_var = tmb.utils.misc.var_2_inv_var(variances)
        else:
            self._var = None
            self._inv_var = None
    
    @property
    def ivar(self):
        if self._inv_var is None:
            return None
        return self._inv_var
    
    @ivar.setter
    def ivar(self, value):
        var = tmb.utils.misc.inv_var_2_var(value)
        self.set_variance(var)
    
    @property
    def var(self):
        return self._var
    
    @var.setter
    def var(self, value):
        self.set_variance(value)

class Spectrum(object):
    """A representation of a collection of relative flux measurements
    """
    
    def __init__(self,
                 wavelengths, 
                 flux, 
                 inv_var=None,
                 rv=None,
                 barycenter_vel=None,
                 lsf=None,
                 norm="ones", 
                 metadata=None,
                 flags=None
             ):
        """makes a spectrum from a wavelength solution, flux and optional inv_var
        """
        if isinstance(wavelengths, WavelengthSolution):
            self.wv_soln = wavelengths
        else:
            #TODO proper error handling for odd inputs
            self.wv_soln = WavelengthSolution(wavelengths, rv=rv, barycenter_vel=barycenter_vel, lsf=lsf)
        
        # TODO: check that the dimensions of the inputs match
        self.flux = np.asarray(flux)
        if inv_var == None:
            self.inv_var = (self.flux > 0.0)*np.ones(self.flux.shape, dtype=float)
            inv_var = tmb.utils.misc.smoothed_mad_error(self, 1.0)
        self.inv_var = tmb.utils.misc.clean_inverse_variances(inv_var)
        #the memory address of the last stored transform
        if norm == "auto":
            norm_res = tmb.utils.misc.approximate_normalization(self, overwrite=True)
        elif norm == "ones":
            self.norm = np.ones(len(self.wv))
        else:
            self.norm = norm
        self._last_rebin_wv_soln_id = None
        self._last_rebin_transform = None
        
        if metadata is None:
            metadata = MetaData()
        elif not isinstance(metadata, MetaData):
            metadata = MetaData(metadata)
        self.metadata = metadata
        
        if flags is None:
            flags = SpectrumFlags()
        else:
            flags = SpectrumFlags(int(flags))
        self.flags = flags
    
    def __add__(self, other):
        if not isinstance(other, Spectrum):
            try:
                other = float(other)
                return Spectrum(self.wv_soln, self.flux+other, self.inv_var)
            except TypeError:
                raise TypeError("operation not supported between Spectrum and type %s" % type(other))
        rebinned_other = other.rebin(self.wv_soln)
        return Spectrum(self.wv_soln, self.flux+rebinned_other.flux, tmb.utils.misc.var_2_inv_var(self.var + other.var))
    
    def __div__(self, other):
        if not isinstance(other, Spectrum):
            try:
                other = float(other)
                return Spectrum(self.wv_soln, self.flux/other, self.inv_var*other**2)
            except TypeError:
                raise TypeError("operation not supported between Spectrum and type %s" % type(other))
        rebinned_other = other.rebin(self.wv_soln)
        return Spectrum(self.wv_soln, self.flux-rebinned_other.flux, tmb.utils.misc.var_2_inv_var(self.var/other.flux**2 + (self.flux/other.flux**2)**2*other.var))
    
    def __equal__ (self,other):
        if not isinstance(other,Spectrum):
            return False
        
        checks = [np.all(other.wv==self.wv),
                  np.all(other.flux==self.flux),
                  np.all(other.inv_var==self.inv_var),
                  other.metadata==self.metadata]
        return np.all(checks)
    
    def __len__(self):
        return len(self.flux)
    
    def __mul__(self, other):
        if not isinstance(other, Spectrum):
            try:
                other = float(other)
                return Spectrum(self.wv_soln, self.flux*other, self.inv_var/other**2)
            except TypeError:
                raise TypeError("operation not supported between Spectrum and type %s" % type(other))
        rebinned_other = other.rebin(self.wv_soln)
        return Spectrum(self.wv_soln, self.flux*rebinned_other.flux, tmb.utils.misc.var_2_inv_var(self.var*other.flux + other.var*self.flux))
    
    def __repr__(self):
        wvs = self.wv
        last_wv = wvs[-1]
        first_wv = wvs[0]
        return "<`thimbles.Spectrum` ({0:8.3f},{1:8.3f})>".format(first_wv, last_wv)
    
    def __sub__(self, other):
        if not isinstance(other, Spectrum):
            try:
                other = float(other)
                return Spectrum(self.wv_soln, self.flux-other, self.inv_var)
            except TypeError:
                raise TypeError("operation not supported between Spectrum and type %s" % type(other))
        rebinned_other = other.rebin(self.wv_soln)
        return Spectrum(self.wv_soln, self.flux-rebinned_other.flux, tmb.utils.misc.var_2_inv_var(self.var + other.var))
    
    @property
    def px(self):
        return self.wv_soln.get_index()
    
    @property
    def rv(self):
        return self.wv_soln.get_rv()
    
    @property
    def wv(self):
        """
        This returns the default values from get_wvs
        """
        return self.get_wvs(pixels=None, frame='emitter')
    
    def set_rv(self, rv):
        self.wv_soln.set_rv(rv)
    
    def get_rv(self):
        return self.wv_soln.get_rv()
    
    def get_wvs(self, pixels=None, frame="emitter"):
        return self.wv_soln.get_wvs(pixels, frame)
    
    def get_index(self, wvs, frame="emitter", clip=False):
        return self.wv_soln.get_index(wvs, frame=frame, clip=clip)
    
    def get_inv_var(self):
        return self.inv_var
    
    def get_var(self):
        #TODO deal with zeros appropriately
        return tmb.utils.misc.inv_var_2_var(self.inv_var)
    
    def normalize(self, **kwargs):
        #TODO: put extra controls in here
        norm_res = tmb.utils.misc.approximate_normalization(self, overwrite=True, **kwargs)    
        return norm_res
    
    def normalized(self):
        nspec = Spectrum(self.wv_soln, self.flux/self.norm, self.get_inv_var()*self.norm**2)
        nspec.flags["normalized"] = True
        return nspec
    
    def rebin_new(self, coords, kind=None, coord_type=None, fill=None):
        """valuate the spectrum at a given set of coordinates
        the method for determining the output spectrum values is dictated
        by this spectrum's valuation policy by default.

        coords: WavelengthSolution or numpy.ndarray
          the coordinates centers of the bins in the output spectrum
          if a np.ndarray is specified it will be assumed that the line
          spread function is gaussian with a sigma width parameter equal to
          the spacing between the pixels (Nyquist sampled).
        kind: string
          "interp", the flux values will be linearly interpolated
          "rebin", a rebinning matrix is built which redistributes flux 
            according to pixel overlap and line spread functions.
            An attempt is made to build a differential line spread function
            to sample 
        """
        pass
    
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
    
    def sample(self, wvs, frame="emitter"):
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
    
    def plot(self, axes=None, frame="emitter", **mpl_kwargs):
        plot_wvs = self.get_wvs(frame=frame)
        plot_flux = self.flux
        if axes == None:
            axes = plt.figure().add_subplot(111)
            xlabel = 'Wavelength in '+str(frame)+" frame"
            axes.set_xlabel(xlabel)
            axes.set_ylabel('Flux')
        l, = axes.plot(plot_wvs, plot_flux, **mpl_kwargs)
        return axes,l
    
    @property
    def var(self):
        return tmb.utils.misc.inv_var_2_var(self.inv_var)
    
    @property
    def ivar(self):
        return self.inv_var
