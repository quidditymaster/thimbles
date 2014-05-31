#
#
# ########################################################################### #
# standard libaray
import logging

# 3rd Party
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt

# internal
from .utils import resampling
from .utils import misc
from .utils.misc import inv_var_2_var, var_2_inv_var
from .metadata import MetaData
from .flags import SpectrumFlags
from . import verbosity
from .reference_frames import InertialFrame
from .resolution import LineSpreadFunction, GaussianLSF, BoxLSF
from .binning import CoordinateBinning1D

# ########################################################################### #

__all__ = ["WavelengthSolution","Spectrum"]

# ########################################################################### #

speed_of_light = 299792.458


class WavelengthSolution(CoordinateBinning1D):
    
    def __init__(self, obs_wavelengths, emitter_frame=None, observer_frame=None, lsf=None, **kwargs):
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
        lsf: LineSpreadFunction or numpy.ndarray
          the associated line spread function for each pixel.
          a 1D ndarray will be interpreted as a gaussian lsf width in pixels.
        """
        super(WavelengthSolution, self).__init__(obs_wavelengths)
        
        if observer_frame is None:
            observer_frame = InertialFrame(0.0)
        self.observer_frame = observer_frame
        
        if emitter_frame is None:
                emitter_frame = InertialFrame(0.0)
        self.emitter_frame = emitter_frame
        
        #TODO: include a radial velocity uncertainty (make it a subclass!)
        if lsf is None:
            lsf = GaussianLSF(np.ones(len(obs_wavelengths)))
        elif not isinstance(lsf, LineSpreadFunction):
            try:
                lsf = GaussianLSF(lsf)
            except:
                verbosity("bad LSF specification defaulting to box LSF")
                lsf = BoxLSF(self)
        self.lsf = lsf
    
    
    def get_wvs(self, pixels=None, frame="emitter"):
        if pixels == None:
            obs_wvs = self.obs_wvs
        else:
            obs_wvs = self.indicies_to_coordinates(pixels)
        return self.observer_to_frame(obs_wvs, frame=frame)
    
    def get_pix(self, wvs, frame="emitter"):
        shift_wvs = self.frame_to_observer(wvs, frame="emitter")
        return self.coordinates_to_indicies(shift_wvs)
    
    @property
    def wvs(self):
        """The emitter frame wavelengths"""
        return self.get_wvs()
    
    @property
    def obs_wvs(self):
        return self.coordinates
    
    
    def observer_to_frame(self, observer_wvs, frame="emitter"):
        if frame == "emitter":
            delta_v = self.emitter_frame - self.observer_frame
        elif frame == "observer":
            delta_v = 0.0
        else:
            delta_v = frame - self.observer_frame
        return observer_wvs*(1.0-delta_v.rv/speed_of_light)
    
    def frame_to_observer(self, frame_wvs, frame="emitter"):
        if frame == "emitter":
            delta_v = self.emitter_frame - self.observer_frame
        elif frame == "observer":
            delta_v = 0.0
        else:
            delta_v = frame - self.observer_frame
        return frame_wvs*(1.0+delta_v.rv/speed_of_light)
    
    def frame_conversion(self, wvs, wv_frame, target_frame):
        rest_wvs = self.frame_to_observer(wvs, frame=wv_frame)
        return self.observer_to_frame(rest_wvs, frame=target_frame)
    
    def set_rv(self, rv):
        self.emitter_frame.rv = rv
    
    def get_rv(self):
        return self.emitter_frame.rv
    
    #TODO: convenient/efficent functions to transform a set of pixel sigmas
    #into a set of wavelength sigmas given the current wavelength solution
    #and turn a set of wavelength sigmas into pixel sigmas.
    #def dx_dlam(self, wvs, frame="emitter"):
    #    """find the gradient in pixel space which corresponds to
    #    a progression of wavelengths.
    #    """
    #    return scipy.gradient(self.get_pix(wvs, frame))
    #
    #def dlam_dx(self, pixels, frame="emitter"):
    #    """find the gradients in wavelength space corresponding to 
    #    a progression of pixels.
    #    """
    #    return scipy.gradient(self.get_wvs(pixels, frame))    
    #
    #def pix_to_wv_delta(self, wvs, wv_sigmas, frame="emitter"):
    #    """given a set of wavelengths 
    #    """
    #    deriv = self.dx_dlam(wvs, frame)
    #    return deriv*wv_sigmas
    #
    #def wv_to_pix_sigma(self, pix_nums, pix_sigmas, frame="emitter"):
    #    deriv = self.dlam_dx(pix_nums, frame)
    #    return deriv*pix_sigmas

class Spectrum(object):
    """A representation of a collection of relative flux measurements
    """
    
    def __init__(self, 
                 wavelength_solution, 
                 flux, 
                 inv_var=None,
                 norm="ones", 
                 metadata=None,
                 flags=None
             ):
        """makes a spectrum from a wavelength solution, flux and optional inv_var
        """
        if isinstance(wavelength_solution, WavelengthSolution):
            self.wv_soln = wavelength_solution
        else:
            #TODO proper error handling for odd inputs
            self.wv_soln = WavelengthSolution(wavelength_solution)
        
        # TODO: check that the dimensions of the inputs match
        self.flux = flux
        if inv_var == None:
            self.inv_var = (flux > 0.0)*np.ones(flux.shape, dtype=float)
            inv_var = misc.smoothed_mad_error(self, 1.0, overwrite_error=True)
        self.inv_var = misc.clean_inverse_variances(inv_var)
        #the memory address of the last stored transform
        if norm == "auto":
            norm_res = misc.approximate_normalization(self, overwrite=True)
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
        return Spectrum(self.wv_soln, self.flux+rebinned_other.flux, var_2_inv_var(self.var + other.var))
    
    def __div__(self, other):
        if not isinstance(other, Spectrum):
            try:
                other = float(other)
                return Spectrum(self.wv_soln, self.flux/other, self.inv_var*other**2)
            except TypeError:
                raise TypeError("operation not supported between Spectrum and type %s" % type(other))
        rebinned_other = other.rebin(self.wv_soln)
        return Spectrum(self.wv_soln, self.flux-rebinned_other.flux, var_2_inv_var(self.var/other.flux**2 + (self.flux/other.flux**2)**2*other.var))
    
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
        return Spectrum(self.wv_soln, self.flux*rebinned_other.flux, var_2_inv_var(self.var*other.flux + other.var*self.flux))
    
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
        return Spectrum(self.wv_soln, self.flux-rebinned_other.flux, var_2_inv_var(self.var + other.var))
    
    @property
    def px(self):
        return self.wv_soln.get_pix()
    
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
    
    def get_pix(self, wvs, frame="emitter"):
        return self.wv_soln.get_pix(wvs, frame=frame)
    
    def get_inv_var(self):
        return self.inv_var
    
    def get_var(self):
        #TODO deal with zeros appropriately
        return inv_var_2_var(self.inv_var)
    
    def normalize(self):
        #TODO: put extra controls in here
        norm_res = misc.approximate_normalization(self, overwrite=True)    
        return norm_res
        
    def normalized(self):
        nspec = Spectrum(self.wv_soln, self.flux/self.norm, self.get_inv_var()*self.norm**2)
        nspec.flags["normalized"] = True
        return nspec
    
    def rebin_new(self, coords, kind=None, coord_type=None, normalized=None, fill_flux=None, fill_var=None, fill_norm=None, fill_lsf=None):
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
        index_vals = self.get_pix(wvs, frame=frame)
        upper_index = np.ceil(index_vals)
        lower_index = np.floor(index_vals)
        alphas = index_vals - lower_index
        interp_vals =  self.flux[upper_index]*alphas
        interp_vals += self.flux[lower_index]*(1-alphas)
        var = self.get_var()
        sampled_var = var[upper_index]*alphas**2
        sampled_var += var[lower_index]*(1-alphas)**2
        return Spectrum(wvs, interp_vals, misc.var_2_inv_var(sampled_var))
    
    def bounding_indexes(self, bounds, frame="emitter"):
        bvec = np.asarray(bounds)
        l_idx, u_idx = map(int, np.around(self.get_pix(bvec, frame=frame)))
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
        out_wvs = self.get_wvs(np.arange(l_idx, u_idx+1), frame=frame)
        if copy:
            out_flux = self.flux[l_idx:u_idx+1].copy()
            out_invvar = self.inv_var[l_idx:u_idx+1].copy()
            out_norm = self.norm[l_idx:u_idx+1].copy()
        else:
            out_flux = self.flux[l_idx:u_idx+1]
            out_invvar = self.inv_var[l_idx:u_idx+1]
            out_norm = self.norm[l_idx:u_idx+1]
        if len(out_flux) < 2:
            return None
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
        return inv_var_2_var(self.inv_var)
