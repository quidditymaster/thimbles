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
from .utils.misc import inv_var_2_var
from .metadata import MetaData
from .flags import SpectrumFlags
from . import verbosity
from .reference_frames import InertialFrame
from .resolution import LineSpreadFunction, GaussianLSF, BoxLSF

# ########################################################################### #

__all__ = ["WavelengthSolution","Spectrum"]

# ########################################################################### #

speed_of_light = 299792.458

def centers_to_bins(coord_centers):
    if len(coord_centers) == 0:
        return np.zeros(0)
    bins = np.zeros(len(coord_centers) + 1, dtype = float)
    bins[1:-1] = 0.5*(coord_centers[1:] + coord_centers[:-1])
    bins[0] = coord_centers[0] - (bins[1]-coord_centers[0])
    bins[-1] = coord_centers[-1] + 0.5*(coord_centers[-1] - coord_centers[-2])
    return bins

class CoordinateBinning1d (object):
    """a container for a set of coordinates
    """
    def __init__(self, coordinates):
        """coordinates must be a monotonically decreasing sequence
        """
        self.coordinates = coordinates
        self.bins = centers_to_bins(coordinates)
        self._cached_prev_bin = (self.bins[0], self.bins[1])
        self._cached_prev_bin_idx = 0
        self._start_dx = self.coordinates[1] - self.coordinates[0]
        self._end_dx = self.coordinates[-1] - self.coordinates[-2]
    
    def indicies_to_coordinates(self, input_indicies):
        # TODO: See if scipy/numpy has auto search
        out_coordinates = np.zeros(len(input_indicies))
        for i in range(len(input_indicies)):
            alpha = input_indicies[i]%1
            int_part = int(input_indicies[i])
            if input_indicies[i] < 0:
                out_coordinates[i] = self.coordinates[0] + alpha*self._start_dx
            elif input_indicies[i] > len(self.coordinates)-2:
                out_coordinates[i] = self.coordinates[-1] + alpha*self._end_dx
            else:
                out_coordinates[i] = self.coordinates[int_part]*(1.0-alpha)
                out_coordinates[i] += self.coordinates[int_part+1]*alpha
        return out_coordinates
    
    def coordinates_to_indicies(self, input_coordinates):
        """assign continuous indexes to the input_coordinates 
        which place them on to the indexing of these coordinates.
        """
        # get the upper and lower bounds for the coordinates
        lb,ub = self.bins[0],self.bins[-1]
        xv = np.asarray(input_coordinates)
        out_idx_vals = np.zeros(len(xv.flat), dtype = int)
        for x_idx in xrange(len(xv.flat)):
            cur_x = xv.flat[x_idx]
            #check if the last solution still works
            if self._cached_prev_bin[0] <= cur_x <= self._cached_prev_bin[1]:
                #if so find the fraction through the pixel
                alpha = (cur_x-self._cached_prev_bin[0])/(self._cached_prev_bin[1]-self._cached_prev_bin[0])
                out_idx_vals[x_idx] = self._cached_prev_bin_idx + alpha
                continue
            #make sure that the x value is inside the bin range or extrapolate
            if lb > cur_x:
                out_idx_vals[x_idx] = (cur_x-self.coordinates[0])/self._start_dx
                continue
            if ub < cur_x:
                out_idx_vals[x_idx] = (cur_x-self.coordinates[1])/self._end_dx
                continue
            lbi, ubi = 0, len(self.bins)-1
            while True:
                mididx = (lbi+ubi)/2
                midbound = self.bins[mididx]
                if midbound <= cur_x:
                    lbi = mididx
                else:
                    ubi = mididx
                if self.bins[lbi] <= cur_x <= self.bins[lbi+1]:
                    self._cached_prev_bin = self.bins[lbi], self.bins[lbi+1]
                    self._cached_prev_bin_idx = lbi
                    break
            alpha = (cur_x-self._cached_prev_bin[0])/(self._cached_prev_bin[1]-self._cached_prev_bin[0])
            out_idx_vals[x_idx] = lbi + alpha
        return out_idx_vals
    
    def get_bin_index(self, xvec):
        """uses interval binary search to quickly find the bin belonging to the 
        input coordinates. If a coordinate outside of the bins is asked for a 
        linear extrapolation of the bin index is returned. (so be warned 
        indexes can be less than 0 and greater than len!)
        """
        # get the upper and lower bounds for the coordinates
        lb,ub = self.bins[0],self.bins[-1]
        xv = np.asarray(xvec)
        out_idxs = np.zeros(len(xv.flat), dtype = int)
        for x_idx in xrange(len(xv.flat)):
            cur_x = xvec[x_idx]
            #check if the last solution still works
            if self._cached_prev_bin[0] <= cur_x <= self._cached_prev_bin[1]:
                out_idxs[x_idx] = self._cached_prev_bin_idx
                continue
            #make sure that the x value is inside the bin range
            if lb > cur_x:
                out_idxs[x_idx] = int((cur_x-lb)/self._start_dx -1)
            if ub > cur_x:
                out_idxs[x_idx] = int((cur_x-ub)/self._end_dx)
            lbi, ubi = 0, self.n_bounds-1
            while True:
                mididx = (lbi+ubi)/2
                midbound = self.bins[mididx]
                if midbound <= cur_x:
                    lbi = mididx
                else:
                    ubi = mididx
                if self.bins[lbi] <= cur_x <= self.bins[lbi+1]:
                    self._cached_prev_bin = self.bins[lbi], self.bins[lbi+1]
                    self._cached_prev_bin_idx = lbi
                    break
            out_idxs[x_idx] = lbi
        return out_idxs
    
class LinearBinning1d(CoordinateBinning1d):
    
    def __init__(self, x0, dx):
        self.x0 = x0
        self.dx = dx
    
    def coordinates_to_indicies(self, input_coordinates):
        return (np.asarray(input_coordinates) - self.x0)/self.dx
    
    def indicies_to_coordinates(self, input_indicies):
        return np.asarray(input_indicies)*self.dx + self.x0
    
    #def map_indicies(self, input_coordinates):
    #    return (input_coordinates-self.x0)/self.dx
    
    def get_bin_index(self, input_coordinates):
        #TODO make this handle negatives properly
        return np.asarray(self.coordinates_to_indicies(input_coordinates), dtype = int)


class WavelengthSolution(CoordinateBinning1d):
    
    def __init__(self, obs_wavelengths, emitter_frame=None, observer_frame=None, lsf=None):
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
        CoordinateBinning1d.__init__(self, obs_wavelengths)
        
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
    
    def __len__(self):
        return len(self.flux)
    
    def __equal__ (self,other):
        if not isinstance(other,Spectrum):
            return False
        
        checks = [np.all(other.wv==self.wv),
                  np.all(other.flux==self.flux),
                  np.all(other.inv_var==self.inv_var),
                  other.metadata==self.metadata]
        return np.all(checks)
    
    def __repr__(self):
        wvs = self.wv
        last_wv = wvs[-1]
        first_wv = wvs[0]
        return "<`thimbles.Spectrum` ({0:8.3f},{1:8.3f})>".format(first_wv, last_wv)
    
    def normalize(self):
        #TODO: put extra controls in here
        norm_res = misc.approximate_normalization(self, overwrite=True)
    
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
    
    def normalized(self):
        nspec = Spectrum(self.wv_soln, self.flux/self.norm, self.get_inv_var()*self.norm**2)
        nspec.flags["normalized"] = True
    
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
        self.coords = coords
    
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
