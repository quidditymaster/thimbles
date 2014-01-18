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

class CoordinateBinning1d:
    """a container for a set of coordinates
    """
    def __init__(self, coordinates):
        """coordinates must be a monotonically decreasing sequence
        """
        self.coordinates = coordinates
        self.bins = centers_to_bins(coordinates)
        self.lb = self.bins[0]
        self.ub = self.bins[-1]
        self.last_bin = (self.bins[0], self.bins[1])
        self.last_bin_idx = 0
        self.start_dx = self.coordinates[1] - self.coordinates[0]
        self.end_dx = self.coordinates[-1] - self.coordinates[-2]
    
    def indicies_to_coordinates(self, input_indicies):
        out_coordinates = np.zeros(len(input_indicies))
        for i in range(len(input_indicies)):
            alpha = input_indicies[i]%1
            int_part = int(input_indicies[i])
            if input_indicies[i] < 0:
                out_coordinates[i] = self.coordinates[0] + alpha*self.start_dx
            elif input_indicies[i] > len(self.coordinates)-2:
                out_coordinates[i] = self.coordinates[-1] + alpha*self.end_dx
            else:
                out_coordinates[i] = self.coordinates[int_part]*(1.0-alpha)
                out_coordinates[i] += self.coordinates[int_part+1]*alpha
        return out_coordinates
    
    def coordinates_to_indicies(self, input_coordinates):
        """assign continuous indexes to the input_coordinates 
        which place them on to the indexing of these coordinates.
        """
        xv = np.asarray(input_coordinates)
        out_idx_vals = np.zeros(len(xv.flat), dtype = int)
        for x_idx in xrange(len(xv.flat)):
            cur_x = xv.flat[x_idx]
            #check if the last solution still works
            if self.last_bin[0] <= cur_x <= self.last_bin[1]:
                #if so find the fraction through the pixel
                alpha = (cur_x-self.last_bin[0])/(self.last_bin[1]-self.last_bin[0])
                out_idx_vals[x_idx] = self.last_bin_idx + alpha
                continue
            #make sure that the x value is inside the bin range or extrapolate
            if self.lb > cur_x:
                out_idx_vals[x_idx] = (cur_x-self.coordinates[0])/self.start_dx
                continue
            if self.ub < cur_x:
                out_idx_vals[x_idx] = (cur_x-self.coordinates[1])/self.end_dx
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
                    self.last_bin = self.bins[lbi], self.bins[lbi+1]
                    self.last_bin_idx = lbi
                    break
            alpha = (cur_x-self.last_bin[0])/(self.last_bin[1]-self.last_bin[0])
            out_idx_vals[x_idx] = lbi + alpha
        return out_idx_vals
    
    def get_bin_index(self, xvec):
        """uses interval splitting to quickly find the bin belonging to the input coordinates
        If a coordinate outside of the bins is asked for a linear extrapolation of the 
        bin index is returned. (so be warned indexes can be less than 0 and greaterh than n!)
        """
        xv = np.array(xvec)
        out_idxs = np.zeros(len(xv.flat), dtype = int)
        for x_idx in xrange(len(xv.flat)):
            cur_x = xvec[x_idx]
            #check if the last solution still works
            if self.last_bin[0] <= cur_x <= self.last_bin[1]:
                out_idxs[x_idx] = self.last_bin_idx
                continue
            #make sure that the x value is inside the bin range
            if self.lb > cur_x:
                out_idxs[x_idx] = int((cur_x-self.lb)/self.start_dx -1)
            if self.ub > cur_x:
                out_idxs[x_idx] = int((cur_x-self.ub)/self.end_dx)
            lbi, ubi = 0, self.n_bounds-1
            while True:
                mididx = (lbi+ubi)/2
                midbound = self.bins[mididx]
                if midbound <= cur_x:
                    lbi = mididx
                else:
                    ubi = mididx
                if self.bins[lbi] <= cur_x <= self.bins[lbi+1]:
                    self.last_bin = self.bins[lbi], self.bins[lbi+1]
                    self.last_bin_idx = lbi
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
        return np.array(self.coordinates_to_indicies(input_coordinates), dtype = int)

n_delts = 1024
z_scores = np.linspace(-6, 6, n_delts)
cdf_vals = scipy.stats.norm.cdf(z_scores)
min_z = z_scores[0]
max_z = z_scores[-1]
z_delta = (z_scores[1]-z_scores[0])

def approximate_gaussian_cdf(zscore):
    if zscore > max_z-z_delta-1e-5:
        return 1.0
    elif zscore < min_z:
        return 0
    idx_val = (zscore-min_z)/z_delta
    base_idx = int(idx_val)
    alpha = idx_val-base_idx
    return cdf_vals[base_idx]*(1-alpha) + cdf_vals[base_idx+1]*alpha

pass
# =========================================================================== #

class LineSpreadFunction:
    """a class for describing line spread functions.
    Integrations are carried out over pixel space. 
    """
    
    def integrate(self, index, lb, ub):
        lower_val = self.get_integral(index, lb)
        upper_val = self.get_integral(index, ub)
        return upper_val-lower_val

class GaussianLSF(LineSpreadFunction):
    
    def __init__(self,widths, max_sigma = 5.0, wv_soln=None):
        self.wv_soln = wv_soln
        self.widths = widths
        self.max_sigma = max_sigma
    
    def get_integral(self, index, pix_coord):
        zscore = (pix_coord-index)/self.widths[index]
        return approximate_gaussian_cdf(zscore)
    
    def get_coordinate_density_range(self, index):
        lb = index - self.max_sigma*self.widths[index]
        ub = index + self.max_sigma*self.widths[index]
        return lb, ub
    
    def get_rms_width(self, index):
        return self.widths[index]

class BoxLSF(LineSpreadFunction):
    
    def __init__(self, wv_soln=None):
        self.wv_soln = wv_soln
    
    def get_integral(self, index, pix_coord):
        if pix_coord > index+0.5:
            return 1
        elif pix_coord < index-0.5:
            return 0
        else:
            return pix_coord - index + 0.5
    
    def get_coordinate_density_range(self, index):
        return index-1, index+1
    
    def get_rms_width(self, index):
        return 1.0/12.0

class DiracLSF(LineSpreadFunction):
    
    def __init__(self, wv_soln=None):
        self.wv_soln = None
        self.epsilon = 1e-12
    
    def get_integral(self, index, pix_coord):
        if pix_coord >= index:
            return 1.0
        else:
            return 0.0
    
    def get_coordinate_density_range(self, index):
        return self.centers[index]-self.epsilon, self.centers[index]+self.epsilon
    
    def get_rms_width(self, index):
        return 0.0

#TODO: add a simple lsf convolution function not perfect but something.

pass
# =========================================================================== #
#TODO: we need to make the reference frames have true 3-space velocity vectors so that
# we can use the ra dec info to project the velocity difference onto 
class ReferenceFrame(object):
    """a point of reference specified with the earth sun barycenter as 0
    velocity. 
    """
    
    def __init__(self, barycenter_rv=0, ra=None, dec=None):
        #TODO: convert a velocity relative to the earth sun barycenter
        #and a position on the sky to a 3-space radial velocity vector.
        self._v = barycenter_rv
    
    def __sub__(self, other):
        """returns the magnitude of the radial velocity difference """
        return self._v - other._v
    
    def set_rv(self, v):
        self._v = v
    
    def get_rv(self):
        return self._v

class EarthSunBarycenterFrame(ReferenceFrame):
    
    def __init__(self):
        self._v = 0

class EarthCenterFrame(ReferenceFrame):
    
    def __init__(self, mjd):
        #TODO:
        raise NotImplemented

class EarthSurfaceFrame(ReferenceFrame):
    
    def __init__(self, latitude, longitude, mjd):
        #TODO:
        raise NotImplemented

class WavelengthSolution(CoordinateBinning1d):
    
    def __init__(self, obs_wavelengths, rv=None, emitter_frame=None, lsf=None, observer_frame=None):
        """a class that encapsulates the manner in which a spectrum is sampled
        pixel_num_array: an array of the pixel indicies
        wv_func; a function that takes pixel indicies and returns 
        wavelengths in the telescope frame
        rv: float
            if rv is specified and emitter_frame is None rv is assumed to 
            be the radial velocity of the emitter.
        emitter_frame: ReferenceFrame or None
            if emitter_frame  == None we assume emitter_frame == EarthSunBarycenterFrame
        lsf: the line spread function
        observer_frame: ReferenceFrame or None
        the frame in which the spectrum is observed if None defaults
        to EarthSunBarycenterFrame
        """
        CoordinateBinning1d.__init__(self, obs_wavelengths)
        if emitter_frame == None:
            if rv == None:
                emitter_frame = EarthSunBarycenterFrame()
            else:
                emitter_frame = ReferenceFrame(rv)
        self.emitter_frame = emitter_frame
        
        if observer_frame==None:
            observer_frame = EarthSunBarycenterFrame()
        self.observer_frame = observer_frame
        
        #TODO: include a radial velocity uncertainty (make it a subclass!)
        if lsf != None:
            if isinstance(lsf, LineSpreadFunction):
                self.lsf = lsf
            else:
                try:
                    self.lsf = GaussianLSF(np.ones(len(lsf)))
                except:
                    raise Exception("don't know what to do with this LSF!")
        else:
            self.lsf = BoxLSF(self)
    
    @property
    def wvs(self):
        """The emitter frame wavelengths"""
        return self.get_wvs()
    
    @property
    def obs_wvs(self):
        return self.coordinates
    
    def telescope_to_frame(self, telescope_wvs, frame="emitter"):
        if frame == "emitter":
            delta_v = self.emitter_frame - self.observer_frame
        elif frame == "telescope":
            delta_v = 0.0
        else:
            delta_v = frame - self.observer_frame
        return telescope_wvs*(1.0-delta_v/speed_of_light)
    
    def frame_to_telescope(self, frame_wvs, frame="emitter"):
        if frame == "emitter":
            delta_v = self.emitter_frame - self.observer_frame
        elif frame == "telescope":
            delta_v = 0.0
        else:
            delta_v = frame - self.observer_frame
        return frame_wvs*(1.0+delta_v/speed_of_light)
    
    def frame_conversion(self, wvs, wv_frame, target_frame):
        rest_wvs = self.frame_to_telescope(wvs, frame=wv_frame)
        return self.telescope_to_frame(rest_wvs, frame=target_frame)
    
    def get_wvs(self, pixels=None, frame="emitter"):
        if pixels == None:
            obs_wvs = self.obs_wvs
        else:
            obs_wvs = self.indicies_to_coordinates(pixels)
        return self.telescope_to_frame(obs_wvs, frame=frame)
    
    def get_pix(self, wvs, frame="emitter"):
        shift_wvs = self.frame_to_telescope(wvs)
        return self.coordinates_to_indicies(shift_wvs)
    
    def set_rv(self, rv):
        self.emitter_frame.set_v(rv)
    
    def get_rv(self):
        self.emitter_frame.get_v()
    
    
    #     def dx_dlam(self, wvs, frame="emitter"):        
    #         return scipy.misc.central_diff_weights(self.get_pix(wvs, frame))
    #     
    #     def dlam_dx(self, pixels, frame="emitter"):
    #         return scipy.misc.central_diff_weights(self.get_wvs(pixels, frame))    
    #     
    #def pix_sig_from_wv_sig(self, wvs, wv_sigmas, frame="emitter"):
    #    deriv = self.dx_dlam(wvs, frame)
    #    return deriv*wv_sigmas
    #
    #def wv_sig_from_pix_sig(self, pix_nums, pix_sigmas, frame="emitter"):
    #    deriv = self.dlam_dx(pix_nums, frame)
    #    return deriv*pix_sigmas

class Spectrum(object):
    
    def __init__(self, wavelength_solution, flux, inv_var=None, 
                 norm="auto",
                 name="",
                 **kwargs):
        """makes a spectrum from a wavelength solution, flux and optional inv_var
        """
        if isinstance(wavelength_solution, WavelengthSolution):
            self.wv_soln = wavelength_solution
        else:
            #TODO proper error handling for odd inputs
            self.wv_soln = WavelengthSolution(wavelength_solution)
        self.kwargs = kwargs
        # TODO: check that the dimensions of the inputs match
        self.flux = flux
        if inv_var == None:
            self.inv_var = (flux > 0.0)*np.ones(flux.shape, dtype=float)
            inv_var = misc.smoothed_mad_error(self, 1.0, overwrite_error=True)
        self.inv_var = inv_var
        #the memory address of the last stored transform
        if norm == "auto":
            norm_res = misc.approximate_normalization(self, overwrite=True)
        else:
            self.norm = norm
        self.name = name
        self._last_rebin_wv_soln_id = None
        self._last_rebin_transform = None
    
    def __repr__(self):
        wvs = self.wv
        last_wv = wvs[-1]
        first_wv = wvs[0]
        return "spectrum % 8.3f to % 8.3f" % (first_wv, last_wv)
    
    @property
    def wv(self):
        """
        This returns the default values from get_wvls
        return self.get_wvs(pixels=None,frame='emitter')
        """
        return self.get_wvs(pixels=None, frame='emitter')
    
    def get_wvs(self, pixels=None, frame="emitter"):    
        return self.wv_soln.get_wvs(pixels, frame)
    
    def get_pix(self, wvs, frame="emitter"):
        return self.wv_soln.get_pix(wvs, frame=frame)
    
    def get_inv_var(self):
        return self.inv_var
    
    def get_var(self):
        #TODO deal with zeros appropriately
        return inv_var_2_var(self.inv_var)
    
    def rebin(self, new_wv_soln, frame="emitter"):
        #check if we have the transform stored
        if self._last_rebin_wv_soln_id == id(new_wv_soln):
            transform = self._last_rebin_transform
        else:
            in_wvs = self.get_wvs(frame=frame)
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

class Continuum(Spectrum):
    
    def __init__(self, wv_soln, flux=None):
        wvs = wv_soln.get_wvs()
        if flux == None:
            flux = np.ones(wvs.shape)
        #TODO make for proper handling of continuum error propagation
        invvar = np.ones(wvs.shape)
        Spectrum.__init__(self, wv_soln, flux, invvar)
    
    def get_continuum(self):
        #TODO make the continuum a black body 
        return self.flux
    
    def get_normalization(self):
        return self.get_continuum()
    
    def get_parameter_list(self):
        return self.parameter_list
    

