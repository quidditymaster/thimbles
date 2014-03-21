import scipy.special
import numpy as np
import scipy.optimize

import thimbles as tmb

sqrt2pi = np.sqrt(2*np.pi)

def _gauss(wvs, center, g_width):
    return 1.0/(sqrt2pi*g_width)*np.exp(-0.5*((wvs-center)/g_width)**2)

def _voigt(wvs, center, g_width, l_width):
    "returns a voigt profile with gaussian sigma g_width and lorentzian width l_width."
    g_w = np.abs(g_width)
    l_w = np.abs(l_width)
    if l_w == 0:
        return _gauss(wvs, center, g_width)
    elif g_w == 0 and l_width != 0:
        g_w = 0.0001*l_w
    elif g_w == 0 and l_width == 0:
        dirac_delta = np.zeros(len(wvs), dtype=float)
        dirac_delta[int(len(wvs)/2)] = 1.0
        return dirac_delta
    z = ((wvs-center)+l_w*1j)/(g_w*np.sqrt(2))
    cplxv = scipy.special.wofz(z)/(g_w*np.sqrt(2*np.pi))
    return (cplxv.real).copy()

def _rotational(wvs, center, vsini, limb_darkening = 0):
    "for wavelengths in angstroms and v*sin(i) in km/s"
    ml = np.abs(vsini/3e5*center)
    eps = limb_darkening
    deltas = wvs-center
    deltas = deltas * (np.abs(deltas) <= ml)
    indep_part = 2*(1-eps)*np.power(1-np.power(deltas/ml, 2.0), 0.5)
    limb_part = 0.5*np.pi*eps*(1-np.power(deltas/ml, 2.0))
    norm = np.pi*ml*(1-eps/3.0)
    result = (np.abs(wvs-center) <= ml)*(indep_part + limb_part)/norm
    if np.sum(np.abs(result)) == 0: #rotation too small to detect at this resolution
        print "rot_too small", len(wvs)
        nwvs = len(wvs)
        result[int(nwvs/2)] = 1.0
    return result

def _voigt_rotational(wvs, center, g_width, l_width, vsini, limb_darkening): #this routine has edge problems with the convolution
    "generates a voigt profile convolved with a rotational broadening profile"
    rprof = _rotational(wvs, center, vsini, limb_darkening)
    if len(wvs) % 2 == 1:
        array_central_wv = wvs[int(len(wvs)/2)]
    else:
        array_central_wv = 0.5*(wvs[int(len(wvs)/2)] + wvs[int(len(wvs)/2)-1])
    #if center is not the central wavelength
    vprof = _voigt(wvs, array_central_wv, g_width, l_width)
    #plt.plot(vprof)
    return np.convolve(rprof, vprof, mode="same")/np.convolve(np.ones(wvs.shape), rprof, mode="same")

class LineProfile:
    
    def __call__(self, wvs, param_vec=None):
        return self.get_profile(wvs, param_vec)

class Voigt(LineProfile):
    
    def __init__(self, center, param_vec):
        """center: the central wavelength
        line_parameters: a vector whose elements go
        offset, g_width, l_width = parameters
        """
        self.center = center
        self.param_vec = np.asarray(param_vec)
    
    def set_center(self, center):
        self.center = center
    
    def set_parameters(self, vec):
        """offset, g_width, l_width = vec"""
        self.param_vec = np.asarray(vec)
    
    def get_parameters(self):
        return self.param_vec
    
    def get_profile(self, wvs, param_vec=None):
        if param_vec == None:
            param_vec = self.param_vec
        offset, g_width, l_width = param_vec
        return _voigt(wvs, self.center+offset, g_width, l_width)

class Gaussian(LineProfile):
    
    def __init__(self, center, param_vec):
        self.center = center
        self.param_vec = np.asarray(param_vec)
    
    def set_center(self, center):
        self.center = center
    
    def set_offset(self, offset):
        self.param_vec[0] = offset
    
    def set_parameters(self, vec):
        """offset, g_width = vec"""
        self.param_vec = vec
    
    def get_parameters(self):
        return self.param_vec
    
    def get_profile(self, wvs, param_vec=None):
        if param_vec == None:
            param_vec = self.param_vec
        offset, g_width = param_vec
        gnorm = 1.0/np.abs(g_width*sqrt2pi)
        return gnorm*np.exp(-0.5*((wvs-(self.center+offset))/g_width)**2)


class HalfGaussian(LineProfile):
    
    def __init__(self, center, param_vec):
        self.center = center
        #offset, left_g_width, right_g_width = param_vec
        self.param_vec = np.asarray(param_vec)
    
    def set_parameters(self, vec):
        """offset, g_width = vec"""
        self.param_vec = vec
    
    def get_parameters(self):
        return self.param_vec
    
    def set_center(self, center):
        self.center = center
    
    def get_profile(self, wvs, param_vec=None):
        if param_vec == None:
            param_vec = self.param_vec
        offset, left_g_width, right_g_width = param_vec 
        l_integ = (left_g_width*sqrt2pi)
        r_integ = (right_g_width*sqrt2pi)
        #pick the average of left and right norms
        avg_norm = 2.0/(l_integ + r_integ)
        sig_vec = np.where(wvs<self.center, self.left_g_width, self.right_g_width)
        return avg_norm*np.exp(-(wvs-self.center)**2/(2*sig_vec**2))


def profile_fit(x, y, y_inv_var, center, profile, offset_sigma=None, additive_background=True):
    """ fits a profile 
    
    x: ndarray
        the x points
    y: ndarray
        the y points
    y_inv_var: ndarray
        the inverse variances associated with the y values
    center: float
        the central x value of the profile
    profile: string or LineProfile
        if a LineProfile object is given it's parameters will be modified to 
        the best fit parameters. If 'voigt' or 'gaussian' is specified then 
        a new instance of that profile type will be created and its parameters
        fit.
    offset_sigma: float or None
        offsets of the profile are subjected to a gaussian prior centered
        around zero with width parameter offset_sigma. If None then 
        offset_sigma will be set to roughly a half pixel 0.5*x([1]-x[0]) 
    """
    if offset_sigma == None:
        offset_sigma = 0.5*(x[1]-x[0]) 
    if profile == "gaussian":
        sig_start = 2.5*(x[1]-x[0])
        profile = Gaussian(center, [0.0, sig_start])
    elif profile == "voigt":
        sig_start = 2.5*(x[1]-x[0])
        profile = Voigt(center, [0.0, sig_start, 0.0])

    tmb.modeling

#def voigt_fit(x, y, y_error, center):
#    return profile_fit(x, y, y_error, center, profile="voigt")
#
#def gauss_fit(x, y, y_error, center):
#    return profile_fit(x, y, y_error, center, profile="gaussian")