import scipy.special
import numpy as np
import scipy.optimize

import thimbles as tmb

sqrt2pi = np.sqrt(2*np.pi)

profile_functions = []

def gauss(wvs, center, g_width):
    return 1.0/np.abs(sqrt2pi*g_width)*np.exp(-0.5*((wvs-center)/g_width)**2)

profile_functions.append(gauss)

def half_gauss(wvs, center, l_width, r_width):
    #pick the average of left and right norms
    avg_norm = 2.0/np.abs(sqrt2pi*(l_width + r_width))
    sig_vec = np.where(wvs< center, l_width, r_width)
    return avg_norm*np.exp(-(wvs-center)**2/(2*sig_vec**2))

profile_functions.append(half_gauss)

def voigt(wvs, center, g_width, l_width):
    "returns a voigt profile with gaussian sigma g_width and lorentzian width l_width."
    g_w = np.abs(g_width)
    l_w = np.abs(l_width)
    if g_w == 0 and l_width == 0:
        dirac_delta = np.zeros(len(wvs), dtype=float)
        dirac_delta[int(len(wvs)/2)] = 1.0
        return dirac_delta
    elif l_w == 0:
        return gauss(wvs, center, g_width)
    elif g_w == 0 and l_width != 0:
        g_w = 0.0001*l_w
    z = ((wvs-center)+l_w*1j)/(g_w*np.sqrt(2))
    cplxv = scipy.special.wofz(z)/(g_w*np.sqrt(2*np.pi))
    return (cplxv.real).copy()

profile_functions.append(voigt)

def rotational(wvs, center, vsini, limb_darkening = 0):
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

profile_functions.append(rotational)

def voigt_rotational(wvs, center, g_width, l_width, vsini, limb_darkening): #this routine has edge problems with the convolution
    "generates a voigt profile convolved with a rotational broadening profile"
    rprof = rotational(wvs, center, vsini, limb_darkening)
    if len(wvs) % 2 == 1:
        array_central_wv = wvs[int(len(wvs)/2)]
    else:
        array_central_wv = 0.5*(wvs[int(len(wvs)/2)] + wvs[int(len(wvs)/2)-1])
    #if center is not the central wavelength
    vprof = voigt(wvs, array_central_wv, g_width, l_width)
    #plt.plot(vprof)
    return np.convolve(rprof, vprof, mode="same")/np.convolve(np.ones(wvs.shape), rprof, mode="same")

class LineProfile:
    
    def __init__(self, center, parameters, profile_func):
        self.center = center
        self.parameters = np.asarray(parameters)
        self.profile_func = profile_func

    def __call__(self, wvs, parameters=None):
        if parameters is None:
            parameters = self.parameters
        return self.profile_func(wvs, self.center, *parameters)

class Voigt(LineProfile):
    
    def __init__(self, center, parameters):
        """
        center: float
          the central wavelength of the profile
        param_vec: ndarray
          gaussian_width = param_vec[0]
          lorentz_width  = param_vec[1]
        """
        super(Voigt, self).__init__(center, parameters, voigt)


class Gaussian(LineProfile):
    
    def __init__(self, center, parameters):
        super(Gaussian, self).__init__(center, parameters, gauss)


class HalfGaussian(LineProfile):
    
    def __init__(self, center, parameters):
        super(Gaussian, self).__init__(center, parameters, half_gauss)


