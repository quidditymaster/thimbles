
import numpy as np
from ..spectrum import WavelengthSolution
    

__all__ = ['NoSolution','LogLinear','Polynomial','Linear','ChebyshevPolynomial',
           'CubicSpline','LegendrePolynomial']
    
# ########################################################################### #

class NoSolution (WavelengthSolution):
    """ No solution to wavelength solution """
    
    def __init__ (self,pixels,*args,**kwargs):
        super(NoSolution, self).__init__(pixels,*args,**kwargs)
        
    def get_wvs (self,pixels=None,frame='emitter'):
        """ Takes pixels to wavelengths
        
        Parameters
        ----------
        pixels : ndarray, shape=(N,)
        start_pixel : integer
            Will offset the pixels 
        
        Returns
        -------
        wavelengths : ndarray, shape=(N,)
        
        """        
        if pixels is None:
            pixels = self.coordinates 
        return np.asarray(pixels)
    
    # TODO: __call__.__doc__ = get_wvs.__doc__
    
    def get_pix (self,wvs,frame='emiiter'):
        """ Take wavelengths and return pixels """
        return wvs

class Polynomial (WavelengthSolution):
    """
    
    Parameters
    ----------
    pixels : ndarray
    coefficients : ndarray, coefficients in decreasing order
    
    """
    
    def __init__ (self, pixels, coefficients, **kwargs):         
        self.coefficients = np.asarray(coefficients) 
        self.pixels = pixels 
        obs_wvs = self.get_wvs(pixels)
        super(Polynomial, self).__init__(obs_wvs,**kwargs)

    def get_wvs(self, pixels=None, frame='emitter'):
        if pixels is None:
            pixels = self.pixels
        return np.lib.polyval(self.coefficients,pixels)

class Linear (Polynomial):
    """ Linear wavelength solution of the form wvs=c1*pix+c0
    
    Parameters
    ----------
    pixels : ndarray
    c1 : float
    c0 : float
    kwargs : passed to WavelengthSolution
    
    
    """
    
    def __init__ (self,pixels,c1,c0,**kwargs):
        coefficients = np.array([c1,c0]).astype(float)
        Polynomial.__init__(self,pixels,coefficients,**kwargs)
    
    def get_wvs (self,pixels=None,frame='emitter'):
        if pixels is None:
            pixels = self.pixels
        return self.coefficients[0]*pixels+self.coefficients[1]

    def get_pix (self, wvs, frame="emitter"):
        if self.coefficients[0] == 0:
            return np.repeat(np.nan,len(wvs))
        return (wvs-self.coefficients[1])/self.coefficients[0]

class LogLinear (Linear):
    """ Linear wavelength solution of the form wvs=10**(c1*pix+c0)
    
    Parameters
    ----------
    pixels : ndarray
    c1 : float
    c0 : float
    kwargs : passed to WavelengthSolution
    
    
    """
    def get_wvs(self, pixels=None, frame="emitter"):
        if pixels is None:
            pixels = self.pixels
        return 10**(self.coefficients[0]*pixels+self.coefficients[1])
            
    def get_pix(self, wvs, frame="emitter"):
        if self.coefficients[0] == 0:
            return np.repeat(np.nan,len(wvs))
        return (np.log10(wvs)-self.coefficients[1])/self.coefficients[0]
        
class ChebyshevPolynomial (Polynomial):
    """ Apply chebyshev polynomial 
    
    
    """
    def __init__ (self,pixels, coefficients, **kwargs):         
        if len(coefficients) != 5:            
            raise ValueError("this particular chebyshev uses 5 coefficients")        
        Polynomial.__init__(self,pixels,coefficients,**kwargs)        
        
    def get_wvs(self, pixels=None, frame='emitter'):
        if pixels is None:
            pixels = self.pixels
        #c20    p = (point - c(6))/c(7)
        #c      xpt = (2.*p-(c(9)+c(8)))/(c(9)-c(8))
        # !! is this right?
        n = len(pixels)
        xpts = (2.0*pixels - float(n+1)/(n-1))        
        # xpt = (2.*point-real(npt+1))/real(npt-1)
        coeff = list(reversed(self.coefficients))
        return coeff[0] + xpts*coeff[1] + coeff[2]*(2.0*xpts**2.0-1.0) + coeff[3]*xpts*(4.0*xpts**2.0-3.0)+coeff[4]*(8.0*xpts**4.0-8.0*xpts**2.0+1.0)
                
class CubicSpline (WavelengthSolution):
    pass

class LegendrePolynomial (Polynomial):
    """ Wavelength solution in using a legrandre polynomial 
    
    
    """    
    def get_wvs(self, pixels=None, frame='emitter'):
        if pixels is None:
            pixels = self.pixels
        n = len(pixels)
        xpts = (2.0*pixels - (n+1)/(n-1))
        return self.coefficients[0]*xpts+self.coefficients[1]
        
    def get_pix(self, wvs, frame="emitter"):
        n = len(wvs)
        return ((wvs-self.coefficients[1])/self.coefficients[0])/2.0+(n+1)/(n-1)
        
        