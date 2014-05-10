
import numpy as np
from ..spectrum import WavelengthSolution


__all__ = ['NoSolution','LogLinear','Linear','ChebyshevPolynomial',
           'CubicSpline','LegendrePolynomial']
    
# ########################################################################### #

class NoSolution (WavelengthSolution):
    """ No solution to wavelength solution """
    
    def __init__ (self,pixels,*args,**kwargs):
        print "no wavelength solution found, using pixel number"
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

class Linear (WavelengthSolution):
    """ Linear wavelength solution of the form wvs=c1*pix+c0
    
    Parameters
    ----------
    pixels : ndarray
    c1 : float
    c0 : float
    kwargs : passed to WavelengthSolution
    
    
    """
    
    def __init__ (self,pixels,c1,c0,**kwargs):
        print "Linear Wavelength Solution c0 %f c1 %f" % (c0, c1)
        wvs = c1*pixels+c0
        super(Linear, self).__init__(wvs, **kwargs)

class LogLinear(WavelengthSolution):
    """ Linear wavelength solution of the form wvs=10**(c1*pix+c0)
    
    Parameters
    ----------
    pixels : ndarray
    c1 : float
    c0 : float
    kwargs : passed to WavelengthSolution
    
    
    """
    def __init__(self, pixels, coefficients, **kwargs):
        wvs = 10**(coefficients[0]*pixels+coefficients[1])
        super(LogLinear, self).__init__(wvs, **kwargs)

class ChebyshevPolynomial (WavelengthSolution):
    """ Apply chebyshev polynomial 
    
    """
    def __init__ (self,pixels, coefficients, **kwargs):         
        if len(coefficients) != 5:            
            raise ValueError("this particular chebyshev uses 5 coefficients")
        #c20    p = (point - c(6))/c(7)
        #c      xpt = (2.*p-(c(9)+c(8)))/(c(9)-c(8))
        # !! is this right?
        n = len(pixels)
        xpts = (2.0*pixels - float(n+1))/float(n-1)        
        # xpt = (2.*point-real(npt+1))/real(npt-1)
        coeff = list(reversed(coefficients))
        wvs =  coeff[0] + xpts*coeff[1] + coeff[2]*(2.0*xpts**2.0-1.0) + coeff[3]*xpts*(4.0*xpts**2.0-3.0)+coeff[4]*(8.0*xpts**4.0-8.0*xpts**2.0+1.0)
        super(ChebyshevPolynomial, self).__init__(wvs,**kwargs)

class CubicSpline (WavelengthSolution):
    pass

class LegendrePolynomial (WavelengthSolution):
    """ Wavelength solution in using a Legendre polynomial 
    
    
    """    
    def __init__(self, pixels, coefficients, **kwargs):
        n = len(pixels)
        xpts = (2.0*pixels - (n+1))/float(n-1)
        wvs = coefficients[0]*xpts+coefficients[1]
        super(LegendrePolynomial, self).__init__(wvs)
