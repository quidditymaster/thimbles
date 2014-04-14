
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

class LogLinear (WavelengthSolution):
    pass

class Polynomial (WavelengthSolution):
    
    
    def __init__ (self,pixels, coefficients, **kwargs): 
        self.coefficients= coefficients 
        self.pixels = pixels 
        super(Polynomial, self).__init__(**kwargs)

    def get_wvs(self, pixels, fra):


class Linear (WavelengthSolution):
    pass 

class ChebyshevPolynomial (WavelengthSolution):
    pass

class CubicSpline (WavelengthSolution):
    pass

class LegendrePolynomial (WavelengthSolution):
    pass    
    



