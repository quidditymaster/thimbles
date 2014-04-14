

from .pixel_wavelength_function import (NoSolution, LogLinear, Polynomial, 
                                        Linear , ChebyshevPolynomial,
                                        CubicSpline, LegendrePolynomial) 

# ########################################################################### #

class MulitpleWavelengthSolution (Exception):
    pass

class IncompatibleWavelengthSolution (Exception):
    pass

class NoWavelengthSolutionError (Exception):
    pass

# ########################################################################### #

def from_w0 (header):
    pass

def from_crval (header):
    pass

def from_crvl (header):
    crvl = header.get("CRVL",None)
    if crvl is None:       
        raise IncompatibleWavelengthSolution("no keyword CRVL")

def from_ctype (header): 
    pass

def from_wcs (header):
    pass

def from_makee_wv (header):
    pass

def from_makee_c0 (header):
    pass

def from_spectre (header):
    """ Extract wavelengths from SPECTRE history keywords
    
    Parameters 
    ----------
    header : dict or `astropy.io.fits.header.Header`
    
    Returns
    -------
    wavelength_soln_list : list of `thimbles.spectrum.WavelengthSolution`
        Callable to take pixels to wavelengths        
    
    Raises
    ------
    IncompatibleWavelengthSolution : no SPECTRE wavelength solution identified
    
    """

    pass
    


