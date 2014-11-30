import warnings
import numpy as np
from thimbles.sqlaimports import *
from thimbles import logger

__all__ = """
from_chebyshev 
from_legendre
""".split()

# ########################################################################### #

def from_chebyshev(pixels, coefficients):
    # THIS VERSION TAKEN FROM SPECTRE
    #c20    p = (point - c(6))/c(7)
    #c      xpt = (2.*p-(c(9)+c(8)))/(c(9)-c(8))
    # !! is this right?    
    #transforming coefficients
    #wvs =  coeff[0] + xpts*coeff[1] + coeff[2]*(2.0*xpts**2.0-1.0) + coeff[3]*xpts*(4.0*xpts**2.0-3.0)+coeff[4]*(8.0*xpts**4.0-8.0*xpts**2.0+1.0)
    logger("chebyshev with coefficients {}".format(coefficients))
    n = len(pixels)
    xpts = (2.0*pixels - float(n+1))/float(n-1)        
    return np.polynomial.chebyshev.chebval(xpts, coefficients)


def from_legendre(pixels, coefficients):
    logger("generating wavelengths from legendre polynomial coefficients {}".format(coefficients))
    pixels = np.asarray(pixels)
    n = len(pixels)
    xpts = (2.0*pixels - float(n+1))/float(n-1)
    return np.polyval(coefficients, xpts)
