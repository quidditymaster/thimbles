# This module is for user defined read functions. When read in file is done
# the code will also come to this module and give the user any of the options
# given here
#

__all__ = ['read_hectochelle', "read_mike"]

# ########################################################################### #
# Import Modules
import numpy as np
import os
import sys
import re
from astropy.io import fits
from ..utils.misc import inv_var_2_var, var_2_inv_var
from .io import Information

import pointsource

# ########################################################################### #

# TODO : add more instructions for how to create a  user read function

pass
# ########################################################################### #
# Example for a user defined read in function
#
# Further explanation of read function...
#

def read_user_defined (filepath,arg1,arg2='12',arg3='default'):
    """
    Optional doc string
    """
    # First thing always passed is the file path
    if not os.path.isfile(filepath):
        raise IOError("File does not exist")
    
    # all arguments are passed as strings so the user 
    # needs to cast them to the correct type
    arg1 = float(arg1)
    arg2 = int(arg2)
    arg3
    
    # then you can do whatever you need to read in the data
    # to wavelength, flux, and inverse variance (inv_var)
    wavelength = []
    flux = []
    inv_var = []
    
    # then return a spectral measurement list and information
    spectral_measurement = Spectrum(wavelength,flux,inv_var)
    measurement_list = []
    measurement_list.append(spectral_measurement)
    information = Information()
    information['filename'] = filepath
    
    return measurement_list,information

def read_mike(filepath):
    return pointsource.io.read_fits(filepath, band=1)

def read_hectochelle (filepath):
    # first thing always passed is the file path
    if not os.path.isfile(filepath):
        raise IOError("File does not exist")

    hdu = fits.open(filepath)[0]
    crval = hdu.header['CRVAL1']
    cdelt = hdu.header['CDELT1']
    crpix = hdu.header['CRPIX1']
    flux = hdu.data[0]
    inv_var = var_2_inv_var(hdu.data[1])
    
    #barycentric velocity correction
    bcv = hdu.header['BCV']
    
    wavelength = (np.arange(len(flux))-crpix)*cdelt + crval
    wavelength *= (1.0-bcv/299796458.0)

    spectral_measurement = Spectrum(wavelength,flux,inv_var)
    measurement_list = [spectral_measurement]
    information = Information()
    information['filename'] = filepath
    return measurement_list,information

    
  
    
    