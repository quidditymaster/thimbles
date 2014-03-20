# This module is for user defined read functions. When read in file is done
# the code will also come to this module and give the user any of the options
# given here
#

__all__ = ['read_hectochelle', "read_mike", "read_elodie"]

# ########################################################################### #
# Import Modules
import numpy as np
import os
import sys
import re
from astropy.io import fits
from ..utils.misc import inv_var_2_var, var_2_inv_var
from .spec_io import MetaData

import thimbles as tmb

# ########################################################################### #

# TODO : add more instructions for how to create a  user read function

pass
# ########################################################################### #
# Example for a user defined read in function
#
# Further explanation of read function...
#

def read_user_defined (filepath):
    """
    Optional doc string
    """
    # First thing always passed is the file path
    if not os.path.isfile(filepath):
        raise IOError("File does not exist")
    
    # then you can do whatever you need to read in the data
    # to wavelength, flux, and inverse variance (inv_var)
    wavelength = []
    flux = []
    inv_var = []
    
    # then return a spectral measurement list and information
    information = MetaData()
    information['filename'] = filepath
    spectral_measurement = tmb.Spectrum(wavelength,flux,inv_var, metadata=information)
    measurement_list = []
    measurement_list.append(spectral_measurement)
    
    return measurement_list


def read_elodie(filepath):
    hdul = fits.open(filepath)
    flux = hdul[0].data
    wvs = hdul[0].header["crval1"] + np.arange(len(flux))*hdul[0].header["cdelt1"]
    rescaling_factor = np.median(flux[flux > 0])
    variance = (hdul[2].data/rescaling_factor)**2
    flux[np.isnan(flux)] = 0.0
    flux /= rescaling_factor
    inv_var = var_2_inv_var(variance)
    info = MetaData()
    info['filename'] = filepath
    spec_list = [tmb.Spectrum(wvs, flux, inv_var)]
    return spec_list

def read_mike(filepath):
    return tmb.io.read_fits(filepath, band=1)

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
    
    information = MetaData()
    information['filename'] = filepath
    spectra = [tmb.Spectrum(wavelength,flux,inv_var, metadata=information)]
    return spectra
