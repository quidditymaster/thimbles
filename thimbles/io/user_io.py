# This module is for user defined read functions. When read in file is done
# the code will also come to this module and give the user any of the options
# given here
#

__all__ = ['read_hectochelle', "read_mike", "read_elodie", "read_aspcap", "read_apstar"]

# ########################################################################### #
# Import Modules
import numpy as np
import os
import sys
import re
from astropy.io import fits
from ..utils.misc import inv_var_2_var, var_2_inv_var
from ..utils.misc import clean_inverse_variances
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
    inv_var = clean_inverse_variances(inv_var)
    info = MetaData()
    info['filename'] = filepath
    info["header"] = hdul[0].header
    spec_list = [tmb.Spectrum(wvs, flux, inv_var, metadata=info)]
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

def read_apstar (filepath, data_hdu=1, error_hdu=2, row=0):
    """ 
  reads the apStar APOGEE fits files.
    
    Paremeters
     ----------
     filepath : string path to fits file
         APOGEE pipeline reduced data with a 0 header unit similar to the below
     use_row : integer 
         APOGEE refers to these as rows, default is row1 ("combined spectrum with individual pixel weighting")
     get_telluric : boolean
         If True then it will also extract the telluric data
 
 
    Returns
     -------
    a list with a single apogee spectrum in it
     
 
     =================================================================
     Example header 0 header unit:
     
     HISTORY APSTAR: The data are in separate extensions:                      
     HISTORY APSTAR:  HDU0 = Header only                                       
     HISTORY APSTAR:  All image extensions have:                               
     HISTORY APSTAR:    row 1: combined spectrum with individual pixel weighti 
     HISTORY APSTAR:    row 2: combined spectrum with global weighting         
     HISTORY APSTAR:    row 3-nvisis+2: individual resampled visit spectra     
     HISTORY APSTAR:   unless nvists=1, which only have a single row           
     HISTORY APSTAR:  All spectra shifted to rest (vacuum) wavelength scale    
     HISTORY APSTAR:  HDU1 - Flux (10^-17 ergs/s/cm^2/Ang)                     
     HISTORY APSTAR:  HDU2 - Error (10^-17 ergs/s/cm^2/Ang)                    
     HISTORY APSTAR:  HDU3 - Flag mask (bitwise OR combined)                   
     HISTORY APSTAR:  HDU4 - Sky (10^-17 ergs/s/cm^2/Ang)                      
     HISTORY APSTAR:  HDU5 - Sky Error (10^-17 ergs/s/cm^2/Ang)                
     HISTORY APSTAR:  HDU6 - Telluric                                          
     HISTORY APSTAR:  HDU7 - Telluric Error                                    
     HISTORY APSTAR:  HDU8 - LSF coefficients                                 
     HISTORY APSTAR:  HDU9 - RV and CCF structure
 
     """
    hdulist = fits.open(filepath)
    metadata = MetaData()
    metadata['filepath'] = hdulist.filename()
    hdr = hdulist[0].header
    metadata['header'] = hdr
    
    if len(hdulist[1].data.shape) == 2:
        flux = hdulist[data_hdu].data[row].copy()
        sigma = hdulist[error_hdu].data[row].copy()
    elif len(hdulist[1].data.shape) == 1:
        flux = hdulist[data_hdu].data
        sigma = hdulist[error_hdu].data
    crval1 = hdr["CRVAL1"]
    cdelt1 = hdr["CDELT1"]
    nwave  = hdr["NWAVE"]
    wv = np.power(10.0, np.arange(nwave)*cdelt1+crval1)
    return [Spectrum(wv, flux, var_2_inv_var(sigma**2))]

def read_aspcap(filepath):
    hdulist = fits.open(filepath)
    metadata = MetaData()
    metadata['filepath'] = hdulist.filename()
    hdr = hdulist[0].header
    metadata['header'] = hdr
    
    flux = hdulist[1].data
    sigma = hdulist[2].data
    invvar = var_2_inv_var(sigma**2)*(flux > 0)
    crval1 = hdulist[1].header["CRVAL1"]
    cdelt1 = hdulist[1].header["CDELT1"]
    nwave  = len(flux)
    wv = np.power(10.0, np.arange(nwave)*cdelt1+crval1)
    return [Spectrum(wv, flux, invvar)]
