# Purpose: For reading in data from fits and text files
# Authors: Dylan Gregersen, Tim Anderton
# Date: 2013/08/13 23:52:42
#

# ########################################################################### #
# Standard Library
import re
import os
import time
from copy import deepcopy
from collections import Iterable
import warnings

# 3rd Party
import scipy
import astropy
import astropy.io
from astropy.io import fits
import numpy as np

# Internal
from ..utils.misc import var_2_inv_var
from ..spectrum import Spectrum, WavelengthSolution
from ..metadata import MetaData
import wavelength_extractors


# ########################################################################### #


__all__ = ["read_spec","read_txt","read_fits",
           "read_txt","read_fits","read_fits_hdu","read_bintablehdu",
           "read_many_fits"]

# ########################################################################### #

# read functions

def read_txt (filepath,**np_kwargs):
    """
    Readin text files with wavelength and data columns (optionally inverse varience)
    
    Parameters
    ----------
    filepath : string
        Gives the path to the text file
    np_kwargs : dictionary
        Contains keywords and values to pass to np.loadtxt
        This includes things such as skiprows, usecols, etc.
        unpack and dtype are set to True and float respectively 
    
    Returns
    -------
    spectrum : list of `thimbles.spectrum.Spectrum` objects 
        If get_data is False returns a Spectrum object
    
    Notes
    -----
    __1)__ Keywords txt_data, unpack, and dtype are forced for the
        np_kwargs.
        
    """ 
    #### check if file exists   ####### #############
    if not os.path.isfile(filepath): 
        raise IOError("File does not exist:'{}'".format(filepath))

    metadata = MetaData()
    metadata['filepath'] = os.path.abspath(filepath)

    # Allows for not repeating a loadtxt
    np_kwargs['unpack'] = True
    np_kwargs['dtype'] = float
    txt_data = np.loadtxt(filepath,**np_kwargs)
    
    # check the input txt_data
    if txt_data.ndim == 1:
        warnings.warn("NO WAVELENGTH DATA FOUND, USING FIRST COLUMN AS DATA")
        data = txt_data 
        wvs = np.arange(len(data))+1
        var = None
    elif txt_data.ndim == 2:
        wvs = txt_data[0]
        data = txt_data[1]
        var = None
    elif txt_data.ndim == 3: 
        wvs,data,var = txt_data
    elif txt_data.shape[0] > 2: 
        warnings.warn(("Found more than 3 columns in text file '{}' "
                       "taking the first three to be wavelength, data,"
                       " variance respectively".format(filepath)))
        wvs,data,var = txt_data[:3]
        
    if var is not None:        
        inv_var = var_2_inv_var(var)
    else:
        inv_var = None
    
    return [Spectrum(wvs,data,inv_var,metadata=metadata)]
    
############################################################################
# readin is the main function for input

def read_fits (filepath, which_hdu=0, band=0, preference=None):
    if not os.path.isfile(filepath): 
        raise IOError("File does not exist:'{}'".format(filepath))
        # TODO: add more checks of the fits header_line to see
        # if you can parse more about it. e.g. it's an apogee/hst/or/makee file     
    hdulist = fits.open(filepath)
    if len(hdulist) > 1 and isinstance(hdulist[1],astropy.io.fits.hdu.table.BinTableHDU):
        return read_bintablehdu(hdulist)
    
    kws = dict(which_hdu=which_hdu,
               band=band,
               preference=preference)
    return read_fits_hdu(hdulist,**kws)

read_fits.__doc__ = """
    Takes a astropy.io.fits hdulist and then for a particular hdu and band
    extracts the wavelength and flux information
    
    This goes through keywords in the header and looks for specific known
    keywords which give coefficients for a wavelenth solution. It then 
    calculates the wavelengths based on that wavelength solution.
    
    Parameters
    ----------
    hdulist : `astropy.io.fits.HDUList` or string
        A header unit list which contains all the header units
    which_hdu : integer
        Which hdu from the hdulist to use, start counting at 0
    band : integer
        If the hdu has NAXIS3 != 0 then this will select which
        value that dimension should be
    preference : {}, or None
        
    Returns
    -------
    spectra : list of `thimbles.spectrum.Spectrum` objects 
    
    Raises
    ------
    IOError : If it encounters unknown KEYWORD options when looking
        for a wavelength solution
    
    """.format(", ".join(wavelength_extractors.from_functions.keys()))

def read_fits_hdu (hdulist,which_hdu=0,band=0,preference=None):
    """
    Reads a fits header unit which contains a wavelength solution
    """
    hdu = hdulist[which_hdu]
    header = hdu.header
    if header.has_key("APFORMAT"):
        warnings.warn(("Received keyword APFORMAT,"
                       " no machinary to deal with this."
                       " [thimbles.io.read_fits_hdu]"))
    # bzero = query_fits_header(header,"BZERO",noval=0)
    # bscale = query_fits_header(header,"BSCALE",noval=1)
    
    ###### read in data ##############################################
    data = hdu.data
    
    # if there's no data return empty
    if data is None:
        raise IOError("No data found for fits file '{}'".format(hdulist.filename))
    
    # hdu selection, band select
    if data.ndim == 3:
        data = data[band]
    elif data.ndim == 1:
        data = data.reshape((1,-1))
    # now the data is ndim==2 with the first dimension as the orders
    # and the second being the data points
    
    ##### Calculate the wavelengths for the data
    wvs = wavelength_extractors.from_header(hdu.header, preference=preference)    
    if len(wvs) != len(data):
        raise IOError(("number of wavelength solutions n={} "
                       "from header incompatable with number of data orders, n={}").format(len(wvs),len(data)))
    
    #=================================================================#
    metadata = MetaData()
    metadata['filepath'] = hdulist.filename
    metadata['hdu_used']=which_hdu
    metadata['band_used']=band
    metadata['header0'] = deepcopy(hdu.header)
        
    spectra = []
    for i in xrange(len(wvs)):
        metadata['order'] = i
        spectra.append(Spectrum(wvs[i],data[i],metadata=metadata))
         
    return spectra

def read_bintablehdu (hdulist,which_hdu=1,wvcolname=None,fluxcolname=None,varcolname=None):
    """
    Read in a fits binary fits table
    
    Parameters 
    ----------
    bintablehdu : `astropy.io.fits.BinTableHDU`
    wvcolname : None or string 
    fluxcolname : None or string
    varcolname : None or string
    
    
    """
    metadata = MetaData()
    metadata['filepath'] = hdulist.filename()
    for i,hdu in enumerate(hdulist):
        metadata['header{}'.format(i)] = hdu.header
    
    guesses = dict(wvcol_guess = ['wavelength','wvs','wavelengths'],
                   fluxcol_guess = ['flux','ergs','intensity'],
                   var_guess = ['variance','varience'],
                   inv_var_guess = ['inv_var','inverse variance','inverse varience'],
                   sigma_guess = ['sigma','error'],
                   )   
    # include all uppercase and capitalized guesses too
    items = guesses.items()     
    for key,values in items:
        guesses[key] += [v.upper() for v in values]
        guesses[key] += [v.capitalize() for v in values]
    
    def get_key (set_key,options,guess_keys):
        if set_key is not None:
            return set_key
        else:
            for key in guess_keys:
                if key in options:
                    return key
    
    which_hdu = abs(which_hdu)
    
    if not (len(hdulist) > which_hdu and isinstance(hdulist[which_hdu],astropy.io.fits.hdu.table.BinTableHDU)):
        raise ValueError("This must be done with a bin table fits file")
        
    # read in data   
    data = hdulist[which_hdu].data
    options = data.dtype.names
    
    # get the keys
    wvs_key = get_key(wvcolname,options,guesses['wvcol_guess'])
    flux_key = get_key(fluxcolname,options,guesses['fluxcol_guess'])
    
    sig_key = get_key(None,options,guesses['sigma_guess'])
    var_key = get_key(varcolname,options,guesses['var_guess'])
    inv_var_key = get_key(None,options,guesses['inv_var_guess'])
    
    # check that these keys are essential
    if wvs_key is None or flux_key is None:
        raise ValueError("No keys which make sense for wavelengths or flux")
    wvs = data[wvs_key]
    flux = data[flux_key]
    
    # check for error keys
    if var_key is not None:
        inv_var = var_2_inv_var(data[var_key])
    elif sig_key is not None:
        var = data[sig_key]**2
        inv_var = var_2_inv_var(var)
    elif inv_var_key is not None:
        inv_var = data[inv_var_key]
    else:
        inv_var = None
    
    # store the spectra 
    spectra = []
    if wvs.ndim == 2:
        for i in xrange(len(wvs)):
            if inv_var is not None:
                ivar = inv_var[i]
            else:
                ivar = None
            metadata['order'] = i
            spectra.append(Spectrum(wvs[i],flux[i],ivar,metadata=metadata.copy()))
    elif wvs.ndim == 1:
        spectra.append(Spectrum(wvs,flux,inv_var,metadata=metadata))
    else:
        raise ValueError("Don't know how to deal with data ndim={}".format(wvs.ndim))
    return spectra

def read_many_fits (filelist,relative_paths=False,are_orders=False):
    """
    takes a list of spectre 1D files and returns `timbles.Spectrum` objects 

    Parameters
    ----------
    filelist : string or list of strings
        Each string gives the file path to a 
    relative_paths : boolean
        If True then each file path will be treated relative to the filelist file 
        directory.
    are_orders : boolean
        If True it will order files by wavelength and assign them order numbers

    Return
    ------
    spectra : list of `thimbles.Spectrum` objects    
    
    """
    list_of_files = []
    relative_paths = bool(relative_paths)
    nofile = "File does not exist '{}'"

    if isinstance(filelist,(basestring)):
        dirpath = os.path.dirname(filelist)
        
        if not os.path.isfile(filelist):
            raise IOError(nofile.format(filelist))
        
        with open(filelist) as f:
            files = f.readlines()
            
        for fname in files:
            fname = fname.split()[0]
            if relative_paths:
                fname = os.path.join(dirpath,fname)  
            if not os.path.isfile(fname):
                warnings.warn(nofile.format(filelist))                      
            else: 
                list_of_files.append(fname)
        f.close()
    #-----------------------------------------------#
    # if given input is not a string assume it's a list/array
    elif isinstance(filelist,Iterable):
        list_of_files = [str(fname).split()[0] for fname in filelist]
    else:
        raise TypeError("filelist must be string or list of strings")
    
    #============================================================#
    
    spectra = []
    for fname in list_of_files:
        spectra += read(fname)
        
    if are_orders:
        med_wvs = []
        for spec in spectra:
            med_wvs.append(np.median(spec.wv))
        sort_spectra = []
        for i in np.argsort(med_wvs):
            spec = spectra[i]
            spec.metadata['order'] = i
            sort_spectra.append(spec)
        spectra = sort_spectra     
    return spectra

pass
# ############################################################################# #
# This is the general read in function for fits and text files

# general read function

def is_many_fits (filepath):
    
    if isinstance(filepath,basestring):
        # check if this is a file with a list of files
        if not os.path.isfile(filepath):
            return False
        with open(filepath,'r') as f:
            for line in f:
                if len(line.strip()) and line.strip()[0] != "#" and not os.path.isfile(line.rstrip().split()[0]):
                    return False
        return True    
    elif isinstance(filepath,Iterable):
        return True
    else:
        return False

def read_spec(filepath,**kwargs):
    """
    General read 
    
    **kwargs are passed either to read_txt or read_fits depending on determined 
    file type
    
    """
    # NOTE: Could also check for 'END' card and/or 'NAXIS  ='
    fits_rexp = re.compile("[SIMPLE  =,XTENSION=]"+"."*71+"BITPIX  =")
    
    # check the first line of the file. What type is it?
    with open(filepath,'r') as f:
        header_line = f.readline()
    s = fits_rexp.search(header_line)
    
    
    if s is None: # is not a fits file
        if is_many_fits(filepath): # is a file of many files
            return read_many_fits(filepath,**kwargs)
        else: # is a single text file
            return read_txt(filepath,**kwargs)
    else: # is a fits file
        return read_fits(filepath,**kwargs)










