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


# ########################################################################### #


__all__ = ["read","read_txt","read_fits",
           "read","read_txt","read_fits","read_fits_hdu","read_bintablehdu",
           "read_many_fits", "read_apstar", "read_aspcap"]

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

def read_fits (filepath, which_hdu=0, band=0, preferred_wvsoln=None):
    if not os.path.isfile(filepath): 
        raise IOError("File does not exist:'{}'".format(filepath))
        # TODO: add more checks of the fits header_line to see
        # if you can parse more about it. e.g. it's an apogee/hst/or/makee file     
    hdulist = fits.open(filepath)
    if len(hdulist) > 1 and isinstance(hdulist[1],astropy.io.fits.hdu.table.BinTableHDU):
        return read_bintablehdu(hdulist)
    
    kws = dict(which_hdu=which_hdu,
               band=band,
               preferred_wvsoln=preferred_wvsoln)
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
    preferred_wvsoln : None or "{}"
        
    Returns
    -------
    spectra : list of `thimbles.spectrum.Spectrum` objects 
    
    Raises
    ------
    IOError : If it encounters unknown KEYWORD options when looking
        for a wavelength solution
    
    """.format(", ".format(ExtractWavelengthCoefficients.resolve_wvsoln))

def read_fits_hdu (hdulist,which_hdu=0,band=0,preferred_wvsoln=None):
    """
    Reads a fits header unit which contains a wavelength solution
    """
    hdu = hdulist[which_hdu]
    header = hdu.header
    if query_fits_header(header,'APFORMAT').found: 
        warnings.warn(("Received keyword APFORMAT,"
                       " no machinary to deal with this."
                       " [thimbles.io.read_fits_hdu]"))
    # bzero = query_fits_header(header,"BZERO",noval=0)
    # bscale = query_fits_header(header,"BSCALE",noval=1)
    
    ###### read in data ##############################################
    data = hdu.data
    
    # if there's no data return empty
    if data is None:
        warnings.warn("no data for header unit [thimbles.io.read_fits_hdu]")
        wvs, flux, inv_var = np.zeros((3,1))
        return [Spectrum(wvs,flux,inv_var)]
    
    # hdu selection, band select
    if data.ndim == 3:
        data = data[band]
    elif data.ndim == 1:
        data = data.reshape((1,-1))
    # now the data is ndim==2 with the first dimension as the orders
    # and the second being the data points
    
    ##### Calculate the wavelengths for the data
    # set up wavelength and inverse_variance
    wvs = np.ones(data.shape)
    
    # get the wavelength coefficients
    extract_wvcoeffs = ExtractWavelengthCoefficients(hdu.header)
    wvcoeff = extract_wvcoeffs.get_coeffs(preferred_wvsoln)
    
    idx_orders = xrange(len(data))
    
    # go through all the orders
    do_progress = True
    progressive_pt = 1 # this will advance and be used when there is no wavelength solution
    
    for i in idx_orders:
        # get the coefficients and function type    
        equ_type = wvcoeff.get_equation_type()
        if equ_type in ('none',None,'no solution') and do_progress: 
            coeff = [progressive_pt,1]
            equ_type = 'pts'
        else: 
            coeff = wvcoeff.get_coeffs(i)
        # pts[0] = 1 :: this was definitely the right thing to do for SPECTRE's 1-D output but may not be for other equations, may need pts[0]=0,  this may be for bzero,bscale
        pts = np.arange(len(wvs[i]))+1
        # apply function
        wvs[i] = wavelength_solution_functions(pts, coeff, equ_type)    
        progressive_pt += len(pts)

    #=================================================================#
    metadata = MetaData()
    metadata['filepath'] = hdulist.filename
    metadata['hdu_used']=which_hdu
    metadata['band_used']=band
    metadata['wavelength_coeffs'] = wvcoeff
    metadata['header0'] = deepcopy(hdu.header)
        
    spectra = []
    for i in idx_orders:
        metadata['order'] = i
        wl_soln = WavelengthSolution(wvs[i],rv=wvcoeff.rv)
        spectra.append(Spectrum(wl_soln,data[i],metadata=metadata))
         
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

def read (filepath,**kwargs):
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

pass
# ############################################################################# #
# special readin functions

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

# 
# def read_fits_makee (filepath,varience_filepath=None,output_list=False,verbose=False):
# 
#     """ 
#     Knows how to identify the KOA MAKEE file structure which ships with extracted data
#     and apply the eyeSpec function readin to the important directories to obtain a coherent 
#     spectrum object from the files
# 
# 
#     INPUTS:
#     filepath : give an individual filepath for the star or give the top level Star directory from MAKEE. 
#                It will go from TOP_LEVEL/extracted/makee/ and use directories ccd1/ etc to find the appropriate files
# 
#     output_list : if it finds multiple chips of data it will return as a list and not a combined object
# 
# 
#     """
#     non_std_fits=False
#     disp_type='default'
#     preferred_disp='makee'
#     
# 
#     def obj_var_2_inv_var (obj,fill=1e50):
#         var = deepcopy(obj._data)
# 
#         # !! how to treat non values, i.e. negative values
#         zeros = (var<=0)
#         bad = (var>=fill/2.0)
#         infs = (var == np.inf)
# 
#         var[zeros] = 1.0/fill
#         inv_var = 1.0/var
# 
#         # set points which are very large to the fill
#         inv_var[zeros] = fill
#         # set points which are almost zero to zero
#         inv_var[bad] = 0.0
#         inv_var[infs] = 0.0
# 
#         obj._inv_var = deepcopy(inv_var)
#         return inv_var
# 
# 
#     filepath = str(filepath)
#     if not os.path.exists(filepath): raise ValueError("the given path does not exist")
# 
#     objs = {}
#     inv_vars = {}
# 
#     if os.path.isdir(filepath):
# 
#         if filepath[-1:] != '/': filepath += "/"
#         
#         # !! could make it smarter so it would know from anywhere within the TOP_FILE/extracted/makee/ chain
#         
#         full_path = filepath+'extracted/makee/'
#         
#         if not os.path.exists(full_path): raise ValueError("Must have extracted files:"+full_path)
#         
#         ccds = os.listdir(full_path)
#         
#         for ccdir in ccds:
#             if not os.path.isdir(full_path+ccdir): continue
#             if not os.path.exists(full_path+ccdir+'/fits/'): continue
# 
#             if ccdir in objs.keys():
#                 print "Directory was already incorporated:"+ccdir
#                 continue
# 
#             fitspath = full_path+ccdir+'/fits/'
#             fitsfiles = os.listdir(fitspath)
# 
#             print ""
#             print "="*20+format("GETTING DATA FROM DIRECTORY:"+ccdir,'^40')+"="*20
#             for ffname in fitsfiles:
#                 fname = fitspath+ffname
# 
#                 # !! if file.find("_*.fits")
#                 # !! I could add in stuff which would go as another band for the current Flux
# 
#                 if ffname.find("_Flux.fits") != -1: 
#                     print "flux file:"+ccdir+'/fits/'+ffname
#                     objs[ccdir] = read_fits(fname,preferred_disp=preferred_disp,disp_type=disp_type,non_std_fits=non_std_fits,verbose=verbose)
# 
#                 elif ffname.find("_Var.fits") != -1:
#                     print "variance file:"+ccdir+'/fits/'+ffname
#                     tmp_obj = read_fits(fname,preferred_disp=preferred_disp,disp_type=disp_type,non_std_fits=non_std_fits,verbose=verbose)
#                     inv_vars[ccdir] = obj_var_2_inv_var(tmp_obj)
# 
# 
#     else:
#         print "Reading in flux file:"+fname
#         objs['file'] = read_fits(filepath,preferred_disp=preferred_disp,disp_type=disp_type,non_std_fits=non_std_fits,verbose=verbose)
# 
#         if varience_filepath is not None:
#             inv_var = read_fits(varience_filepath,preferred_disp=preferred_disp,disp_type=disp_type,non_std_fits=non_std_fits,verbose=verbose)
#             inv_vars['file'] = obj_var_2_inv_var(inv_var)
# 
# 
#     num_objs = 0
# 
#     OUTPUT_list = []
#     OUT_header = []
#     OUT_wl = []
#     OUT_data = []
#     OUT_inv_var = []
# 
# 
#     for key in objs.keys():
#         obj1 = objs[key]
#         inv_var1 = inv_vars[key]
#         # !! note I'm masking out the inf values
#         mask = (inv_var1 == np.inf)
#         inv_var1[mask] = 0.0
# 
#         if obj1._inv_var.shape != inv_var1.shape:
#             print "HeadsUp: object and inverse variance shape are not the same"
#         else: obj1._inv_var = deepcopy(inv_var1)
#             
#         num_objs += 1
# 
#         if output_list:
#             OUTPUT_list.append(obj1)
#         else:
#             OUT_header.append(obj1.header)
#             if obj1._wl.shape[0] > 1: print "HeadsUp: Multiple bands detected, only using the first"
#             OUT_wl.append(obj1._wl[0])
#             OUT_data.append(obj1._data[0])
#             OUT_inv_var.append(obj1._inv_var[0])
# 
#     if output_list:
#         if num_objs == 1: return OUTPUT_list[0]
#         else: return OUTPUT_list        
#     else:
#         print ""
#         print "="*30+format("COMBINING",'^20')+"="*30
#         wl = np.concatenate(OUT_wl)
#         data = np.concatenate(OUT_data)
#         inv_var = np.concatenate(OUT_inv_var)
# 
#         obj = eyeSpec_spec(wl,data,inv_var,OUT_header[0])
#         obj.hdrlist = OUT_header
#         obj.filepath = fitspath
#         obj.edit.sort_orders()
#         return obj
# 
# def save_spectrum_txt (spectrum,filepath):
#     """
#     Outputs the eyeSpec spectrum class into a given file as text data.
# 
#     INPUTS:
#     =============   ============================================================
#     keyword         (type) Description
#     =============   ============================================================
#     spec_obj        (eyeSpec_spec) spectrum class for eyeSpec
#     filepath        (str) This gives the filename to save the as
#     band            (int,'default') This tells which of the first dimensions of
#                       spec_obj to use. 'default' is spec_obj.get_band()
#     use_cropped     (bool) If True it will crop off points at the begining and 
#                       end of orders which have inverse varience = 0, i.e. have
#                       inf errors
#     order           (int,array,None) you can specify which orders are output
#                       If None then it will output all possible orders
#     clobber         (bool) If True then the function will overwrite files of the
#                       same file name that already exis
# 
#     include_varience (bool) If True the third column which gives the varience 
#                       will be included
#     divide_orders    (bool) If True it will but a commend line with '#' between
#                       each of the orders
#     comment          (str) What symbol to use as a comment
#     divide_header    (bool,None) If False it will give one long string as the first header line
#                                  If True it will divide it up by 80 character lines with a comment of '#:' 
#                                  If None then no header will be printed
#     =============   ============================================================
# 
#     """
#     
#     pass






