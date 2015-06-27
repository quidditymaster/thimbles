# Purpose: For reading in data from fits and text files
# Authors: Dylan Gregersen, Tim Anderton
# Date: 2013/08/13 23:52:42
#

# ########################################################################### #
# Standard Library
import re, os
from copy import copy, deepcopy
from collections import Iterable
import warnings
import json

# 3rd Party
import astropy.io
from astropy.io import fits
import numpy as np
import h5py
import sqlalchemy as sa

# Internal
import thimbles as tmb
from thimbles.thimblesdb import find_or_create
from thimbles.tasks import task
from thimbles.utils.misc import var_2_inv_var
from thimbles.spectrum import Spectrum
from . import wavelength_extractors


__all__ = ["read_spec","read_ascii","read_fits",
           "read_fits","read_fits_hdu","read_bintablehdu",
           "read_many_fits", "read_hdf5",
           "read_json", "read_spec_list",
           "write_ascii", "write_ascii", "write_hdf5",
           "write_spec",
]

def read_hdf5(filepath):
    hf = h5py.File(filepath, "r")
    spectra = []
    spec_keys = list(hf.keys())
    for spec_idx in range(len(spec_keys)):
        spec_grp_name = "spec_{}".format(spec_idx)
        wv_soln_path = spec_grp_name + "/wv_soln" 
        wvs = np.array(hf[wv_soln_path + "/wv_centers"])
        lsf = np.array(hf[wv_soln_path + "/lsf"])
        rv, vhelio = hf[wv_soln_path + "/velocity_offsets"]
        wvs = WavelengthSolution(wvs, rv=rv, delta_helio=vhelio, lsf=lsf)
        
        quant_path = spec_grp_name + "/spectral_quantities" 
        flux = np.array(hf[quant_path+"/flux/values"])
        ivar = np.array(hf[quant_path+"/flux/ivar"])
        
        new_spec = Spectrum(wvs, flux, ivar)
        spectra.append(new_spec)
    return spectra

def write_hdf5(filepath, spectra):
    """writes a list of lists of spectra to a hdf5 file
    """
    hf = h5py.File(filepath, "w")
    for spec_idx in range(len(spectra)):
        cspec = spectra[spec_idx]
        spec_grp_name = "spec_{}".format(spec_idx)
        wv_soln_path = spec_grp_name + "/wv_soln" 
        hf[wv_soln_path + "/wv_centers"] = cspec.wv
        hf[wv_soln_path + "/lsf"] = cspec.wv_soln.lsf
        hf[wv_soln_path + "/velocity_offsets"] = [cspec.rv, cspec.vhelio]
        
        quant_path = spec_grp_name + "/spectral_quantities" 
        hf[quant_path+"/flux/values"] = cspec.flux
        hf[quant_path+"/flux/ivar"] = cspec.inv_var
    hf.close()


_spectrum_separator = "#new_spectrum"

def read_ascii(
        fname, 
        wv_col=0, 
        flux_col=1, 
        ivar_col=None, 
        lsf_col=None, 
        lsf_degree=2,
        ivar_as_var=False, 
        spectrum_separator=None, 
        comment="#",
):
    """
    Readin ascii tables as a spectrum
    
    Parameters
    ----------
    fname : string
      path to the text file
    wv_col: int
      index of wavelength column
    flux_col: int
      index of flux colum
    ivar_col: int
      index of inverse variance column
    lsf_col: int
      index to interpret as line spread function
    ivar_as_var: bool
      if True interpret the ivar_col as variance instead of inverse variance.
    spectrum_separator: string
      lines matching this token will cause subsequent lines to be
      loaded into a new spectrum.
    Returns
    -------
    spectra : list of `thimbles.spectrum.Spectrum` objects     
    """ 
    #### check if file exists   ####### #############
    if not os.path.isfile(fname): 
        raise IOError("File does not exist:'{}'".format(fname))
    
    if spectrum_separator is None:
        spectrum_separator = _spectrum_separator
    
    spectra = []
    wvs = []
    flux = []
    ivar = []
    lsf = []
    comment_pat = re.compile(comment)
    spec_sep_pat = re.compile(spectrum_separator)
    for line in open(fname):
        if spec_sep_pat.match(line):
            if len(ivar)==0:
                ivar = None
            if len(lsf) == 0:
                lsf = None
            spectra.append(Spectrum(wvs, flux, ivar=ivar, lsf=lsf))
            wvs = []
            flux = []
            ivar = []
            lsf = []
        if comment_pat.match(line):
            continue
        else:
            lspl = line.split()
            if len(lspl) == 0:
                continue
            wvs.append(float(lspl[wv_col]))
            flux.append(float(lspl[flux_col]))
            if not ivar_col is None:
                ivar.append(float(lspl[ivar_col]))
            if not lsf_col is None:
                lsf.append(float(lspl[lsf_col]))
    if len(wvs) > 0:
        if len(ivar) == 0:
            ivar = None
        if len(lsf) == 0:
            lsf=None
        spec = Spectrum(wvs, flux, ivar=ivar, lsf=lsf)
        spectra.append(spec)
    return spectra


def write_ascii(
        spectra, 
        fpath, 
        fmt="%.8e", 
        ivar_as_var=False,
        spectrum_separator=None, 
):
    if isinstance(spectra, tmb.Spectrum):
        spectra = [spectra]
    if spectrum_separator is None:
        spectrum_separator = _spectrum_separator
    if spectrum_separator[-1] != "\n":
        spectrum_separator = spectrum_separator + "\n"
    f = open(fpath, "w")
    fmt_str = "{wv: 11.4f} {flux: 15.5f} {ivar: 15.5f} {lsf: 15.5f}\n"
    for spec_idx, spec in enumerate(spectra):
        wvs, flux, ivar, lsf = spec.wvs, spec.flux, spec.ivar, spec.lsf
        if ivar_as_var:
            ivar = inv_var_2_var(ivar)
        for idx in range(len(wvs)):
            line = fmt_str.format(wv=wvs[idx], flux=flux[idx], ivar=ivar[idx], lsf=lsf[idx])
            f.write(line)
        if spec_idx != (len(spectra)-1):
            f.write(spectrum_separator)
    return True


def read_json(fname, database=None, auto_add=True):
    file = open(fname)
    jsonstr = file.read()
    file.close()
    if database is None:
        database = tmb.ThimblesDB("")
    in_dict = json.loads(jsonstr)
    defaults = in_dict.get("defaults")
    if defaults is None:
        defaults = {}
    defaults.setdefault("read_func", "read_spec")
    spectra_dicts = in_dict["spectra"]
    spectra = []
    for spec_dict in spectra_dicts:
        sdict = copy(defaults)
        sdict.update(spec_dict)
        fname = sdict.pop("fname")
        parent_dir = sdict.get("parent_dir", "")
        fpath = os.path.join(parent_dir, fname)
        read_func_str = sdict.pop("read_func")
        read_func = eval(read_func_str, tmb.io.spec_io.__dict__)
        sub_spectra = read_func(fpath)
        if "source.name" in sdict:
            source_name = sdict.pop("source.name")
        else:
            source_name = None
        if "source_type" in sdict:
            source_type = sdict.pop("source_type")
        else:
            source_type = "Source"
        source_class = eval(source_type, tmb.wds.__dict__)
        source = find_or_create(
            instance_class=source_class,
            name=source_name,
            auto_add=auto_add
        )
        if "source.ra" in sdict:
            source.ra = sdict.pop("source.ra")
        if "source.dec" in sdict:
            source.dec = sdict.pop("source.dec")
        spec_flag_space = tmb.flags.spectrum_flag_space
        flag_dict = {}
        for flag_name in spec_flag_space.flag_names:
            flag_str = "flags.{}".format(flag_name)
            if flag_str in sdict:
                flag_val = bool(sdict.pop(flag_str))
                flag_dict[flag_name] = flag_val
        for spec in sub_spectra:
            spec.source = source
            spec.flags.update(**flag_dict)
            spec.info.update(sdict)
            spectra.append(spec)
    return spectra


def read_fits (filepath, which_hdu=0, band=0, preference=None):
    if not os.path.isfile(filepath): 
        raise IOError("File does not exist:'{}'".format(filepath))
        # TODO: add more checks of the fits header_line to see
        # if you can parse more about it. e.g. it's an apogee/hst/or/makee file     
    hdulist = fits.open(filepath)
    if len(hdulist) > 1 and isinstance(hdulist[1],astropy.io.fits.hdu.table.BinTableHDU):
        return read_bintablehdu(hdulist)
    
    return read_fits_hdu(hdulist, which_hdu=which_hdu, band=band, preference=preference)

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
    preference : string, or None
        
    Returns
    -------
    spectra : list of `thimbles.spectrum.Spectrum` objects 
    
    Raises
    ------
    IOError : If it encounters unknown KEYWORD options when looking
        for a wavelength solution
    
    """.format(", ".join(list(wavelength_extractors.from_functions.keys())))

def read_fits_hdu (hdulist,which_hdu=0,band=0,preference=None):
    """
    Reads a fits header unit which contains a wavelength solution
    """
    hdu = hdulist[which_hdu]
    header = hdu.header
    if 'APFORMAT' in header:
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
    spectra = []
    for i in range(len(wvs)):
        spectra.append(Spectrum(wvs[i],data[i]))
    
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
    #metadata = MetaData()
    #metadata['filepath'] = hdulist.filename()
    #metadata['header'] = hdulist[0].header
    #for i,hdu in enumerate(hdulist[1:]):
    #    metadata['header{}'.format(i+1)] = hdu.header
    
    guesses = dict(wvcol_guess = ['wavelength','wvs','wavelengths','wave','waves'],
                   fluxcol_guess = ['flux','ergs','intensity'],
                   var_guess = ['variance','varience','var'],
                   inv_var_guess = ['inv_var','inverse variance','inverse varience','ivar','invvar'],
                   sigma_guess = ['sigma','error','noise'],
                   )   
    # include all uppercase and capitalized guesses too
    items = list(guesses.items())     
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
        for i in range(len(wvs)):
            if inv_var is not None:
                ivar = inv_var[i]
            else:
                ivar = None
            #metadata['order'] = i
            spectra.append(Spectrum(wvs[i],flux[i],ivar))#,metadata=metadata.copy()))
    elif wvs.ndim == 1:
        spectra.append(Spectrum(wvs,flux,inv_var))#,metadata=metadata))
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

    if isinstance(filelist,(str)):
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
        spectra += read_spec(fname)
    
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

def probably_file_list(filepath):
    if isinstance(filepath,str):
        # check if this is a file with a list of files
        with open(filepath,'r') as f:
            for line in f:
                line = line.split("#")[0].strip()
                if len(line) == 0:
                    continue
                if not os.path.isfile(line):
                    return False
        return True    
    elif isinstance(filepath,Iterable):
        return True
    else:
        return False

def read_spec_list(fname, file_type="detect"):
    lines = open(fname).readlines()
    spectra = []
    for line in lines:
        cur_fname = line.split("#")[0].strip()
        if len(cur_fname) == 0:
            continue
        cur_specl = read_spec(line.rstrip(), file_type=file_type)
        spectra.extend(cur_specl)
    return spectra

def probably_fits_file(fname):
    """check to see if the file might be a fits file
    """
    # NOTE: Could also check for 'END' card and/or 'NAXIS  ='
    fits_rexp = re.compile("[SIMPLE  =,XTENSION=]"+"."*71+"BITPIX  =")
    with open(fname,'r') as f:
        file_header = f.read(300)
    s = fits_rexp.search(file_header)
    if s is None:
        return False
    else:
        return True

def probably_ascii_columns(fname):
    raise NotImplementedError()
    with open(fname, "r") as f:
        file_head = f.read(1000)
    lines = file_head.split("\n")
    lines = [line.split("#")[0] for line in lines]
    len_lines = [len(line.split()) for line in lines if len(line.split()) > 0]
    if len(len_lines) > 2:
        len_lines = np.array(len_lines[:-1])
        if np.all(len_lines == len_lines[0]):
            return True
    return False

def probably_hdf5(fname):
    return fname[-3:] == ".h5"

def detect_spectrum_file_type(fname):
    if probably_fits_file(fname):
        return "fits"
    elif probably_hdf5(fname):
        return "hdf5"
    elif probably_file_list(fname):
        return "file_list"
    elif probably_ascii_columns:
        return "ascii"
    return None

pass
# ############################################################################# #
@task(
    result_name="spectra",
    sub_kwargs=dict(
        fname=dict(editor_style="file"),
        file_type={},
))
def read_spec(fname, file_type="detect", extra_kwargs=None):
    """
    a swiss army knife read in function for spectra, attempts to detect 
    and read in a large variety of formats.
    
    file type: string
      "detect"    attempt to determine the file type automatically
      "fits"      read a fits type file
      "hdf5"      read a hdf5 format spectrum file
      "ascii"     read a file with ascii columns for wavelength and flux
      "file_list" file is actually a list of files to read in.
    extra_kwargs: dictionary
      a dictionary to unpack into the readin function as keyword arguments
      that gets called on the basis of the file_type.
    """
    if extra_kwargs is None:
        extra_kwargs = {}
    if not isinstance(file_type, str):
        try:
            return file_type(fname, **extra_kwargs)
        except IOError as e:
            raise e
    
    if file_type == "detect":
        file_type = detect_spectrum_file_type(fname)
        if file_type is None:
            raise Exception("unable to detect spectrum file type")
    
    if file_type == "fits":
        res =  read_fits(fname, **extra_kwargs)
    elif file_type == "hdf5":
        res = read_hdf5(fname, **extra_kwargs)
    elif file_type == "file_list":
        res = read_spec_list(fname, **extra_kwargs)
    elif file_type == "ascii":
        res = read_ascii(fname, **extra_kwargs)
    return res


@task(
    result_name="write_success",
)
def write_spec(spectra, fname, file_type="ascii"):
    if file_type == "ascii":
        write_ascii(spectra, fname)
    elif file_type == "hdf5":
        write_hdf5(spectra, fname)
    else:
        raise NotImplementedError("spectrum output of file_type: {} has not been implemented".format(file_type))
