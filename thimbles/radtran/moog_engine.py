#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import sys
import re
from warnings import warn
import subprocess
import numpy as np

import thimbles as tmb
from thimbles.radtran.engines import RadiativeTransferEngine
from thimbles.radtran.marcs_engine import MarcsInterpolator
from thimbles.options import Option,opts

Option('moog',option_style="parent_dict")
ex_opt = Option("executable",parent="moog",envvar="MOOGSILENT", default=None)
if not ex_opt.runtime_str is None:
    ex_opt.evaluate()
Option("opac_rad", parent="moog", default=3.0)
Option("delta_wv", parent="moog", default=0.01)
Option(
    "working_dir", 
    parent="moog",
    envvar="MOOGWORKINGDIR", 
    default=os.getcwd(),
)
#TODO: make a default strong lines file to crop to wavelengths and use always.

# =========================================================================== #
default_par_template=\
"""damping        1
freeform       0
gfstyle        1
atmosphere     1
molecules      2"""

common_par_components=\
"""
flux/int       {flux_int}
lines_in     {lines_in}
standard_out {outfile}.std
summary_out  {outfile}.sum
model_in     {model_in}
"""

Option("par_template", parent="moog", default=default_par_template)

abfind_template="abfind\n"+opts["moog.par_template"] + common_par_components
ewfind_template="ewfind\n"+opts["moog.par_template"] + common_par_components
synth_template= "synth\n" +opts["moog.par_template"] + common_par_components +\
"""synlimits
          {min_wv: 10.5f} {max_wv: 10.5f} {delta_wv: 10.5f} {opac_rad: 10.5f}  
"""


class MoogEngine(RadiativeTransferEngine):
    """an abstract class specifying the API for wrapping radiative transfer
    codes.
    """
    
    def __init__(self, working_dir=None, photosphere_engine=None):
        if working_dir is None:
            working_dir = opts["moog.working_dir"]
        if photosphere_engine == None:
            photosphere_engine = MarcsInterpolator()
        RadiativeTransferEngine.__init__(self,working_dir, photosphere_engine)
    
    def _exec_moog(self):
        """execute moog silent in this engines working directory"""
        cur_dir = os.getcwd()
        #change to the working directory and execute
        if opts.moog.executable.runtime_str is None:
            raise Exception("path to MOOG executable not found, set moog.executable in config file or set an environment variable called MOOGSILENT with the path.")
        os.chdir(self.working_dir)
        try:
            x = opts['moog.executable']
            p = subprocess.call(x)
        except Exception as e:
            print(e)
        finally:
            os.chdir(cur_dir)
    
    def ew_to_abundance(
            self, 
            linelist, 
            stellar_params, 
            central_intensity=False
    ):
        """generate abundances on the basis of equivalent widths
        by performing a fit to predicted ew's instead of inverting
        the line by line abundances.

        for a line by line individual abundance determination use
        the line_abundance method.
        """
        self._not_implemented()
    
    def line_abundance(
            self, 
            linelist, 
            stellar_params, 
            inject_as=None, 
            central_intensity=False
    ):
        """line by line abundances for the given linelist (with an ew column)
        and stellar_params.
        
        parameters
        
        linelist: LineList
          the line data and ew measurements
        stellar_params: StellarParameters or String
          if a StellarParameters object use the photosphere engine to create a 
          model photosphere, if a string use it as a file path to a pre generated
          photosphere file (the file will be copied to the working_dir)
        """
        #write out the model atmosphere
        self._make_photo_file(stellar_params)
        #write out the linelist in moog format
        line_name = "templines.ln.tmp"
        line_file = os.path.join(self.working_dir, line_name)
        tmb.io.moog_io.write_moog_linelist(line_file, linelist)
        out_fname = "result.tmp"
        flux_int = 0
        if central_intensity:
            flux_int = 1 
        f = open(os.path.join(self.working_dir, "batch.par"), "w")
        f.write(abfind_template.format(model_in=self._photosphere_fname,
                               lines_in=line_name,
                               outfile=out_fname,
                               flux_int=flux_int
        ))
        f.flush()
        f.close()
        self._exec_moog()
        summary_fname = os.path.join(self.working_dir, out_fname + ".sum")
        result = tmb.io.moog_io.read_moog_abfind_summary(summary_fname)
        return result
    
    def abundance_to_ew(
            self, 
            linelist, 
            stellar_params, 
            abundances=None, 
            central_intensity=False
    ):
        self._make_photo_file(stellar_params)
        #write out the linelist in moog format
        line_name = "templines.ln.tmp"
        line_file = os.path.join(self.working_dir, line_name)
        tmb.io.moog_io.write_moog_linelist(line_file, linelist)
        out_fname = "result.tmp"
        flux_int = 0
        if central_intensity:
            flux_int = 1 
        f = open(os.path.join(self.working_dir, "batch.par"), "w")
        f.write(ewfind_template.format(model_in=self._photosphere_fname,
                               lines_in=line_name,
                               outfile=out_fname,
                               flux_int=flux_int,
        ))
        f.flush()
        f.close()
        self._exec_moog()
        summary_fname = os.path.join(self.working_dir, out_fname + ".sum")
        result = tmb.io.moog_io.read_moog_ewfind_summary(summary_fname)
        return result
    
    def spectrum(
            self, 
            linelist, 
            stellar_params, 
            wavelengths, 
            sampling_mode="rebin",
            normalized=True, 
            delta_wv=None,
            opac_rad=None,
            central_intensity=False,
            abundances=None, #dictionary ion-->log_eps
            differential_abundances=True,
    ):
        if not normalized:
            self._not_implemented("moog only generates normalized spectra")
        self._make_photo_file(stellar_params)
        #write out the linelist in moog format
        line_name = "templines.ln.tmp"
        line_file = os.path.join(self.working_dir, line_name)
        if linelist == "reuse":
            if not os.path.isfile(line_file):
                raise ValueError("reuse linelist option only available if linelist pre-exists in moog working directory")
            tmb.io.moog_io.write_moog_linelist(line_file, linelist)
        out_fname = "result.tmp"
        flux_int = 0
        if central_intensity:
            flux_int = 1 
        f = open(os.path.join(self.working_dir, "batch.par"), "w")
        if opac_rad is None:
            opac_rad = opts["moog.opac_rad"]
        if delta_wv is None:
            delta_wv = opts["moog.delta_wv"]
        f.write(synth_template.format(
            model_in=self._photosphere_fname,
            lines_in=line_name,
            outfile=out_fname,
            min_wv=wavelengths[0],
            max_wv=wavelengths[-1],
            delta_wv=delta_wv,
            opac_rad=opac_rad,
            flux_int=flux_int
        ))
        
        if not abundances is None:
            f.write("abundances     {}    1\n".format(len(abundances)))
            z_map = {}
            for ion in abundances:
                if isinstance(ion, int):
                    ion_id = ion
                else:
                    ion_id = ion.z
                if ion_id >= 100:
                    continue
                if ion_id == 1:
                    continue
                cur_ab = z_map.get(ion_id) 
                if not cur_ab is None:
                    #average over multiples
                    z_map[ion_id] = 0.5*(cur_ab + abundances[ion])
                else:
                    z_map[ion_id] = abundances[ion]
            
            if not differential_abundances:
                raise NotImplementedError("abundances for this function need to be specified relative to iron")
            for ion_z in z_map:
                delta_ab = abundances[ion]
                nl = "          {}   {}\n".format(ion_z, delta_ab)
                f.write(nl)
        f.flush()
        f.close()
        self._exec_moog()
        summary_fname = os.path.join(self.working_dir, out_fname + ".sum")
        result_spectra = tmb.io.moog_io.read_moog_synth_summary(summary_fname)
        resampled_spectra = [result.sample(wavelengths, mode=sampling_mode) for result in result_spectra]
        return resampled_spectra
    
    def continuum(self, stellar_params):
        self._not_implemented()


mooger = MoogEngine()


def get_model_name (teff,logg,feh,vt,modtype=None):
    """ Based on the input atmosphere parameters it returns a formatted string representation

    Parameters 
    teff : float 
        Stellar effective temperature
    logg : float 
        Gravitational acceleration at the surface of the star
    feh  : float 
        Normalized solar metallicity [Fe/H]
    vt   : float 
        Stellar microturbulence
    modtype : str
        The type of model used, if None then no model type will be added

    Returns 
    model_name : str


    Example
    >>> model_representation = get_model_name(5000, 4.1, -2.13, 1.1, 'ODFNEW')
    >>> print model_representation
    "5000p410m213p110.ODFNEW"

    """

    feh_sign = feh/abs(feh)
    if feh_sign < 0: 
        sign = "m"
    else: 
        sign = 'p'
    steff = str(int(teff))
    slogg = format(logg,'03.2f').replace('.','')
    sfeh = sign+format(abs(feh),'03.2f').replace('.','')
    svt = format(vt,'03.2f').replace('.','')
    
    out = steff+"p"+slogg+sfeh+'v'+svt
    if modtype is not None: 
        out += "."+str(modtype)
    return out

# =========================================================================== #

# all stuff for moog parameters file

def _moog_par_format_fluxlimits (fluxlimits):
    """
    fluxlimits = [start, stop, step]

    ```
    fluxlimits 
      5555.0 5600.0 10.0
    ```
    """
    if fluxlimits is None:
        return ""
    lines = [\
        "fluxlimits",
        (" {:10}"*3).format(*list(map(float,fluxlimits)))
        ]
    return "\n".join(lines)

def _moog_par_format_plot (driver,plot):
    """
    moogpars['plot']

    ```
    plot    1
    ```

    """
    if plot is None:
        return ""
    val = str(plot)
    if driver == 'synth' and val not in ('1','2'):
        val = 0 # synth can have values 0,1,2
    elif (driver in ('abfind','blends')) and val in ('1','2'):
        val = 0 # abundance fit can have values 0,n
    else:
        val = 0 # everything else can only have 0, default
    return "plot      {:<6}".format(val)

def _moog_par_format_plotpars (plotpars):
    """

    moogpars['plotpars']

    plotpars = [[leftedge,rightedge,loweredge,upperedge],
                [rv,wlshift,vertadd,vertmult],
                [smo_type,fwhm,vsini,limbdark,fwhm_micro,fwhm_lorentzian]]            

    """
    if plotpars is None:
        return ""
    assert len(plotpars[0])==4,"First row of 'plotpars':[[leftedge,rightedge,loweredge,upperedge],..]"
    assert len(plotpars[1])==4,"Second row of 'plotpars':[..,[rv,wlshift,vertadd,vertmult],..]"
    assert len(plotpars[2])==6, "Third row of 'plotpars':[...,[smo_type,fwhm,vsini,limbdark,fwhm_micro,fwhm_lorentzian]]"

    lines = ['plotpars       1']
    lines.append((" {:>.2f}"*4).format(*plotpars[0]))
    lines.append((" {:>.2f}"*4).format(*plotpars[1]))
    lines.append((" {:>5}"+" {:>.2f}"*5).format(*plotpars[2]))
    return "\n".join(lines)

def _moog_par_format_synlimits (synlimits):
    """ 
    moogpars['synlimits']

    synlimits = [syn_start,syn_end,wl_step,opacity_width]

    """
    if synlimits is None:
        return ""
    lines = ["synlimits "]
    # synstart,synend,wlstep,opacity_width
    lines.append((" "+" {:<.2f}"*4).format(*list(map(float,synlimits))))
    return "\n".join(lines)

def _moog_par_format_abundances (abundances):
    """ Abundances for MOOG parameter file 

    abundances = [[el,offset1,offset2],[el,offset1,offset2],..] 
    abundances = [[26.0,-9, -1.2, 0],
                  [8.0, -9,   -1, 0],
                  [6.0, -9,   -9, -9]]

    """
    if abundances is None:
        return ""
    errmsg = (\
        "Abundances not include because not proper format "
        "[[Z_1, offset1_1, offset2_1,...],[Z_2, offset1_2, offset2_2,..],...] "
        "5 is maximum number of offsets")
    if not len(abundances):
        return ""
    abunds = np.asarray(abundances,dtype=float)
    nz,no = abunds.shape 
    if no > 6 or no < 2:
        raise ValueError(errmsg)

    lines = ["abundances {:>5} {:>5}".format(nz-1,no-1)]
    fmt = " {:>6.1f}"+" {:>7}"*(no-1)
    for row in abunds:
        z = row[0] # proton number
        if not (0 < z < 100):
            raise ValueError(errmsg)
        lines.append(fmt.format(*row))
    return "\n".join(lines)

moog_max_pathlength = Option("moog_max_pathlength",parent="moog")

def _check_moog_files (fp,mode='r',clobber=True,max_filename_length=None):
    """ Takes a moog keyword and extracts from the moogpars """ 
    # - - - - - - - - - - - - filepath 
    if fp is None:
        return 
    # - - - - - - - - - - - - check file mode
    if mode not in ('r','w'):
        raise ValueError("mode must be 'r' or 'w'")
    # - - - - - - - - - - - - check the maximum filelength
    if max_filename_length is None:
        max_filename_length = opts['moog.moog_max_pathlength']
    if len(fp) > max_filename_length:
        warn("Filepath '{}' is too long for MOOG (max {}) omitting".format(fp,max_filename_length))
        return 
    # - - - - - - - - - - - - check file
    exists = os.path.isfile(fp)
    if not exists and mode == 'r':
        raise IOError("File does not exist '{}'".format(fp))
    elif exists and mode == 'w' and not clobber:
        raise IOError("File exist, not clobbering '{}'".format(fp))
    return fp 

def write_moog_par (driver,filename='batch.par',clobber=True,max_filename_length=80,**moogpars):
    """
    Writes a MOOG parameter file based on WRITEMOOG.ps

    Parameters
    ----------
    driver : string
        The subroutine driver for MOOG to use  
        Possible Drivers: synplot, synth, cogsyn, blends, abfind, ewfind, cog, calmod,  
            doflux, weedout, gridsyn, gridplo, binary, abpop, synpop  
            
    fname : string or None 
        If string then it will write to that filename. If None then it will return the lines it would have written
    clobber: boolean
        If 'True' then the code will continue and overwrite existing files
    
    **moogpars keywords : dictionary or keywords. See the list below
    
    -----------------   ---------------------------------------------------------------------------------------
    
    _NOTE:_ The code will check the files to see if they exist and whether to overwrite (based on clobber)
        files which do
          
    standard_out        (string) The file name for writing the standard output
    summary_out         (string) The file name for writing the summary output
    smoothed_out        (string) The file name for writing the smoothed output
    iraf_out            (string) The file name for writing the IRAF output

    -----------------   ---------------------------------------------------------------------------------------

    model_in            (string) The file name for the model
    lines_in            (string) The file name for the input line list
    stronglines_in      (string) The file name for the input strong line list
    observed_in         (string) The file name for the observed input data

    -----------------   ---------------------------------------------------------------------------------------

    _NOTE:_ The default value is given by [value]

    atmosphere          (integer) see WRITEMOOG.ps, possible values are 0, [1], 2
    molecules           (integer) see WRITEMOOG.ps, possible values are [0], 1, 2
    trudamp             (integer) see WRITEMOOG.ps, possible values are 0, [1]

    lines               (integer) see WRITEMOOG.ps, possible values are 0, [1], 2, 3, 4
    flux/int            (integer) see WRITEMOOG.ps, possible values are [0], 1
    damping             (integer) see WRITEMOOG.ps, possible values are 0, [1], 2

    units               (integer) see WRITEMOOG.ps, possible values are [0], 1, 2
    obspectrum          (integer) see WRITEMOOG.ps, possible values are -1, [0], 1, 3, 5
    iraf                (integer) see WRITEMOOG.ps, possible values are [0], 1

    freeform            (integer) see WRITEMOOG.ps, possible values are [0], 1
    strong              (integer) see WRITEMOOG.ps, possible values are [0], 1
    histogram           (integer) see WRITEMOOG.ps, possible values are [0], 1
    gfstyle             (integer)  0 for straight gf values
                                  [1] = base-10 logarithms of the gf values
    
    -----------------   ---------------------------------------------------------------------------------------
    
    abundances          (array) This gives the abundances to offset from the input model and 
                           the values to do so by
                           takes an array [[el,offset1,offset2],[el,offset1,offset2],..] 
                                           = e.g. 
                           the max number of offsets to give is 1
                           Example:
                           >>> abundances = [[26.0,-9, -1.2, 0],
                                             [8.0, -9,   -1, 0],
                                             [6.0, -9,   -9, -9]]
                           >>> write_moog_par('abind',abundances=abundances)

    -----------------   ---------------------------------------------------------------------------------------                                             
    
    plotpars             (list) The plotting parameters for the data and syntheses
                          [[leftedge, rightedge, loweredge, upperedge],
                           [rv, wlshift, vertadd, vertmult],
                           [smo_type, fwhm, vsini, limbdark, fwhm_micro, fwhm_lorentzian]]
    
    -----------------   ---------------------------------------------------------------------------------------                                                      
    
    synlimits           (array) Parameters for the synthesis
                          equals [wavelength_start, wavelenght_end, step_size, opacity_radius]
    
    -----------------   ---------------------------------------------------------------------------------------                                             
    
    isotopes            Not Available : not supported in most recent MOOG. Use synth instead
    
    -----------------   ---------------------------------------------------------------------------------------                                                
    
    fluxlimits           (array) gives the wavelength parameters for flux curves
                            equals [start, stop, step] as floating points
    
    -----------------   ---------------------------------------------------------------------------------------                                                
    
    coglimits            Not Available : buggy implementation in most recent MOOG

    -----------------   ---------------------------------------------------------------------------------------                                                
    
    blenlimits           (array) gives the parameters for blended line abundance matches
                            equals [delwave, step, cogatom]
    
    -----------------   ---------------------------------------------------------------------------------------                                                
    
    lumratio             not_available
    
    -----------------   ---------------------------------------------------------------------------------------                                                
    
    delaradvel           not_available
    
    -----------------   ---------------------------------------------------------------------------------------                                                
    
    scat                 not_available
    
    -----------------   ---------------------------------------------------------------------------------------                                                
    
    opacit               not_available
    
    -----------------   ---------------------------------------------------------------------------------------                                                


    Returns
    -------
    lines_out : list (optional)
        Return a list of the lines it would write to the file if filename is None
       
    Notes
    -----
    __1)__ See WRITEMOOG.ps for more information about all the keyword options



    Example
    -------
    >>>
    >>>
    >>>
    
    TODO: add an example or 2
    
    """
    #######################################################################################################

    # This value is given in the MOOG file Atmos.f
    # the limit is given here to prevent later undo problems 

    # these are the possible drivers, check
    moogdrivers = ['synplot','synth','cogsyn','blends','abfind','ewfind','cog','calmod',
                   'doflux','weedout','gridsyn','gridplo','binary','abpop','synpop']
    if driver not in moogdrivers: 
        raise ValueError("MOOG driver must be in: "+", ".join(moogdrivers))

    # these shortcuts convert moogpars inputs to those which will be understood
    shortcuts = {'stdout':'standard_out',"sumout":"summary_out"}
    moogpars = moogpars.copy()
    for key in moogpars:
        moogpars[shortcuts.get(key,key)] = moogpars[key]
        
    pass
    #######################################################################################################
    #######################################################################################################
    # these next lines actually build the information
    # parlines are the lines for the parameter file, each line is appended into the list
    # pars is a dictionary with the values used in parlines, similar to moogpars but has some differences 
    # pars is mostly used for debugging

    options = {} # default,[all_options],default
    options['terminal'] = (None,['none','0','7','11','13','x11','xterm','sunview','graphon'],None)

    options['standard_out'] = "{}.stdout".format(driver)
    options['summary_out']  = None #"{}.sumout".format(driver)
    options['smoothed_out'] = None #"{}.smout".format(driver)
    options['iraf_out']     = None #"{}.irafout".format(dirver)

    options['atmosphere'] = (1,['0','1','2'])
    options['molecules']  = (0,['0','1','2'])
    options['lines']      = (1,['0','1','2','3','4'])
    options['flux/int']   = (0,['0','1'])
    options['damping']    = (1,['0','1','2'])
    options['units']      = (0,['0','1','2'])
    options['obspectrum'] = (0,['-1','0','1','2','3','5'])
    options['iraf']       = (0,['0','1'])
    options['freeform']   = (0,['0','1'])
    options['strong']     = (0,['0','1'])
    options['histogram']  = (0,['0','1'])
    options['gfstyle']    = (1,['0','1'])


    # --------------------------------------------------------------------------- #
    # create parlines, all the lines for a parameter file

    parlines = [driver]

    d,defaults = options['terminal']
    term = moogpars.pop('terminal',d)
    if term is not None:
        if not (term in defaults):
            raise ValueError("terminal {} not in {}".format(term,defaults))
        parlines.append("terminal   {}".format(term))

    # ----------------------- files
    add = (\
        ("standard_out",'w'),
        ("summary_out",'w'),
        ("smooth_out",'w'),
        ("iraf_out",'w'),
        ("model_in",'r'),
        ("lines_in",'r'),
        ("stronglines_in",'r'),
        ("observed_in",'r'),
        )
    for k,mode in add:
        fp = moogpars.pop(k,options[k])
        fp = _check_moog_files(fp,mode=mode,clobber=clobber)
        if fp is None:
            continue
        parlines.append("{:<10} '{}'".format(k,fp))

    # ----------------------- single value parameters
    add = (\
        'atmosphere',
        'molecules',
        'trudamp',
        'lines',
        'flux_int',
        'damping',
        'units',
        'opspectrum',
        'iraf',
        'freeform',
        'strong',
        'histogram',
        'gfstyle',
        )
    for k in add:
        d,defaults = options[k]
        val = format(moogpars.pop(k,d))
        if val not in defaults:
            raise ValueError("moog parameter {} must be in {}".format(val,defaults))
        parlines.append("{:<10} {}".format(k,val))

    # ----------------------- parameter files
    parlines.append(_moog_par_format_plot(moogpars.pop('plot')))
    parlines.append(_moog_par_format_plotpars(moogpars.pop('plotpars')))
    parlines.append(_moog_par_format_abundances(moogpars.pop('abundances')))
    parlines.append(_moog_par_format_synlimits(moogpars.pop('synlimits')))
    parlines.append(_moog_par_format_fluxlimits(moogpars.pop('fluxlimits')))

    # ----------------------- 
    if len(moogpars):
        raise KeyError("These keywords are not supported {}".format(list(moogpars.keys())))

    # ----------------------- 
    parlines.append("\n")
    if filename is None: 
        return parlines
    else:
        with open(filename,'w') as fn:
            fn.write("\n".join(parlines))





