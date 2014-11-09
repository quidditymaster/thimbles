#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PURPOSE:
AUTHOR: Dylan Gregersen
DATE: Sun Nov  9 15:12:50 2014
"""
# ########################################################################### #

# import modules

from __future__ import print_function, division, unicode_literals
import os
import sys
import re
from subprocess import Popen,PIPE

from thimbles.options import Option,opts

moog = Option('moog',option_style="parent_dict")

# =========================================================================== #

parfile_templates = {}
parfile_templates['abfind'] = \
"""
abfind
standard_out   '{standard_out}'
summary_out    '{summary_out}'
model_in       '{model_in}''
lines_in       '{lines_in}'
atmosphere    1
molecules     2
lines         1
flux/int      0
damping       1
plot          0
"""

def write_abfind_par (\
    filename=None,clobber=True,
    standard_out="std.out",
    summary_out="sum.out",
    model_in="model_in",
    lines_in="lines_in",
    ):
    """

    """
    pars = locals().copy()
    del pars['clobber']
    del pars['filename']
    return parfile_templates.format(**pars)

def _write_moog_par (driver,filename=None,clobber=True,max_filename_length=80,**moogpars):
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

    _NOTE:_ The default value is given by *value*

    atmosphere          (integer) see WRITEMOOG.ps, possible values are 0, *1*, 2
    molecules           (integer) see WRITEMOOG.ps, possible values are *0*, 1, 2
    trudamp             (integer) see WRITEMOOG.ps, possible values are 0, *1*

    lines               (integer) see WRITEMOOG.ps, possible values are 0, *1*, 2, 3, 4
    flux/int            (integer) see WRITEMOOG.ps, possible values are *0*, 1
    damping             (integer) see WRITEMOOG.ps, possible values are 0, *1*, 2

    units               (integer) see WRITEMOOG.ps, possible values are *0*, 1, 2
    obspectrum          (integer) see WRITEMOOG.ps, possible values are -1, *0*, 1, 3, 5
    iraf                (integer) see WRITEMOOG.ps, possible values are *0*, 1

    freeform            (integer) see WRITEMOOG.ps, possible values are *0*, 1
    strong              (integer) see WRITEMOOG.ps, possible values are *0*, 1
    histogram           (integer) see WRITEMOOG.ps, possible values are *0*, 1
    gfstyle             (integer)  0 for straight gf values
                                  *1* = base-10 logarithms of the gf values
    
    -----------------   ---------------------------------------------------------------------------------------
    
    abundances          (array) This gives the abundances to offset from the input model and 
                           the values to do so by
                           takes an array [[el,offset1,offset2],[el,offset1,offset2],..] 
                                           = e.g. [[26.0,-9,-1,0],[8.0,-9,-1,0],[6.0,-9,-9,-9]]
                           the max number of offsets to give is 1
    
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
    # These are functions used to build the MOOG parameter file    
    
    def format_keyword (keyword):
        return format(keyword,'<10')+" "
      
    def confirm_opts (keyword,pos_vals):
        if keyword not in moogpars: return False
        val = str(moogpars[keyword])
        if val not in pos_vals: raise ValueError("Invalid value for MOOG keyword '"+str(keyword)+"', possible are: "+", ".join(pos_vals))
        return True

    def moog_keyword_values (keyword,opts,default,terminal=False):
        if confirm_opts(keyword,opts): 
            val = moogpars[keyword]
            del moogpars[keyword]
        else: val = opts[default]
        if val is None: return
        
        # !! could avoid including if you don't want
        if terminal: line = format_keyword(keyword)+"'"+str(val)+"'"
        else: line = format_keyword(keyword)+format(val,'<6')
        parlines.append(line)
        pars[keyword] = val
         
    def moog_filename (keyword,which,clobber=True,default=None):
        if keyword in moogpars: fname = str(moogpars[keyword])
        else:
            if default is not None: fname = default
            else: return
    
        if fname is 'None': return
        if len(fname) > max_filename_length: 
            print "HeadsUp: File name is long (i.e. >"+str(max_filename_length)+") '"+fname+"'"
            return
        
        line = format_keyword(keyword)+format("'"+fname+"'","<10")
        if not os.path.exists(fname) and which == 'r': raise ValueError("File does not exist: "+line)
        if os.path.exists(fname) and which == 'w' and not clobber: raise ValueError("File exists: "+line)
        parlines.append(line)
        pars[keyword] = fname
        del moogpars[keyword]

    pass
    #====> The following functions use variables in the local space of this parent function
    # namely pars, parlines, moogpars
    
    def do_plot ():
        keyword = 'plot'
        if keyword not in moogpars: val = '0'
        else:
            val = str(moogpars[keyword])
            del moogpars[keyword]
            if   (driver == 'synth') and (val not in ('1','2')): val = '0' # synth can have values 0,1,2
            elif (driver in ('abfind','blends')) and (val in ('1','2')): val = '0' # abundance fit can have values 0,n
            else: val = '0' # everything else can only have 0, default
        
        pars[keyword] = val
        parlines.append(format_keyword(keyword)+format(val,'<6'))
                
    def do_abundances ():
        keyword = 'abundances'
        format_error = " Abundances not include because not proper format  [[Z_1, offset1_1, offset2_1,...],[Z_2, offset1_2, offset2_2,..],...]  5 is maximum number of offsets"
        if keyword not in moogpars: return
        
        nope = False
        abunds = moogpars['abundances']
        
        # !! could check what type abunds is given in and provide an option to give a dictionary
        #
        
        if len(abunds) == 0: return
        
        try: abunds = np.array(abunds,dtype=float)
        except: nope =True

        try: NZ,NO = abunds.shape
        except: nope=True

        if NO > 6: nope = True
         
        if nope:
            print "HeadsUp:"+format_error
            return

        pars[keyword] = abunds
        parlines.append(format_keyword(keyword)+format(NZ,'<5')+" "+format(NO,'<5'))
        given_z = {}
        for ab in abunds:
            Z = int(ab[0])
            # check to make sure these are atomic transitions
            if not (0<Z<100): 
                print "Given Z value is out of range, "+str(Z)+" not in (0,100)"
                continue
            # check to see if the value was given more than once
            if Z not in given_z: given_z[Z] = ab[0]
            else: 
                print "Given Z value twice, rounded to integer value val1,val2 = "+str((given_z[Z],ab[0]))
                continue
            
            line = " "+format(Z,'>6.1f')
            for val in ab[1:]: line += "  "+format(val,'>7')
            parlines.append(line)
        del moogpars[keyword]
        
    def do_plotpars ():
        keyword = 'plotpars'
        if keyword not in moogpars: return
         
        pars[keyword] = moogpars[keyword]
        
        arr = moogpars[keyword]
        assert len(arr[0])==4,"First row of 'plotpars':[[leftedge,rightedge,loweredge,upperedge],..]"
        assert len(arr[1])==4,"Second row of 'plotpars':[..,[rv,wlshift,vertadd,vertmult],..]"
        assert len(arr[2])==6, "Third row of 'plotpars':[...,[smo_type,fwhm,vsini,limbdark,fwhm_micro,fwhm_lorentzian]]"

        # !! check the smoothing type?
        
        parlines.append(format_keyword(keyword)+format(1,">5"))
        def add_line (ind,val1='',srtind=0):
            line = val1
            for val in arr[ind][srtind:]:
                try: line += " "+format(val,'>5.2f')
                except: raise ValueError("Failed for:"+str(val))
            parlines.append(line)


        add_line(0,'')
        add_line(1,'')
        add_line(2," "+format(arr[2][0],'>5'),1)
        del moogpars[keyword]
         
    def do_synlimits ():
        keyword = 'synlimits'
        if keyword not in moogpars: return
        arr = moogpars[keyword]
        pars[keyword] = arr
        
        try: synstart,synend,wlstep,neighbor_opacity = np.array(arr,dtype=float)
        except: raise ValueError("Invalid synlimits entry")

        line = [format(synstart,'<5.2f'),
                format(synend,'<5.2f'),
                format(wlstep,'<5.2f'),
                format(neighbor_opacity,'<5.2f')]
        
        parlines.append(format_keyword(keyword))
        parlines.append(" ".join(line))
        del moogpars[keyword]          

    def do_isotopes ():
        raise NotImplementedError("Deprecated MOOG2011+")
        keyword = 'isotopes'
        if keyword not in moogpars: return
        
        isotope_matrix = moogpars[keyword]
        
        format_error = "isotopes must be given in moogpars as [[isotope_num, ratio1, ratio2,...],[isotope_num, ratio1, ratio2,...],...]]\n all values must be floating points, the where the number of ratios given equals the number of synthesis done"
        try: isotope_matrix = np.asarray(isotope_matrix,dtype=float)
        except: raise ValueError(format_error)
        
        if isotope_matrix.ndim == 1: isotope_matrix = isotope_matrix.reshape((1,len(isotope_matrix)))
        elif isotope_matrix.ndim > 2:  raise ValueError("Dimension error: "+format_error)
                    
        num_isotopes = isotope_matrix.shape[0]
        num_syntheses = isotope_matrix.shape[1]-1

        if num_syntheses == 0: raise ValueError("No ratios given for isotopes: "+format_error)

        pars[keyword] = isotope_matrix
        # isotopes    #iso   #syn
        parlines.append(format_keyword(keyword)+format(num_isotopes,"<5")+"  "+format(num_syntheses,'<5'))
        
        # write out the pairs 
        for i in xrange(len(isotope_matrix)):
            iso = isotope_matrix[i]
            line = "  "+format(iso[0],'<10.5f')
            for ratio in iso[1:]: line += "  "+format(ratio,'>10')
            parlines.append(line)
        del moogpars[keyword]
            
    def do_fluxlimits ():
        keyword = 'fluxlimits'
        if keyword not in moogpars: return
        
        fluxlimits = moogpars['fluxlimits']

        format_error = "fluxlimits must be given as a floating point array, [start,stop,step]"
        try: fluxlimits = np.asarray(fluxlimits,dtype=float)
        except: 
            print "Fluxlimits not included: "+format_error
            return

        if fluxlimits.ndim != 1: raise ValueError("Dimension error: "+format_error)
        if fluxlimits.shape[0] == 3: raise ValueError("Dimension error: "+format_error)
        
        pars[keyword] = fluxlimits
        line = format_keyword(keyword)
        parlines.append(line)
        
        line = ''
        for val in fluxlimits: line += "  "+format(val,"10")
        parlines.append(line)
        del moogpars[keyword]
      
    def do_blenlimits ():
        keyword = 'blenlimits'
        if keyword not in moogpars: return
        blenlimits = moogpars[keyword]

        format_error = "blenlimits must be given as a floating point array, [delta_wavelength,step,cogatom]"
        try: blenlimits = np.asarray(blenlimits,dtype=float)
        except: 
            print "blenlimits not included: "+format_error
            return

        if blenlimits.ndim != 1: raise ValueError("Dimension error: "+format_error)
        if blenlimits.shape[0] == 3: raise ValueError("Dimension error: "+format_error)
        
        pars[keyword] = blenlimits
        line = format_keyword(keyword)
        parlines.append(line)
        
        line = ''
        for val in blenlimits: line += "  "+format(val,"10")
        parlines.append(line) 
        del moogpars[keyword]

    def do_coglimits ():
        raise NotImplementedError("For curves-of-growth calucate the ews and create your own")
        keyword = 'coglimits'
        if keyword not in moogpars: return
        coglimits = moogpars[keyword]

        format_error = "coglimits must be given as a floating point array, [rwlow, rwhigh, rwstep, wavestep, cogatom]"
        try: coglimits = np.asarray(coglimits,dtype=float)
        except: 
            print "coglimits not included: "+format_error
            return

        if coglimits.ndim != 1: raise ValueError("Dimension error: "+format_error)
        if coglimits.shape[0] == 3: raise ValueError("Dimension error: "+format_error)
        
        pars[keyword] = coglimits
        line = format_keyword(keyword)
        parlines.append(line)
        
        line = ''
        for val in coglimits: line += "  "+format(val,"10")
        parlines.append(line)
        del moogpars[keyword]

    def not_avail (keyword):
        if keyword in moogpars: 
            print "HeadsUp: Keyword not currently supported '"+keyword+"'"
            del moogpars[keyword]

    pass
    #######################################################################################################
    #######################################################################################################
    # these next lines actually build the information
    # parlines are the lines for the parameter file, each line is appended into the list
    # pars is a dictionary with the values used in parlines, similar to moogpars but has some differences 
    # pars is mostly used for debugging
    

    parlines = [driver]
    pars = {'driver':driver}
    moog_keyword_values('terminal',['none','0','7','11','13','x11','xterm','sunview','graphon'],0,terminal=True)
    
    #===> input parameters for files
    moog_filename('standard_out','w',clobber,'STDOUT')
    moog_filename('summary_out','w',clobber)
    moog_filename('smoothed_out','w',clobber)
    moog_filename('iraf_out','w',clobber)
    
    moog_filename('model_in','r',clobber,'FINALMODEL')
    moog_filename('lines_in','r',clobber)
    moog_filename('stronglines_in','r',clobber)
    moog_filename('observed_in','r',clobber)
    
    #====> input flag parameters
    moog_keyword_values('atmosphere',['0','1','2'],1)
    moog_keyword_values('molecules',['0','1','2'],0)
    moog_keyword_values('trudamp',['0','1'],1)
    moog_keyword_values('lines',['0','1','2','3','4'],1)
    moog_keyword_values('flux/int',['0','1'],0)
    moog_keyword_values('damping',['0','1','2'],1)
    moog_keyword_values('units',['0','1','2'],0)
    
    moog_keyword_values('obspectrum',['-1','0','1','3','5'],1)
    moog_keyword_values('iraf',['0','1'],0)
    do_plot()
    moog_keyword_values('freeform',['0','1'],0)
    moog_keyword_values('strong',['0','1'],0)
    moog_keyword_values('histogram',['0','1'],0)
    moog_keyword_values("gfstyle",['0','1'],1)
    
    
    do_abundances()
    do_plotpars()
    do_synlimits()
    do_isotopes()
    do_fluxlimits()
    do_coglimits()
    do_blenlimits()
        
    not_avail('lumratio')
    not_avail('delaradvel')
    not_avail('scat')
    not_avail('opacit')
    
    for key in moogpars: 
        print("Keyword unknown to MOOG pars :"+str(key))
    
    if filename is None: 
        return parlines
    else:
        FILE = open(filename, "w")
        FILE.writelines("\n".join(parlines)+"\n")
        FILE.close()                  

class TestMoogExec (object):

    def setUp (self):
        pass 

    def test_synth_driver (self):
        pass 

moog_silent_exec_path = Option("path",parent="moog",envvar="MOOGSILENT")

class MoogEngine(RadiativeTransferEngine):
    """an abstract class specifying the API for wrapping radiative transfer
    codes.
    """
    
    def __init__(self, working_dir, photosphere_engine=None):
        self.photosphere_engine = photosphere_engine
        RadiativeTransferEngine.__init__(self,working_dir)

    def exec_moog_silent (self):
        """ Execute moog silent specified in opts['moog.path'] """
        x = opts['moog.path']
        p = Popen(x,stdout=PIPE,stdin=PIPE)
        p.wait()

    def write_moog_par (self,*args,**kws):
        return write_moog_par(*args,**kws)

    def _not_implemented(self, msg=None):
        if msg is None:
            msg = "Not implemented for this engine type"
        raise NotImplementedError(msg)
    
    def ew_to_abundance(self, linelist, stellar_params):
        """generate abundances on the basis of equivalent widths"""
        self._not_implemented()
    
    def line_abundance(self, linelist, stellar_params, inject_as=None):
        self._not_implemented()
    
    def abundance_to_ew(self, linelist, stellar_params, abundances=None):
        self._not_implemented()
    
    def spectrum(self, linelist, stellar_params, normalized=True):
        self._not_implemented
    
    def continuum(self, stellar_params):
        self._not_implemented()
    
    def line_strength(self, linelist, stellar_params):
        self._not_implemented()


