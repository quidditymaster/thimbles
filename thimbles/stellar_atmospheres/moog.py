
# Modules
import pdb #@UnusedImport    
import os, time, subprocess, threading
import numpy as np
from ..stellar_atmospheres import executables 
from ..utils.prompt_user import get_bounds, get_filename 
from .moog_utils import read_moog_linelist, write_moog_par, write_moog_lines_in, simple_llist_data, get_model_name

pass
#=============================================================================#
# checking to see if the exectuable exists

def check_moog (error=False):
    _avail_error = "Must have MOOG executable declared to use this function not: "+executables['moog']
    check =  os.path.isfile(executables['moog'])
    if not check and error: raise ValueError(_avail_error)
    return check
        
def check_moog07 (error=False):
    _avail_error = "Must have MOOG 2007-3 executable declared to use this function not: "+executables['moog07']
    check =  os.path.isfile(executables['moog07'])
    if not check and error: raise ValueError(_avail_error)
    return check

def check_moogsilent (error=False):
    _avail_error = "Must have MOOGSILET executable declared to use this function not: "+executables['moogsilent']
    check =  os.path.isfile(executables['moogsilent'])
    if not check and error: raise ValueError(_avail_error)
    return check

pass
#=============================================================================#
# basic MOOG calls

def synth (lines_in, model_in, wlrange='default',synlimits_step_size=0.02, synlimits_opacity_radius=1.0, clean_up = True, verbose=True, **moogpars):
    """
    This creates a synthesis model of a spectrum using MOOG     
    
    Parameters
    ----------
    lines_in : string or array 
        Gives the file name for a line list to be read in from the file must be MOOG readable (see WRITEMOOG.ps)
        if array then give the data to create a line list from (wl, spe, ep, loggf) as the columns
    model_in : string 
        Gives the file name for a MOOG readable stellar model
    wlrange : array-like 
        Gives (lower_bound,upper_bound) anything else and it will default to the lines_in wavelength range
    clean_up : boolean
        If 'True' it will delete temporary files 
    verbose : boolean 
        will give printed progress outputs
    **moogpars : dictionary or keywords  
        Keywords to be included in the batch.par file with will be given to MOOG. 
        See the function write_moog_par
    
    
    Returns
    -------   
    moog_output : dictionary
        This dictionary gives the output names of the files it just created
        "summary_out" -> gives the summary output file which was created
        "standard_out" -> gives the standard output file which was created
        
    Notes
    -----
    __1)__ This creates temporary files "MOOG_INPUT_LINELIST" and "batch.par"    
    __2)__ As a default it will create files:
        STDOUT_{wlmin}_{wlmax}.std
        SUMOUT_{wlmin}_{wlmax}.sum
    
    Examples
    --------
    >>> files_out = moog_synth("linelist_file.ln","FINALMODEL")
    >>> summary_out_file = files_out['summary_out']
    
    >>> linelist = np.loadtxt("linelist_file.ln",usecols=[0,1,2,3])
    >>> files_out = moog_synth(linelist,"FINALMODEL",wlrange=(4300,5100))
    
    Add abundances
    >>> abundances = [[26.0, -9, 1, 0, 1],[8.0, -9, 1, 0, 1]]
    >>> files_out = moog_synth(linelist,"FINALMODEL",wlrange=(4300,5100),abundances=abundances)
    
    Add abundances, and change the output names
    >>> moogpars = {}
    >>> moogpars['abundances'] = abundances
    >>> moogpars['summary_out'] = "new_summary_out.txt"
    >>> moogpars['standard_out'] = "new_standard_out.txt"
    >>> files_out = moog_synth(linelist,"FINALMODEL", wlrange=(4300,5100), **moogpars)
    
    """
    # !! check CHPC version for more notes on this function
    
    
    ######################## USER SETUP #############################################
    
    if   check_moogsilent(False): MOOGEXE = executables['moogsilent']
    elif check_moog(False): MOOGEXE = executables['moog']
    else: raise ValueError("Neither MOOGSILENT or MOOG is available to use") 

    ######################## USER SETUP #############################################
    overshoot = 0
    t1 = time.time()
    def check_filename (fname):
        if not os.path.exists(fname): raise ValueError("File does not exists '"+fname+"'")
        return fname
    
    pass
    #================================================================================#
    # create the linelist for moog
    moogin = write_moog_lines_in("MOOG_INPUT_LINELIST")
    if type(lines_in) in (str,np.string_,np.str_,np.str):
        llist = simple_llist_data(lines_in)
        lines = llist.get_cropped_lines(wlrange)
        syn_start,syn_end = llist.get_wlbounds()
        moogin.writelines(lines)
        
    else:
        llist = np.asarray(lines_in)
        default = False
        try: syn_start, syn_end = wlrange
        except: default = True
        
        if not default:
            wls = llist.T[0]
            mask = (syn_start < wls )*(wls < syn_end)
            llist = llist[mask]

        for i in xrange(len(llist)): moogin.add_line(*llist[i])
        
    moogin.close()
    
    pass
    #================================================================================#
    # create parameter file for the synthesis
    
    def check_moogpars (keyword,default):
        if keyword not in moogpars: moogpars[keyword]=default
        
    moogpars['terminal'] = 'none'
    moogpars['lines_in'] = 'MOOG_INPUT_LINELIST'
    moogpars['model_in'] = check_filename(model_in)
    if 'observed_in' in moogpars: check_filename(moogpars['observed_in'])
    
    wl_range_string = str(int(syn_start))+"_"+str(int(syn_end))
    results = {}
    
    check_moogpars('summary_out', "SUMOUT_"+wl_range_string+".sum")
    check_moogpars('standard_out', "STDOUT_"+wl_range_string+".std")
         
    results['summary_out']  = moogpars['summary_out'] 
    results['standard_out'] = moogpars['standard_out']
    
        
    if 'stronglines_in' in moogpars:
        check_filename(moogpars['stronglines_in'])
        if 'strong' not in moogpars: moogpars['strong'] = 1
    elif 'strong' in moogpars and 'stronglines_in' not in moogpars:
        print "HeadsUp: 'stronglines_in' was not given in moogpars, changing 'strong' to 0"
        moogpars['strong'] = 0

    check_moogpars('atmosphere',1)
    check_moogpars('molecules',2)
    
    # add in overshoot to avoid bad end behavior
    syn_start -= overshoot
    syn_end += overshoot
    
    moogpars['synlimits'] = [syn_start,
                             syn_end,
                             float(synlimits_step_size),
                             float(synlimits_opacity_radius)]
        
    # create batch.par, the input MOOG parameter file
    lines = write_moog_par('synth',filename='batch.par',clobber=True,**moogpars)
    if verbose: print "  Synthesizing range: "+str(tuple((round(syn_start,2),round(syn_end,2))))

    #================================================================================#
    # use MOOG to create synthesis
    _MOOG = subprocess.call([MOOGEXE], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    #downstream = MOOG.communicate(input="batch.par\n")
    if verbose:
        print "-"*60
        print "Time to create synthesis: "+str(round((time.time()-t1)/60,2))+" minutes"
        

    def check_clean (fname):
        if os.path.exists(fname): os.system("rm "+fname)
        
    if clean_up: 
        check_clean("batch.par")
        check_clean("MOOG_INPUT_LINELIST")

    return results

def ewfind (lines_in, model_in, fname_out=None, lines_formatted=True, lines_defaults={}, verbose=False, wlbounds=None, ewbounds=None, clean_up=True, logfile=None):
    """
PURPOSE:
   Run MOOG ewfind on a linelist and get equivalent width estimates
   
CATEGORY:
   MOOG

INPUT ARGUMENTS:
    lines_in : (string or array) the filename to be read using function read_moog_linelist, the lines_formatted and lines_defaults are used in this function.
                if array then needs to have columns [[wl,spe,ep,loggf],...]    can optionally include columns for vwdamp, d0, and ew
    model_in : (string) give the name of the model file

INPUT KEYWORD ARGUMENTS:
    fname_out : (string or None) If filename string is given then it will output the linelist to that file
    lines_formatted : (bool) If True it will assume MOOG formatting (see function read_moog_linelist)
    lines_defaults : (dictionary) Gives the default values for vwdamp, d0 and ew (see function read_moog_linelist)
    vebose : (bool) give verbose output
    wlbounds : (arraylike or None) if not None wavelength bounds = (wlmin, wlmax) else it will adopt the min and max wavelength from linelist
    ewbounds: (arrayline or None) if not None equivalent width bounds = (wlmin, wlmax) else it will just output all 
    clean_up : (bool) If True it will delete temporary files -> 'std', 'sum', 'MOOGLINES.txt', 'batch.par'
    logfile : (string or None) to create log of output give a filename 
   

OUTPUTS:
   (array) output linelist, the linelist with an added column of ewquivelant width cropped by wavelength and equivalent width

DEPENDENCIES:
   External Modules Required
   =================================================
    numpy, os
   
   External Functions and Classes Required
   =================================================
    moog07_exe, read_moog_linelist, _CoreMOOGewfind,
    join_threads 
    
NOTES:
   (1) MOOG will sometimes fail to produce an EW for a line and will just hang
       This program runs a parallel thread and will kill MOOG if it takes to long
       on a particular line. Moving onto the next line.

EXAMPLE:
    >>> lines_in= 'moog_linelist.ln'
    >>> model_in = 'FINALMODEL'
    >>> linelist = moog_ewfind(lines_in, model_in)
   
    >>> lines_in = np.loadtxt('moog_linelist.ln',usecols=[0,1,2,3])
    >>> linelist = moog_ewfind(lines_in, model_in, fname_out='linelist_ew.ln', wlbounds=(4100,5300), logfile='LOGFILE.txt')
    

MODIFICATION HISTORY:
    13, Jun 2013: Dylan Gregersen
    """
    check_moog07(True)
    # check type of input
    
    # get the lines from the line list
    # if you receive a file then try using read_moog_linelist on it
    if type(lines_in) in (str,np.string_,np.str_,np.str): 
        linelist_filename = lines_in       
        linelist = read_moog_linelist(lines_in, formatted=lines_formatted, defaults=lines_defaults, convert_gf=False)
        
        wls = linelist['wl']
        llist = []
        cols = ('wl','spe','ep','loggf','vwdamp','d0','ew')
        for col in cols: llist.append(linelist[col])
        
        linelist = np.dstack(llist)[0]
    
    # assume it's an array  
    else:
        linelist = np.asarray(lines_in,dtype=float)
        if linelist.ndim != 2: raise ValueError("Linelist must be two dimensional columns of parameters by rows of lines")
        
        # try to correct for a possible error in shape
        shape = linelist.shape
        if shape[0] < shape[1] and shape[1] > 7: llist = linelist.T
        
        wls = linelist[:,0]
        linelist_filename = 'unknown'

    # get the wlbounds
    if wlbounds is not None: xmin,xmax = np.min(wlbounds),np.max(wlbounds)
    else: xmin, xmax = np.min(wls), np.max(wls)

    # crop the input linelist
    mask = (xmin < linelist[:,0])*(linelist[:,0] < xmax)
    linelist = linelist[mask]

    # run the routine for the file
    t = _CoreMOOGewfind(linelist, model_in, moog_parfile='batch.par',timeout=2, verbose= verbose,logfile=logfile, linelist_filename=linelist_filename)
    t.go()

    crashed = False
    try: join_threads(t.threads)
    except KeyboardInterrupt:
        print "\nKeyboardInterrupt catched."
        #print "Terminate main thread."
        #print "If only daemonic threads are left, terminate whole program."
        crashed = True

    if t.logfile is not None:
        print >> t.logfile,"- "*30
        print >> t.logfile,"OUTPUT FILE = "+fname_out
            
    t.stop()  
    
    if clean_up: t.clean_up()
    if crashed: return 0
    
    if fname_out is not None: return t.create_linelist(fname_out,clobber=True, ewbounds=ewbounds)
    else: return t.get_output_data(ewbounds=ewbounds)

pass
#=============================================================================#
# ewfind tools

class _CoreMOOGewfind(object):
    nan_ew = -1
    
    moog_linein = 'MOOGLINELIST.txt'

    moog07_exe = executables['moog07']

    def __init__(self,linelist,model_in='FINALMODEL',moog_parfile='batch.par',timeout=5,verbose=True,logfile=None,linelist_filename='unknown'):
        check_moog07(True)
        #=================================================================================#
        # check inputs
        def check_file (fname):
            if not os.path.exists(fname): raise ValueError("File does not exist:'"+fname+"'")
            return fname

        self.linelist = np.asarray(linelist)

        self.moog_parfile = moog_parfile
        self.moog_model   = check_file(model_in)

        f = open(self.moog_model)
        l1 = f.readline().rstrip().strip().split()
        l2 = f.readline().rstrip().strip().split()
        f.close()
        self.model_info = "'FINALMODEL' - "+" ".join(l1+l2)



        if logfile is not None:
            self.logfile = open(logfile,'w')
            print >> self.logfile, "INPUT LINELIST =  "+linelist_filename
            print >> self.logfile, "MODEL FILE =  "+self.moog_model
            print >> self.logfile, "MODEL INFO =  "+self.model_info
            print >> self.logfile, "- "*30                       
        else: self.logfile = None


        self.timeout = float(timeout)
        self.verbose = bool(verbose)

        #=================================================================================#
        # create outputs
        self.output_data = []

        #=================================================================================#
        # internal parameters
        self.threads = []
        self.running = True        
        self.termination = False
        self.quit = False

    def __len__ (self):
        return len(self.output_data)

    def clean_up (self):
        os.system('rm std sum '+self.moog_parfile+' '+self.moog_linein)

    def timeout_moog (self):
        checktime = False
        # runs while the code is running
        while (self.running and not checktime):
            checktime =  (time.time() - self.start_time > self.timeout)

            #print "checking ",self.running,checktime
            #time.sleep(0.5)

        self.termination = True
        # print "=====",('MOOG' not in dir(self)),self.MOOG.poll()
        if 'MOOG' not in dir(self): return
        try: self.MOOG.terminate() # kill moog
        except: return

    def run_moog (self):
        self.running = True
        N = len(self.linelist)
        # go through the line list and one by one create the lines
        for i in xrange(N):
            if self.quit: break

            # create a moog linelist file
            write_moog_lines_in(self.moog_linein,oneline=self.linelist[i])

            # create the batch.par
            moogpars = {'lines_in':self.moog_linein,
                       'model_in':self.moog_model}
            
            write_moog_par('ewfind',filename=self.moog_parfile,**moogpars)
            
            # make a call to MOOG
            perdone = str(round( (((float(i)+1.0) /float(N))*100.0), 1))
            lo =  perdone+"% WORKING ON LINE "+str(i)+" : "+str(tuple(self.linelist[i]))

            if self.verbose: print lo
            if self.logfile is not None: print >> self.logfile, lo

            #pdb.set_trace()
            self.start_time = time.time()
            t0 = self.create_thread(self.timeout_moog)
            ew = self.call_moog()
            self.stop_thread(t0)


            if ew == self.nan_ew: lo = "***MOOG CRASHED ON LINE***"
            else: lo =  "          =======> EW="+str(ew)

            if self.verbose: print lo
            if self.logfile is not None: print >> self.logfile, lo

            arr = list(self.linelist[i])+[ew]
            if len(self.output_data) == 0: self.output_data = [arr]
            else: self.output_data.append(arr)

        self.running = False

    def call_moog (self):
        self.termination = False
        # run moog for one line
        self.MOOG = subprocess.Popen(self.moog07_exe, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        inp = self.moog_parfile+"\nQ\n"
        
        # if it hangs it will be on this communicate
        datastream = self.MOOG.communicate(input=inp)
        
        if self.termination: return self.nan_ew
        
        # record the data
        index = str(datastream).find("wavelength")
        if index == -1: return self.nan_ew
        
        ew = float(str(datastream)[index+117:index+122])
        return ew

    def stop_thread (self,T):
        T._Thread__stop()

    def create_thread (self,target):
        t0 = threading.Thread(target=target)
        # Make threads daemonic, i.e. terminate them when main thread
        # terminates. From: http://stackoverflow.com/a/3788243/145400
        t0.daemon = True
        t0.start()
        return t0
        
    def go(self):
        self.start_time = time.time()
        self.threads.append(self.create_thread(self.run_moog))

    def stop (self):
        self.verbose = False
        self.quit = True
        time.sleep(1)
        for t0 in self.threads: self.stop_thread(t0)
        if self.logfile is not None: self.logfile.close()

    def get_output_data (self,ewbounds=None):
        data = []
        if ewbounds is not None:
            ewb = (np.min(ewbounds),np.max(ewbounds))
            if ewb[0] == ewb[1]: ewb[1] = 1e30
        
        for arr in self.output_data:
            _vals = arr[:-1]
            ew = arr[-1]
            if (ewbounds is not None) and (not (ewb[0] < ew < ewb[1])): continue
            data.append(arr)
        return np.asarray(data)
    
    def create_linelist (self, fname, headerline='default',clobber=True,ewbounds=None):
        if headerline == 'default':
            if self.moog_model == 'FINALMODEL': modfile = self.model_info
            else: modfile = self.moog_model
            headerline = "# This line list was created from an input line list into MOOG's ewfind routine using model file "+modfile

        if ewbounds is not None:
            ewb = (np.min(ewbounds),np.max(ewbounds))
            if ewb[0] == ewb[1]: ewb[1] = 1e30

        data = []
        mooglist = write_moog_lines_in(fname,headerline,clobber=clobber)
        for arr in self.output_data:
            vals = arr[:-1]
            ew = arr[-1]
            if (ewbounds is not None) and (not (ewb[0] < ew < ewb[1])): continue
            data.append(arr)
            mooglist.add_line(*vals,ew=ew)
        mooglist.close()
        return np.asarray(data)
    
def join_threads(threads):
    """
    Join threads in interruptable fashion.
    From http://stackoverflow.com/a/9790882/145400
    """
    for t in threads:
        while t.isAlive():
            t.join(5)

def ewfind_model (lines_in,model,modtype='ODFNEW',fname_out='default'):
    """
PURPOSE:
   Take a model input and linelist and create a ewfind output
   
CATEGORY:
   MOOG

INPUT ARGUMENTS:
    lines_in : (string or array) Gives the file name for a line list to be read in from the file must be MOOG readable (see WRITEMOOG.ps)
                if array then give the data to create a line list from (wl, spe, ep, loggf) as the columns
    model : (array) Gives the atmosphere model  [teff,logg,feh,turb]
    
INPUT KEYWORD ARGUMENTS:
    modtype : (string) the type of model to use, Possible Choices ->
        ODFNEW -> 
        AODFNEW -> Same as the ODFNEW only with an alpha enhancement
        NOVER ->
        KUROLD ->

    fname_out : (string) Gives the output filename, 'default' ==> 'llist_{teff}p{logg}pm{feh}p{vt}.txt'
    wlrange  : (array-like) Gives (lower_bound,upper_bound) anything else and it will default to the lines_in wavelength range
    clean_up : (bool) If True it will delete temporary files 
    verbose  : (bool) will give printed progress outputs
    
    **moogpars  Keywords to be included in the batch.par file with will be given to MOOG. See the function write_moog_par

OUTPUTS:
   (dictionary) MOOG output files
       "summary_out" -> gives the summary output file which was created
       "standard_out" -> gives the standard output file which was created

DEPENDENCIES:
   External Modules Required
   =================================================
    None
   
   External Functions and Classes Required
   =================================================
    moog07_exe, get_model_name, create_kurucz_atmo_model
       
NOTES:
    (1) This runs moog_ewfind and returns that output

EXAMPLE:
   >>> models = [[5000,4.1,-2.13,1.1],[4700,3.9, -1.75, 1.2]]
   >>> lines_in = 'my_linelist_file.txt"
   >>> for model in models:
   >>>      moog_ewfind_model(lines_in, model, modtype='ODFNEW', fname_out='default')
   >>>

MODIFICATION HISTORY:
    13, Jun 2013: Dylan Gregersen
        
    """
    check_moog07(True)
    teff,logg,feh,turb = model

    from lte_atmospheres import create_kurucz_atmo_model
    check = create_kurucz_atmo_model(teff,logg,feh,turb,modtype=modtype,verbose=False)
    if not check: return False
    
    if fname_out == 'default':
        modfile = get_model_name(teff,logg,feh,turb)
        fname_out = 'llist_'+modfile+'.txt'

    check = ewfind(lines_in, 'FINALMODEL',fname_out,verbose=False)
    return check

def run_ewfind ():
    """
PURPOSE:
    This prompts the user for inputs to autoEWfind then runs the program
   
CATEGORY:
   MOOG    

DEPENDENCIES:
   External Modules Required
   =================================================
    os, numpy
   
   External Functions and Classes Required
   =================================================
    get_filename, run_create_atmo_model
    get_bounds, moog_ewfind
    
NOTES:
    
EXAMPLE:
   >>> run_moog_ewfind()

MODIFICATION HISTORY:
    13, Jun 2013: Dylan Gregersen

    """
    from .lte_atmospheres import run_create_atmo_model #@UnresolvedImport

    check_moog07(True)    
    # get the input filename
    inputf, = get_filename("Please give input linelist file:",'r')
    linelist = np.loadtxt(inputf,usecols=[0,1,2,3])

    run_create_atmo_model() # ==> FINALMODEL

    ewb = get_bounds("Give EW bounds: ",True,default=None,
                     display_help='Seporate values by spaces. If one value given it will use it as a lower bound, two values become upper and lower, no values then the entire range will be used\n  EXAMPLE===> for EWs between 10 and 200 mA ;; Give EW bounds: 10 200')


    wlb = get_bounds("Give wavelength bounds: ",False,None,
                     display_help="Seporate values by spaces. Give two values lower_bound and upper_bound or no values to use the default range taken from the input linelist.\n EXAMPLE====> for wavelengths between 3600 7000 ;; Give wavelength bounds:  3600  7000")

        
    # get output filename
    output_file, = get_filename("Please give output file name:",'w')

    log_file, = get_filename("Please optional give a logfile, for no file press enter:",'w',default=False)
    if log_file == False: log_file = None

    print " "
    print "- "*60
    print " "
    ewfind('FINALMODEL',linelist,output_file,verbose=True,ewbounds=ewb,wlbounds=wlb,logfile=log_file,linelist_filename=inputf)
        
