# these are functions which are useful for dealing with MOOG, independent of the MOOG executable

import pdb #@UnusedImport
import os, time, re
from copy import deepcopy
from datetime import datetime
import numpy as np
import numpy.lib.recfunctions as np_recfunc 
import xlwt
from .utils import int_to_roman, SpreadsheetCells

from ..utils.prompt_user import yesno, get_filename, get_choice
from ..utils.matching import get_matches

pass
#####################################################################################################################
# Misc. Functions
  
def user_get_smoothing_func_pars (smoothing_func_pars):
    """ 
    To get the smoothing function parameters
    """
    output = {}
    print "Please give appropriate values, press enter to keep previous"
    for key in smoothing_func_pars:
        prompt = key
        if key.replace(" ",'_') == 'smoothing_scaler': prompt += " (gaussian_smoothing/wavelength)"
        prompt += ": "
        value = raw_input(prompt)
        
        if value == '': value = smoothing_func_pars[key]

        if value.find("/") != -1:
            value = value.split("/")
            try: value =  float(value[0])/float(value[1])
            except: pass
    
        output[key] = value
    return output

pass

# TODO: change Batom to be AbundanceSystem, which will contain the AndersGreeves(Batom) and Lodders, etc

class Batom (object):
    """
    
    loads the data from the file Batom.f written by Chris Sneden in MOOG
    "'solar' abundance set of Anders and Grevesse (1989, Geochim. Cosmichim. Acta, 53, 197" 
                      - from WRITEMOOG (http://www.as.utexas.edu/~chris/codes/WRITEMOOG.ps)
       
    This creates a batom object which can be access via a getitem call 
    
    Parameters
    ----------
    none
    
    Attributes
    -------
    batom : list
        The primary list which the data is stored
    batom_by_el : dictionary
        A dictionary of the self.batom information with element names as keys
    batom_by_z : dictionary
        A dictionary of the self.batom information with proton number as keys
    elements : list
        Returns a list of all the element names
    array : ndarray
        Returns a numpy array with columns (Z,abundance)
        
    
    Raises
    ------
    KeyError : If the key given to the getitem call isn't correct
    
    
    Notes
    -----
    __1)__ For more information see WRITEMOOG (http://www.as.utexas.edu/~chris/codes/WRITEMOOG.ps)
    __2)__ Can use a getitem call (e.g. batom[item]) to access elements (an integer doesn't map to the
        index in main self.batom list but to a element of that Z)
    __3)__ getitem access works for element names (case-insensitive and can include species info with a space)
        it also works with proton number (species can be given but the value rounds)
    
    Examples
    --------
    >>> batom = Batom()
    
    >>> print(batom[1])
    [1,"H","Hydrogen",12.00]
    
    >>> print batom[26.1]
    [26, 'Fe', 'Iron', 7.52]
    
    >>> print(batom['Hg'])
    [80, 'Hg', 'Mercury', 1.09]
    
    >>> print(batom['au']
    [79, 'Au', 'Gold', 0.83]
    
    >>> print(batom['ba II'])
    [56, 'Ba', 'Barium', 2.13]
    
    >>> # you can also give a list/array of Z
    >>> batom[[22.1,22,20.0,25.2]]
    array([ 4.99,  4.99,  6.36,  5.39])
    
    >>> # can use to convert
    >>> batom.to_xfe(22, 3.1, -2.3)
    0.4099999999999997
    
    >>> batom.to_xfe([22,22,20,25],[3.1,3.2,4.4,3.1],-2.3)
    array([ 0.41,  0.51,  0.34,  0.01])
    
    Modification History
    --------------------
    13, Jun 2013: Dylan Gregersen
        - Updated comments and format of returns
    20 July 2013: Dylan Gregersen
        - changed from a function to a class
              
    """
    
    batom=[[1,"H","Hydrogen",12.00],
            [2,"He","Helium",10.99],
            [3,"Li","Lithium",3.31],
            [4,"Be","Beryllium",1.42],
            [5,"B","Boron",2.88],
            [6,"C","Carbon",8.56],
            [7,"N","Nitorgen",8.05],
            [8,"O","Oxygen",8.93],
            [9,"F","Florine",4.56],
            [10,"Ne","Neon",8.09],
            [11,"Na","Sodium",6.33],
            [12,"Mg","Magnesium",7.58],
            [13,"Al","Aluminium",6.47],
            [14,"Si","Silicon",7.55],
            [15,"P","Phosphorus",5.45],
            [16,"S","Slufer",7.21],
            [17,"Cl","Chlorine",5.50],
            [18,"Ar","Argon",6.56],
            [19,"K","Potassium",5.12],
            [20,"Ca","Calcium",6.36],
            [21,"Sc","Scandium",3.10],
            [22,"Ti","titanium",4.99],
            [23,"V","Vanadium",4.00],
            [24,"Cr","Chromium",5.67],
            [25,"Mn","Maganese",5.39],
            [26,"Fe","Iron",7.52],
            [27,"Co","Cobalt",4.92],
            [28,"Ni","Nickel",6.25],
            [29,"Cu","copper",4.21],
            [30,"Zn","Zinc",4.60],
            [31,"Ga","Gallium",2.88],
            [32,"Ge","Germanium",3.41],
            [33,"As","Arsnic",2.37],
            [34,"Se","Selenium",3.35],
            [35,"Br","Bormine",2.63],
            [36,"Kr","Krypton",2.23],
            [37,"Rb","Rubidium",2.60],
            [38,"Sr","Strontium",2.90],
            [39,"Y","Yttrium",2.24],
            [40,"Zr","Zirconium",2.60],
            [41,"Nb","Niobium",1.42],
            [42,"Mo","Molybdenum",1.92],
            [43,"Tc","Technetium",0.00],
            [44,"Ru","Ruthenium",1.84],
            [45,"Rh","Rhodium",1.12],
            [46,"Pd","Palladium",1.69],
            [47,"Ag","Silver",1.24],
            [48,"Cd","Cadmium",1.86],
            [49,"In","Indium",0.82],
            [50,"Sn","Tin",2.00],
            [51,"Sb","Antimony",1.04],
            [52,"Te","Tellurium",2.24],
            [53,"I","Iodine",1.51],
            [54,"Xe","Xenon",2.23],
            [55,"Cs","Caesium",1.12],
            [56,"Ba","Barium",2.13],
            [57,"La","Lanthanum",1.22],
            [58,"Ce","Cerium",1.55],
            [59,"Pr","Praseodymium",0.71],
            [60,"Nd","Neodymium",1.50],
            [61,"Pm","Promethium",0.00],
            [62,"Sm","Samarium",1.00],
            [63,"Eu","Europium",0.51],
            [64,"Gd","Gadolinium",1.12],
            [65,"Tb","Terbium",0.33],
            [66,"Dy","Dyspeosium",1.10],
            [67,"Ho","Holmium",0.50],
            [68,"Er","Erbium",0.93],
            [69,"Tm","Thulium",0.13],
            [70,"Yb","Ytterbium",1.08],
            [71,"Lu","Luletium",0.12],
            [72,"Hf","Hafnium",0.88],
            [73,"Ta","Tantalum",0.13],
            [74,"W","Tungsten",0.68],
            [75,"Re","Rhenium",0.27],
            [76,"Os","Osmium",1.45],
            [77,"Ir","iridium",1.35],
            [78,"Pt","Plantinum",1.80],
            [79,"Au","Gold",0.83],
            [80,"Hg","Mercury",1.09],
            [81,"Tl","Thallium",0.82],
            [82,"Pb","Lead",1.85],
            [83,"Bi","Bismuth",0.71],
            [84,"Po","Polomium",0.00],
            [85,"At","Astatine",0.00],
            [86,"Rn","Radon",0.00],
            [87,"Fr","Francium",0.00],
            [88,"Ra","Radium",0.00],
            [89,"Ac","Actinium",0.00],
            [90,"Th","Thorium",0.12],
            [91,"Pa","Protactinium",0.00],
            [92,"U","Uranium",0.00],
            [93,"Np","Neptunium",0.00],
            [94,"Pu","Plutonium",0.00],
            [95,"Am","Americium",0.00]]
        
    
    def __init__ (self):
        self.batom_by_el={}
        self.batom_by_z ={}
        for i in range(len(self.batom)): 
            self.batom_by_el[self.batom[i][1].lower()] = self.batom[i]
            self.batom_by_z [self.batom[i][0]] = self.batom[i]

    def __repr__ (self):
        return "instance of Batom.f"
     
    def __str__ (self):
        rep = ['[',str(self.batom[0]),",\n"]
        for i in xrange(1,len(self.batom)-1):
            rep += [" ",str(self.batom[i]),",\n"]
            
        rep += [" ",str(self.batom[-1]),"]\n"]
        return ''.join(rep)
        
    @property
    def elements (self):
        """ list of the element names """
        return [x[1] for x in self.batom]
    
    @property
    def array (self):
        """ 2-D ndarray with columns (Z,solar_abund)"""
        return np.array([[x[0],x[-1]] for x in self.batom])
    
    def to_logeps (self, spe, xfe, feh):
        """
        
        Convert value in [X/Fe] to logeps solar hydrogen scale
        
        [Fe/H] = logeps(Fe) - logeps(Fe)_sun
        logeps(X) = [X/Fe] + ([Fe/H] + logeps(X)_sun)
    
        Parameters
        ----------
        spe : float or array of floats
            Gives the specie(s) which correspond to the [X/Fe] values
        xfe : float or array of floats
            Gives the [X/Fe] bracket notation of the abundance for the 
            corresponding species
        feh : float
            the [Fe/H] value to be used for the converstion
            
        Returns
        -------
        if spe is a float:
            logeps : float
                The abundance of the species with abundance [X/Fe]
        if spe is an array of floats:
            logeps : array
                The abudnaces for the species with abundances [X/Fe]

        """
        return xfe + (feh + self.solar_abundance(spe))
        
    def to_xfe (self, spe, logeps, feh):
        """
        
        Convert value in logeps solar hydrogen scale to [X/Fe]
        
        [Fe/H] = logeps(Fe) - logeps(Fe)_sun
        [X/Fe] = logeps(X) - ([Fe/H] + logeps(X)_sun)

        Parameters
        ----------
        spe : float or array of floats
            Gives the specie(s) which correspond to the logeps values
        xfe : float or array of floats
            Gives the logeps notation of the abundance for the 
            corresponding species
        feh : float
            the [Fe/H] value to be used for the converstion
            
        Returns
        -------
        if spe is a float:
            xfe : float
                The abundance of the species with abundance logeps
        if spe is an array of floats:
            xfe : array
                The abudnaces for the species with abundances logeps
                
        """
        return logeps - (feh + self.solar_abundance(spe))

    def solar_abundance (self,spe):
        """
        Get the solar logeps abundance for a species (or list of species)
        
        Parameters
        ----------
        spe : float or array of floats
            Give the species identification for which to return the solar abundance
                    
        Returns
        -------
        if spe is float:
            logeps : float
        if spe is array of floats:
            logeps : array
        """
        
        getitem = self.__getitem__(spe)
        if isinstance(getitem,list):
            return getitem[3]
        elif isinstance(getitem,np.ndarray):
            return getitem
        else:
            raise StandardError("Whoops, error in logic I should only"\
                                " have received a list or ndarray")

    def _single_species_name (self,spe):
        if isinstance(spe,int):
            return self[spe][1]
        
        elif isinstance(spe,float):
            z = self[spe][1]
            ionz = int(round((spe - round(spe-0.5))*10))+1
            return z+" "+int_to_roman(ionz)
        
        else:
            return 'unknown'

    def species_name (self,spe):
        """
        Takes a species id and converts to a name representation
        
        species_id := proton_number + 0.1*(ionization_state)
        where the ionization_state is 0 for ground, 1 for singally, etc
        
        """
        if isinstance(spe,(float,int)):
            return self._single_species_name(spe)
        elif isinstance(spe,(list,tuple,np.ndarray)):
            return [self._single_species_name(s) for s in spe]
        else:
            raise TypeError("must receive either a floating value of the species or an array")

    def __iter__ (self):
        return iter(self.batom)

    def __contains__ (self,spe):
        if isinstance(spe,str):
            return spe.lower().strip().split()[0] in [x.lower() for x in self.elements]
        if isinstance(spe,float) or isinstance(spe,int):
            return spe in self.array[:,0]
    
    def __reversed__ (self):
        return reversed(self.batom)
    
    def __len__ (self):
        return len(self.batom)

    def _getitem (self,spe):
        """ check input and return appropriate output """
        
        if isinstance(spe,str):
            return self.batom_by_el[spe.lower().strip().split()[0]]
        
        if isinstance(spe,int) or isinstance(spe,float):
            return self.batom_by_z[int(spe)]

        if isinstance(spe,list):
            try: spe = np.asarray(spe,dtype=int)
            except IndexError:
                raise IndexError("arrays used as indices must be of integer (or boolean) type")
        
        if isinstance(spe,np.ndarray):
            if spe.shape[0] == 0: return np.array([])
            idx = np.round(spe)-1
            return self.array[idx.astype(int)][:,1]
            
    def __getitem__ (self,spe):
        try: return self._getitem(spe)
        except KeyError:
            raise KeyError("Please enter valid element name or Z, not "+str(spe))

def get_model_name (teff,logg,feh,vt,modtype=None):
    """
PURPOSE:
    Based on the input atmosphere parameters it returns a formatted string representation

CATEGORY:
    MOOG based functions

INPUT ARGUMENTS:
    teff : (float) stellar effective temperature
    logg : (float) gravitational acceleration at the surface of the star
    feh  : (float) Normalized solar metallicity [Fe/H]
    vt   : (float) stellar microturbulence
    
INPUT KEYWORD ARGUMENTS:
    modtype : (string) the type of model used, if None then no model type will be added

OUTPUTS:
    (string) representation of the model (see example)
       
DEPENDENCIES:
   External Modules Required
   =================================================
    None
   
   External Functions and Classes Required
   =================================================
    None
       
NOTES:
       (1) This is a specific naming convention

EXAMPLE:
    >>> model_representation = get_model_name(5000, 4.1, -2.13, 1.1, 'ODFNEW')
    >>> print model_representation
    "5000p410m213p110.ODFNEW

MODIFICATION HISTORY:
       13, Jun 2013: Dylan Gregersen
                    - updated to include in eyeSpec

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
    
def read_moog_linelist (fname,formatted=True,defaults={},convert_gf=False):
    """
PURPOSE:
    This function reads a MOOG formatted linelist. MOOG linelists 
    have 7 columns plus added information. See NOTES.

CATEGORY:
   MOOG functions

INPUT ARGUMENTS:
    fname : (string) The filename of the linelist, if None it will return 
    a list of lines it would output to a file

       ARG : (type) Description
       -> directory name to be searched for *.pro files

INPUT KEYWORD ARGUMENTS:
    formatted : (bool) If True it will assume formatting (7e10.3) else it will 
    split on whitespace (must have >= 7 columns)
    defaults  : (dictionary) can specify other default values for 
        'vwdamp', 'd0', 'ew'
    convert_gf: (bool) if True then will take the log10 of column 4

OUTPUTS:
    (numpy recarray) Returns a numpy record array with columns associated to the short names for the columns (see NOTES 1 and 2)
    
DEPENDENCIES:
   External Modules Required
   =================================================
    Numpy
   
   External Functions and Classes Required
   =================================================
    None
   
NOTES:
    (1) The 7 columns MOOG expects for it's linelists are:
        1) wavelength ('wl')
        2) species ID ('spe')
        3) excitation potential ('ep')
        4) oscillator strength ('loggf')
        5) Van der Waals Damping ('vwdamp')
        6) dissociation Energy ('d0')
        7) equivalenth width ('ew')
        all that follows is considered information ('info')   
        
    (2) In NOTE_1 the short hand names for the various columns are given in ('short_name')
    
    (3) MOOG formats 10 spaces with three decimal places for the 7 columns (7e10.3) 
        or (if all columns are supplied including default zeros for columns 5-7) then
        an unformatted read can be done where the splitting is done on whitespace


EXAMPLE:
    >>> linelist = read_moog_linelist("my_moog_linelist")
    >>> wavelengths = linelist['wl']
    >>> species = linelist['spe']
    >>> transition = linelist[2] # this will give all columns for the line

MODIFICATION HISTORY:
       13, Jun 2013: Dylan Gregersen
    
    """
    
    
    # check default values for columns
    vwdamp = 0.0
    d0 = 0.0
    ew = 0.0
    if 'vwdamp' in defaults: vwdamp = defaults['vwdamp']
    if 'd0' in defaults: d0 = defaults['d0']
    if 'ew' in defaults: ew = defaults['ew']


    f = open(fname)
    lines = f.readlines()
    f.close()
    
    # guess formatting is formatted is None
    if formatted is None:
        formatted = True
        for line in lines:
            if line.strip() == '' or line.strip()[0] == '#': continue
            try: 
                wl = float(line[:10])
                formatted = True
            except: pass
            
    # list to hold data
    data = []
    
    
    # update to have unformatted reads
    if formatted:
        for line in lines:
            if line.strip()=='' or line.strip()[0] == "#":continue
            
            data.append([line[:10], # wavelength
                         line[10:20], # species
                         line[20:30], # excitation potential
                         line[30:40], # oscillator strength
                         (line[40:50].strip() or vwdamp), # Van der Waals Damping
                         (line[50:60].strip() or d0), # Dissociation energy
                         (line[60:70].strip() or ew), # Equivalent Width
                         str(line[70:].strip())]) # extra information 
    else:
        for line in open(fname):
            sline = line.rstrip().split()
            
            info = "F="+fname
            
            if len(sline) == 8:  wl,spe,ep,loggf,vwdamp,d0,ew,info = sline
            elif len(sline) == 7: wl,spe,ep,loggf,vwdamp,d0,ew = sline
            elif len(sline) == 6: wl,spe,ep,loggf,vwdamp,d0 = sline
            elif len(sline) == 4: wl,spe,ep,loggf = sline
            else: raise ValueError("Wrong columns length when split on white space for: "+line)
            data.append([wl,spe,ep,loggf,vwdamp,d0,ew,info])
        
    dtypes = [('wl',float),
              ('spe',float),
              ('ep',float),
              ('loggf',float),
              ('vwdamp',float),
              ('d0',float),
              ('ew',float),
              ('info','a200')]
    
    data = np.rec.array(data,dtype=dtypes)
    if convert_gf: data['loggf'] = np.log10(data['loggf'])
    return data
    
class simple_llist_data:
    """
    Holds information for a linelist file
    the wavelengths taken from the first column 
    the lines from the actual file
    """
    def __init__ (self,fname):
        """
        Takes the linelist filename and then reads it in
        """
        self.fname = fname
        self.readin_file(fname)
        
    def readin_file (self,fname):
        """
        Read in the linelist file
        """
        f = open(fname)
        wls = []
        lines = []
        for line in f:
            line = line.rstrip()
            sline = line.split()
            # skip comment lines
            if len(sline) == 0 or line[0] == '#': continue
            # get the wavelength information
            wls.append(float(sline[0]))
            # get the full line
            lines.append(line)
        
        self.wls = np.abs(np.asarray(wls))
        self.lines = np.asarray(lines)
        # find the bounds
        self._wlbounds = (np.min(self.wls),np.max(self.wls))

    def get_wlbounds (self):
        return self._wlbounds
    
    def get_cropped_lines (self,wlbounds):
        """
        Crop the linelist by wavelength and return those lines
        wlbounds = (lower_b, upper_b)
        """
        dmin = np.min(self.wls)
        dmax = np.max(self.wls)
        
        default = False
        try: wlstart,wlend = wlbounds
        except: 
            default = True
            wlstart, wlend = dmin, dmax
        
        if not default:
            if wlstart < dmin: wlstart = dmin
            if wlend > dmax: wlend = dmax

        mask = (wlstart < self.wls)*(self.wls < wlend)
        outlines = self.lines[mask]
        if len(outlines) == 0: raise ValueError("No lines found for range:"+str((wlstart,wlend)))
        self._wlbounds = (wlstart,wlend)
        return outlines
 
pass
#####################################################################################################################
# Input functions

def write_moog_par (driver,filename=None,clobber=True,**moogpars):
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
    damping             (integer) see WRITEMOOG.ps, possible values are *0*, 1, 2

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
    
    isotopes            (array) Parameters for the isotopes
                           equals [[isotope_num, ratio1, ratio2,...],[isotope_num, ratio1, ratio2,...],...] 
                           where the number of ratios given equals the number of synthesis done  
                           The max number of ratios to give is 5
    
    -----------------   ---------------------------------------------------------------------------------------                                                
    
    fluxlimits           (array) gives the wavelength parameters for flux curves
                            equals [start, stop, step] as floating points
    
    -----------------   ---------------------------------------------------------------------------------------                                                
    
    coglimits            (array) gives the log(W/lambda) limits for curves-of-growth
                            equals [rwlow, rwhigh, rwstep, wavestep, cogatom] as floating points
    
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
    pass
    #######################################################################################################

    # This value is given in the MOOG file Atmos.f
    # the limit is given here to prevent later undo problems 
    max_filename_length = 80
 
    # these are the possible drivers, check
    moogdrivers = ['synplot','synth','cogsyn','blends','abfind','ewfind','cog','calmod',
                   'doflux','weedout','gridsyn','gridplo','binary','abpop','synpop']
    if driver not in moogdrivers: raise ValueError("MOOG driver must be in: "+", ".join(moogdrivers))
 
    
    # these shortcuts convert moogpars inputs to those which will be understood
    shortcuts = {'stdout':'standard_out'}
    
    # fix up the moog parameters
    correct_moogpars = deepcopy(moogpars)
    moogpars = {}
    for key in correct_moogpars:
        newkey = str(key).lower()
        if newkey in shortcuts: newkey = shortcuts[newkey]
        moogpars[newkey] = correct_moogpars[key]
        
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
    moog_keyword_values('damping',['0','1','2'],0)
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
    
    
    for key in moogpars: print "Keyword unknown to MOOG pars :"+str(key)
    
    if filename is None: return parlines
    else:
        FILE = open(filename, "w")
        FILE.writelines("\n".join(parlines)+"\n")
        FILE.close()                  

class write_moog_lines_in:
    """
PURPOSE:
    Use to write MOOG readable linelists to use for MOOG parameter lines_in

CATEGORY:
   MOOG class

INPUT ARGUMENTS: on initialize
    filename : (string) the output file name

INPUT KEYWORD ARGUMENTS: on initialize
    headerline : (string) notes to add to the first line of MOOG if None then no line will be added
    oneline : (list or array) If you want just one line this does it quickly
              [wl,spe,ep,loggf] optionally add += [vwdamp, d0, ew] in that order
    
    clobber : (bool) delete file if it exists

OUTPUTS:
   OUT : (type) Description

METHODS:
    write      : use to write out a line (as string) to the file
    writelines : iterate over object and write each entry to file
    add_line   : take input variables and add a line to the file
    close      : close the open file
       
DEPENDENCIES:
   External Modules Required
   =================================================
   None
   
   External Functions and Classes Required
   =================================================
   None
       
NOTES:
    None
       
EXAMPLE:
    >>> linelist = write_moog_lines_in('lines_file.ln',headerline='# this is a new linelist file')
    >>> linelist.add_line(wl,spe,ep,loggf)
    >>> linelist.close()

MODIFICATION HISTORY:
       13, Jun 2013: Dylan Gregersen
    
    """
    def __init__ (self,filename, headerline='default',oneline=None,clobber=True):
        # check if you're going to overwrite
        if os.path.exists(filename) and not clobber: raise ValueError("About to overwrite file '"+filename+"'")
        
        # open the file
        self.file = open(filename,'w')
        
        # write the header line if desired
        if headerline == 'default': headerline = "# Automatically generated subset of primary line list. "
        if headerline != None: self.file.write(str(headerline)+"\n")
        
        # if oneline, then write that line and close
        if oneline is not None:
            self.add_line(*oneline[:7])
            self.close()

    def writelines (self,lines):
        """
PURPOSE:
    To write a list of lines using the 'write' method
        """
        for line in lines: self.write(line)
    
    def write (self,line):
        """
PURPOSE:
    Directly add a line to the output file

CATEGORY:
   Method

INPUT ARGUMENTS:
   line : (string) The line to be inserted into the file

INPUT KEYWORD ARGUMENTS:
   None

OUTPUTS:
   None


DEPENDENCIES:
   External Modules Required
   =================================================
    None
   
   External Functions and Classes Required
   =================================================
    None
       
NOTES:
    (1) If the line is not formatted for MOOG to read it will have a problem

EXAMPLE:
    >>> linelist = write_moog_lines_in('lines_file.ln',headerline='# this is a new linelist file')
    >>> linelist.write(" 15000.3    26.0     3.14     -1.34   ")
    >>> linelist.close()    
        """
        print >> self.file, str(line)
    
    def add_line (self,wl,spe,ep,loggf,vwdamp=None,d0=None,ew=None,spe_fmt='9.5f'):
        """
PURPOSE:
    Add a line to the output file based on the input atomic parameters

CATEGORY:
   Method

INPUT ARGUMENTS: on initialize
    wl : (float) wavelength of the transition
    spe: (float) species identifier (recognizable by MOOG)
    ep : (float) excitation potential
    loggf: (float) oscillator strenth

INPUT KEYWORD ARGUMENTS: on initialize
    vwdamp : (float or None) Van der Waals damping, if None then won't be specified
    d0 : (float or None) Dissociation energy, if None then won't be included
    ew : (float or None) Equivalent width, if None then won't be included
    spe_fmt : (string) the format for the species identifier

OUTPUTS:
    None
       
DEPENDENCIES:
   External Modules Required
   =================================================
   None
   
   External Functions and Classes Required
   =================================================
   None
       
NOTES:
    None
       
EXAMPLE:
    >>> linelist = write_moog_lines_in('lines_file.ln',headerline='# this is a new linelist file')
    >>> linelist.add_line(wl,spe,ep,loggf)
    >>> linelist.close()
       """        
        fmt_string = "{0:>10.3f} {1:>"+str(spe_fmt)+"} {2:>9.3f} {3:>9.3f}" 
        data = [wl,spe,ep,loggf]
        
        for var in (vwdamp, d0, ew):
            if var != None:
                data.append(var)
                i = len(data)-1
                fmt_string += " {"+str(i)+":>9.3f}"
            else: 
                fmt_string += " "*10
        
        print >> self.file, fmt_string.format(*data)
    
    def close (self):
        """
PURPOSE:
    Close the file
        """
        self.file.flush()
        self.file.close()

pass
#####################################################################################################################
# synth Driver Functions

def parse_synth_summary_out (fname):
    """
    Parse information from the MOOG synth summary_out file

    Parameters
    ----------
    fname : string
        The filename to the MOOG summary out file from the synth driver
        
    Returns
    -------
    synth : np.ndarray, shape(2,Npts)
        xy data from the synth summary out file
    header : string
        header line which includes all the lines prior to the data block


    Notes
    -----
    __1)__ The file must be formatted by MOOG synth driver
    
    Examples
    --------
    >>> xy, header = parse_synth_summary_out("my_sum_out.txt")
    >>> wavelengths = xy[0]
    >>> synth_flux = xy[1]
        
    """
    # open the file and read in lines
    fin = open(fname)
    lines_in = fin.readlines()
    fin.close()
    
    # get the pertinent information from the first lines of the summary_out file
    header = "# "
    model_line = False
    for j in xrange(len(lines_in)):
        line = lines_in[j]
        header += line
        if line.find('MODEL') == 0: # found both "MODEL:" and "MODEL ATMOSPHERE HEADER" 
            model_line = True
            continue
        if model_line: 
            params = np.array(lines_in[j].rstrip().split(),dtype=float) # wlstart, wlend, wlstep, opacity width
            break

    # get the wavelength information
    wls = []
    wlvalue = params[0] 


    # if you need to repeat to get more data tables from the same summary output file then you can use these:
    # estimate_wls = np.arange(params[0],params[1],params[2])
    # estimate_num_rows = round(len(estimate_wls)/10+0.5) # possibly - 1

    # get the data from the table    
    data= []
    for i in xrange(j+1,len(lines_in)):
        line = lines_in[i]
        sline = line.split()
        for val in sline: 
            data.append(1.0-float(val))
            wls.append(wlvalue)
            wlvalue += params[2]
            
    # stack the data together into wl,flux pairs
    xy = np.dstack((wls,data))[0].T
    return xy, header      

def data_table_line_parser (line,strongline=False,scale_up_strength=1):
    """
    This reads the lines from the linelist data table in the moog file and then outputs the columns
    NOTE: this is a formatted read

    examples of input:
    =====================
strongline = True

INPUT LINES DATA FOR  24 STRONG LINES
 j     wave1       species        E.P.     gf     damptype  strength
  1 15004.970   H  I     1.000   12.75   1.81E-02  UNSLDc6  4.69E-07

    =====================
strongline = False

INPUT LINES DATA FOR  2476 LINES
   j     wave1       species        E.P.     gf     damptype  strength      E.W.
    1 15004.157   CN   607.01214    4.45  1.85E-04   UNSLDc6  3.31E-15      0.00
    2 15004.177   Ni I    28.000    6.34  2.13E-02   MYgamma  1.94E-09      0.00
   ....
 1158 15019.346   Si III  14.200   25.77  1.18E+00   MYgamma  9.25E-25      0.00
    

    """
    def convert_to_float (val):
        val = val.lower().replace('d','e')
        return float(val)

    if strongline:
        moog_input = int(line[:3])
        wl = convert_to_float(line[3:13])
        species = line[13:30]        
        ep = float(line[30:38])
        gf = convert_to_float(line[39:49])
        damptype = line[49:58]
        strength = convert_to_float(line[59:68])*scale_up_strength
        ew = 0.0
    else:
        # for input lines
        moog_input = int(line[:5])
        wl = convert_to_float(line[5:15])
        species = line[15:32]
        ep = float(line[32:40])
        gf = convert_to_float(line[40:50])
        damptype = line[50:60]
        strength = convert_to_float(line[60:70])*scale_up_strength
        ew = float(line[70:] or 0.0)

    spl = species.split()
    spe = float(spl[-1])
    spe_name = "_".join(spl[:-1])
    
    loggf = np.log10(gf)

    return moog_input, wl, spe_name, spe, ep, loggf, damptype, strength, ew

def parse_synth_standard_out (fname):
    """
PURPOSE:
    Parse information from the MOOG synth standard_out file
      
CATEGORY:
   MOOG synth functions

INPUT ARGUMENTS:
    fname : (string) The filename to the MOOG summary out file from the synth driver
   
INPUT KEYWORD ARGUMENTS:
   None

OUTPUTS:
   (dictionary) Stored results for various sections of information
        'data table' -> a numpy rec array of the data tables following the "INPUT LINES DATA FOR" lines, columns are based upon the columns in the file:
                       [('wl','<f8'),('spe_name','a10'),('spe','<f8'),('ep','<f8'),('loggf','<f8'),('damptype','a10'),('strength','<f8'),('ew',"<f8")]
        'model' -> an array [teff,logg,feh,vt] 
    
DEPENDENCIES:
   External Modules Required
   =================================================
   Numpy, os
   
   External Functions and Classes Required
   =================================================
    None
       
NOTES:
    (1) The file must be formatted by MOOG synth driver
       
EXAMPLE:
    >>> results = parse_synth_standard_out("my_std_out.txt")
    >>> data_table = results['data table']
    >>> wavelengths = data_table['wl']
    >>> strengths = data_table['strength']
    >>>
    >>> teff = results['model'][0]

MODIFICATION HISTORY:
       13, Jun 2013: Dylan Gregersen    
    """
    results = {}
    scale_up_strength = 1e10

    # open up the file
    f = open(fname,'r')
    lines = f.readlines()
    f.close()

    # initialize the array
    data_table = [] # hold all the information 
    spe_names = [] # species names

    # go through the lines and figure out
    strongline = False
    N = -1
    model = None
        
    table_values = []
    # find all the indicies for the table start and stop
    for i in xrange(len(lines)):
        # find the linelist information
        if lines[i].find("INPUT LINES DATA FOR") == 0: 
            if lines[i].find('STRONG') == -1: strongline = False
            else: strongline = True
            
            N = int(lines[i].split()[4])
            table_values.append((i+2,N+i+2,strongline))
            
        # find the model atmosphere used
        if model != None and lines[i].find('MODEL ATMOSPHERE HEADER:') == 0:
            model = np.array(lines[i+1].split()[:4],dtype=float)
        
        
    # go through the data table subsection and extract data
    for pair in table_values:
        strongline = pair[2]
        for i in xrange(pair[0],pair[1]):
            line = lines[i].rstrip()          
            _, wl, spe_name, spe, ep, loggf, damptype, strength, ew = data_table_line_parser(line,strongline,scale_up_strength)
            
            data_table.append((wl, spe_name, spe, ep, loggf, damptype, strength, ew))
            spe_names.append(spe_name)
    
    dtype = [('wl','<f8'),('spe_name','a10'),('spe','<f8'),('ep','<f8'),('loggf','<f8'),('damptype','a10'),('strength','<f8'),('ew',"<f8")]
    data_table = np.rec.array(data_table,dtype=dtype) 

    # sort the data by wavelength
    sortit = np.argsort(data_table['wl'])
    data_table = data_table[sortit]
    
    # get rid of duplicates 
    data_table = np.unique(data_table)
        
    # get the results to return
    results['data table']=data_table
    results['model'] = model
    
    return results

def cropping_data_table_by_strength (data_table,strength_tol=None):
    """
    Takes the numpy recarray given from the parse_synth_standard_out for the data table and 
    crops based on opacity strenth
    """
    strengths = data_table['strength']
    #strength_tol = sig_tol*np.std(data_tablet[4])/len(data_tablet[4])
    if strength_tol is None: strength_tol = np.min(strengths) + np.std(strengths)/(len(strengths))
    strength_tol_mask = strengths > strength_tol
    
    return (strength_tol_mask, strength_tol)

def crop_data_table (data_table,tol=None):
    """
PURPOSE:
    Performs a tolerance cropping by strength
    
CATEGORY:
   MOOG functions

INPUT ARGUMENTS:
   data_table : (numpy recarray) takes the data table given by parse_synth_standard_out function
                [('wl','<f8'),('spe_name','a10'),('spe','<f8'),('ep','<f8'),('loggf','<f8'),('damptype','a10'),('strength','<f8'),('ew',"<f8")]

INPUT KEYWORD ARGUMENTS:
    tol : (float or None) Gives the value of the tolerance to crop by. If None it will attempt to guess the tolerance

OUTPUTS:
    (recarray) A cropped version of the data table with only the columns ['wl','spe','ep','loggf'] 
    (ndarray) boolean array which was used to make the cropping (True == above the tolerance)
    (float) the tolerance used to make the crop
       
DEPENDENCIES:
   External Modules Required
   =================================================
    Numpy
   
   External Functions and Classes Required
   =================================================
    cropping_data_table_by_strength
       
NOTES:
    (1) 

EXAMPLE:
    >>> results = parse_synth_standard_out("my_std_out.txt")
    >>> data_table = results['data table']
    >>>
    >>> linelist_data, mask, tol = crop_data_table (data_table,tol=None)
    >>> wavelengths = linelist_data['wl']
    >>> wavelengths == data_table['wl'][mask]
    True
    

MODIFICATION HISTORY:
       13, Jun 2013: Dylan Gregersen                       
         
    """   
    mask,tol = cropping_data_table_by_strength(data_table,tol)
    # check to see if there are values above the tolerance
    if not np.any(mask): return None
    
    cols = ['wl','spe','ep','loggf']
    linelist_data = data_table[cols][mask]
    
    return (linelist_data, mask, tol)

pass
# == these are useful for looking at the relative strengths of a particular species:

def add_STR_data_table (data_table,teff):
    """
PURPOSE:
    Add the parameter STR to the numpy rec array
   
CATEGORY:
   MOOG function

INPUT ARGUMENTS:
   data_table : (numpy recarray) takes the data table given by parse_synth_standard_out function
                [('wl','<f8'),('spe_name','a10'),('spe','<f8'),('ep','<f8'),('loggf','<f8'),('damptype','a10'),('strength','<f8'),('ew',"<f8")]
   teff : (float) The effective temperature to use in equation give in NOTE 1
   
INPUT KEYWORD ARGUMENTS:
   None

OUTPUTS:
   (numpy recarray) same format as the data_table but with an added column "STR"

       
DEPENDENCIES:
   External Modules Required
   =================================================
    numpy, nump.lib.recfunctions
   
   External Functions and Classes Required
   =================================================
       
       
NOTES:
    (1) STR := loggf - (5040.0/teff)*ep
        

EXAMPLE:
    >>> data_table = add_STR_data_table(data_table,5775.0)
    >>> STR = data_table['STR']

MODIFICATION HISTORY:
       13, Jun 2013: Dylan Gregersen
                           
    """
    loggfs = data_table['loggf']
    eps = data_table['ep']
    theta = 5040.0/float(teff)
    
    STR = loggfs - theta*eps
    
    if 'STR' not in data_table.dtype.names:
        data_table = np_recfunc.append_fields(data_table,'STR',STR,usemask=False)
    else: data_table['STR'] = STR
    
    return data_table
 
def cropping_data_table_by_STR (data_table, species, teff, STR_tol=None):
    """
PURPOSE:
    Crop by a particular STR value for a particular species
   
CATEGORY:
   MOOG function

INPUT ARGUMENTS:
   data_table : (numpy recarray) takes the data table given by parse_synth_standard_out function
                [('wl','<f8'),('spe_name','a10'),('spe','<f8'),('ep','<f8'),('loggf','<f8'),('damptype','a10'),('strength','<f8'),('ew',"<f8")]
   species    : (float) A species identifier 
   teff : (float) The effective temperature to use in equation give in NOTE 1
   
INPUT KEYWORD ARGUMENTS:
   STR_tol : (float or None) The STR tolerance to crop by. If None it will make a guess 

OUTPUTS:
   (numpy recarray) same format as the data_table but with an added column "STR" for only the specified element
   (ndarray bool) This is the cropping mask for the 
   (float) the STR_tol used to create the mask

       
DEPENDENCIES:
   External Modules Required
   =================================================
    numpy, nump.lib.recfunctions
   
   External Functions and Classes Required
   =================================================
    None
       
NOTES:
    (1) STR := loggf - (5040.0/teff)*ep
      
    (2) Only works to compare the relative probablity of a particular element

EXAMPLE:
    >>> data_table, STR_mask, STR_tol = cropping_data_table_by_STR(data_table, 22.1, 5775.0, -5.0)
    >>> STR = data_table['STR'] # STR for element 22.1
    >>> 

MODIFICATION HISTORY:
       13, Jun 2013: Dylan Gregersen
                           
    """ 
    spe_mask = (data_table['spe'] == float(species))
    # if no species were selected
    if not np.any(spe_mask): return (spe_mask,STR_tol)
    
    dt = add_STR_data_table(data_table[spe_mask],teff)
    
    STR = dt['STR']

    if STR_tol is None: STR_tol = -5.0 # taken as a guess, in the solar spectrum this produces a TiII line with log(ew/lambda) ~ -6 (ew~0.5mA at lambda=5000A) 
    STR_mask = STR > STR_tol
        
    return (dt,STR_mask,STR_tol)

pass
# == these are useful for analyzing the moog synth output:

def plot_process_moog_synth (synth_smoothed, data_table, llist_mask, tol, smoothing_func_pars, observed_xydata, show=True, save_file=None):
    """
    Plotting routine used by the process_moog_synth function
    
    """
    import matplotlib.pylab as plt
    from matplotlib.ticker import FormatStrFormatter
    plt.ioff()
    fig = plt.figure(figsize=(10,7))
    
    def closeplot (event):
        if event.key not in ('q','esc'): return
        plt.close()
        
    fig.canvas.mpl_connect('key_press_event',closeplot)
    
    ############## ADD TO UPPER PLOT #############################
    # upper plot of synthesis and data
    ax_upper = fig.add_subplot(211)

    xmin = np.min(synth_smoothed[0])
    # add observed data is possible
    if type(observed_xydata) in (list,np.ndarray):
        xy = observed_xydata
        ax_upper.plot(xy[0],xy[1],marker='o',color='k',alpha=0.6)
        # change the observed bounds
        xmin = np.min(xy[0])

    # plot synthesis on upper bounds
    if len(synth_smoothed[0]) == 2: print "WARNING: I don't think you gave the synth_smoothed[0] = xarray, synth_smoothed[1] = yarray"
    ax_upper.plot(synth_smoothed[0],synth_smoothed[1],color='b')
   
    ax_upper.vlines(data_table['wl'][llist_mask],-0.5,5,color='k')
    
    ax_upper.xaxis.set_major_formatter(FormatStrFormatter('%5.1f'))

    ############## ADD TO LOWER PLOT #############################
    # lower plot shows how the cut was made, depends on which tolerance type was used
    
    ax_lower = fig.add_subplot(212,sharex=ax_upper)
    
    #if tolerance_type == 'strength':
    X,Y = data_table['wl'],data_table['strength']
    ax_lower.plot(X,Y,c='g') # plot all the strengths
    ax_lower.plot(X[llist_mask],Y[llist_mask],linestyle='none',marker='o',color='r')
    ax_lower.set_ylabel("Strength")
    ax_lower.set_ylim(0,1.5*tol)

    #    elif tolerance_type == 'STR':
    #        X = data_table[:,0]
    #        Y = data_table[:,3]-(5040/teff)*data_table[:,2]
    #        ax_lower.scatter(X[llist_mask],Y[llist_mask],s=20,marker='o',edgecolor='r',color='g',linewidths=1.5)
    #        rmask = np.logical_not(llist_mask)
    #        ax_lower.scatter(X[rmask],Y[rmask],marker='o',color='g')
    #        ax_lower.set_ylabel("STR == log(gf)-(5040/teff)*EP")
    #        ax_lower.set_ylim(-10,2)
    #        
        
    ax_lower.vlines(data_table['wl'][llist_mask],np.min(Y)-10,np.max(Y)+10,color='k')
    ax_lower.axhline(tol,linestyle='--',color='r',lw=2,alpha=.5)
    ax_lower.xaxis.set_major_formatter(FormatStrFormatter('%5.1f'))

    # set up the plots
    
    ax_upper.set_xlim(xmin,xmin+5)
    ax_upper.set_ylim(-.1,1.05)

    ax_lower.set_xlabel("Wavelength")
    ax_upper.set_ylabel("Flux")
    
    title = 'tolerance:'+str(tol)+"\n"
    kount = 0
    for key in smoothing_func_pars:
        title += '  '+key+":"+str(smoothing_func_pars[key])
        kount += 1
        if kount >= 4:
            title += '\n'
            kount = 0
            
    
    ax_upper.set_title(title,fontsize='small')

    if show: plt.show()
    if save_file is not None: fig.savefig(save_file, format='pdf')
    plt.close() 
 
def process_moog_synth_output (summary_out, standard_out, smoothing_func=None, smoothing_func_pars={}, tol=None, observed_xydata=None, interactive=True):
    """
PURPOSE:
    Take the output from MOOG synth driver (namely the summary_out file and standard_out file) and parse the information into a useful format

CATEGORY:
    MOOG functions

INPUT ARGUMENTS:
    summary_out         :  (string) File name of the summary out file from MOOG synth driver
    standard_out        :  (string) File name of the standard out file from MOOG synth driver


INPUT KEYWORD ARGUMENTS:
    smoothing_func      :  (function or None) This takes as the first argument the xy data from the synthesis (x,y = xy)
                            Then it takes all the parameters for the smoothing_func_pars as keywords (see example below)
                            if None then no smoothing will be done and smoothing_func_pars is pointless

    smoothing_func_pars :  (dictionary) Gives the keywords and values to pass to the smoothing function

    tol                 :  (float or None) The standard out file has strengths for each of the input lines. This tolerance
                            crops out everything below the value given. If None it will guess at the tolerance based on:
                            min(strengths) + std(strengths)/len(strengths)  because empirically this gave a good estimate
    
    observed_xydata     :  (array or None) if interactive then a plot will be made and the x,y data of the observed will be plotted
                            x,y = observed_xydata    
                            
    interactive         :  (bool) If True then it will prompt you with a plot that shows the observed_data, smoothed_synthesis, 
                            strengths of all the lines, the strength tolerance, and the lines which passed the tolerance
                            Once the plot is closed you have the option to change the strength tolerance and keyword values for
                            your smoothing function         

OUTPUTS:
    (array) linelist data has the (wavelength,species,EP,loggf) data for all the lines which passed the tolerance cut
    (array) synth smoothed data has the x,y data from the smoothing of the synthesis
    (float) tolerance is the value used to make the strength cut on the lines

DEPENDENCIES:
   External Modules Required
   =================================================
   Numpy, os
   
   External Functions and Classes Required
   =================================================
    plot_process_moog_synth, crop_data_table
    parse_moog_summary_out, pars_moog_standard_out
       
NOTES:
   (1) The interactive version will prompt you with questions and allow you to edit the tolerance and smoothing parameters on the go

EXAMPLE:
    ************************************************
    SMOOTHING FUNCTION EXAMPLE:
    ************************************************
    def smooth_by_scaler (xy,smoothing_scaler=1):
        if smoothing_scaler == 1: return xy
        gd = Gaussian_Density(xy[0],smoothing_scaler*xy[0])
        T = get_resampling_matrix(xy[0],xy[0],gd,perserve_normalization=True)
        xy[1] = T*xy[1]
        return xy       
    
    smoothing_func = smooth_by_scaler
    smoothing_func_pars = {'smoothing_scaler':2.3}
   
   >>> observed_xydata = np.loadtxt("my_data.txt",unpack=True)
   >>> wavelengths = observed_xydata[0]
   >>>
   >>> linelist_data, synth_smoothed, tol = process_moog_synth_output ('moog_sum.txt', 'moog_std.txt', smoothing_func=smooth_by_scaler, smoothing_func_pars={'smoothing_scaler':2.3}, tol=None, observed_xydata=observed_xydata, interactive=True)
   >>> wl,spe,ep,loggf = linelist_data[0]



MODIFICATION HISTORY:
    13, Jun 2013: Dylan Gregersen

    """
        
    # get the (wl,flux) data from the summary out file(s)
    # !! could create so that you have a filelist
    synth, _header = parse_synth_summary_out(summary_out) 
    synth_smoothed = synth.copy()
    
    do_smoothing = (smoothing_func is not None)
    if do_smoothing:
        # check smoothing_func
    
        if type(smoothing_func).__name__ != 'function': raise ValueError("Must give a smoothing_func value which is type function")
        if type(smoothing_func_pars) != dict: raise ValueError("Smoothing func pars must be type dict")    
        synth_smoothed = smoothing_func(synth_smoothed,**smoothing_func_pars)



    proper_format = 'Your smoothing function needs to return a numpy array that has x = array[0] and y = array[1], aka shape=(2,number_points)'
    if synth_smoothed.ndim != 2:   raise ValueError("Dimension Wrong ==> "+proper_format)
    if synth_smoothed.shape[0] != 2: raise ValueError("Shape Wrong ==> "+proper_format)

    if interactive: print "getting data table"
    # get the lines based on a tolerance
    parsed_std =  parse_synth_standard_out(standard_out)
    if parsed_std is None: raise ValueError( "Error: Problem parsing the standard out file, function returned None")
    if interactive: print "cropping table by tolerance"   
    linelist_data, llist_mask, tol  = crop_data_table(parsed_std['data table'],tol=tol)
    
    
    ####################
    # inspect the data, lines, and smoothed data
    if interactive: print "-"*60+"\n"

    # visual inspection of result
    if interactive and yesno("View data with lines?",'y'):
        while True:
            print ">"*10+" plot data to check tolerance threshold and synthesis smoothing"
            plot_process_moog_synth(synth_smoothed, parsed_std['data table'], llist_mask, tol, smoothing_func_pars, observed_xydata)
            break_it = True
            
            # get a new tolerance, or not
            if yesno("recalc lines with new tolerance?",'n'):
                try: tol = float(input("give new tol: "))
                except: 
                    print "Invalid tolerance ==>  proceed directly to go :)"
                    continue
                linelist_data, llist_mask, tol = crop_data_table(parsed_std['data table'],tol=tol)
                break_it = False

            # get a new smoothing, or not
            if do_smoothing and yesno("recalc synthesis smoothing?",'n'):
                smoothing_func_pars = user_get_smoothing_func_pars(smoothing_func_pars)
                print "Smoothing synthesis spectrum. This may take sometime to compute, so take a breath and relax for a moment :)"
                tprev = time.time()
                synth_smoothed = smoothing_func(synth.copy(),**smoothing_func_pars)
                print "Time to smooth:",time.time() - tprev
                break_it = False

#            if yesno("run python debugger?",'n'): 
#                pdb.set_trace()
#                break_it = False
                
            # if neither new smoothing or tolerance then break
            if break_it: break

    return linelist_data, synth_smoothed, tol

def write_linelist_data (linelist_data, linelist_outfile, clobber=True, headerline='default'):
    """
PURPOSE:
    Takes linelist_data = [[wl,spe,ep,loggf],...] and writes it out to a MOOG readable format


CATEGORY:
   MOOG functions

INPUT ARGUMENTS:
    linelist_data : (array) must have rows of four columns [[wl,spe,ep,loggf],...]
    linelist_outfile : (string) The output filename

INPUT KEYWORD ARGUMENTS:
    clobber : (bool) If the linelist_outfile exists and True then will overwrite, otherwise it will raise warning
    headerline : (string) The comment line to appear at the top of the file. "default" gives the default header line

OUTPUTS:
   None

DEPENDENCIES:
   External Modules Required
   =================================================
    None
   
   External Functions and Classes Required
   =================================================
    None
       
NOTES:
   (1) This was written to deal with the output of the process_moog_synth_output function
   
   (2) Note that all the zeros in the species ID is just to deal with the molecular ids

EXAMPLE:
    >>> linelist_data, synth_smoothed, tol = process_moog_synth_output ('moog_sum.txt', 'moog_std.txt', smoothing_func=smooth_by_scaler, smoothing_func_pars={'smoothing_scaler':2.3}, tol=None, observed_xydata=observed_xydata, interactive=True)
    >>>
    >>> write_linelist_data(linelist_data, 'file_linelist.ln', clobber= True, headerline = "# I created this file")
    >>>


MODIFICATION HISTORY:
    13, Jun 2013: Dylan Gregersen

    """
    if os.path.exists(linelist_outfile) and not clobber: raise IOError("File already exists: '"+linelist_outfile+"'")
    
    f = open(linelist_outfile,'w')
    if headerline == 'default': headerline = "# line list created from a subset of strong lines in a synthesis \n"
    else:
        if headerline[-1] != '\n': headerline += '\n'
    f.write(headerline)
    
    fmt_string  = "{0:>10.3f} {1:>9.5f} {2:>9.3f} {3:>9.3f}"+" "*30 # wl,spe,ep,loggf
    for i in xrange(len(linelist_data)): f.write(fmt_string.format(*linelist_data[i])+"\n")
    f.close()

def write_synth_data (synth_xydata, synth_outfile, clobber=True):
    """
PURPOSE:
    Takes smoothed xy synthesis data and write it out

CATEGORY:
   MOOG functions

INPUT ARGUMENTS:
    synth_xydata : (array) Has wavelength flux as synth_xydata[0] = wavelength
    synth_outfile : (string) Give the output filename for the data

INPUT KEYWORD ARGUMENTS:
    clobber : (bool) If the linelist_outfile exists and True then will overwrite, otherwise it will raise warning

OUTPUTS:
   None

DEPENDENCIES:
   External Modules Required
   =================================================
    None
   
   External Functions and Classes Required
   =================================================
    None
       
NOTES:
   (1) This was written to deal with the output of the process_moog_synth_output function
   
EXAMPLE:
    >>> linelist_data, synth_smoothed, tol = process_moog_synth_output ('moog_sum.txt', 'moog_std.txt', smoothing_func=smooth_by_scaler, smoothing_func_pars={'smoothing_scaler':2.3}, tol=None, observed_xydata=observed_xydata, interactive=True)
    >>>
    >>> write_synth_data(synth_smoothed, 'synthdata.txt', clobber= True)
    >>>


MODIFICATION HISTORY:
    13, Jun 2013: Dylan Gregersen
    Takes xy data and writes it to the synth_outfile
    
    If clobber = False and the file exists it will raise an error
    
    """
    if os.path.exists(synth_outfile) and not clobber: raise IOError("File already exists: '"+synth_outfile+"'")
    synth_xydata = np.asarray(synth_xydata)
    np.savetxt(synth_outfile, synth_xydata.T, fmt='%10.5f  %10.5f', delimiter="  ")


pass
#####################################################################################################################
# abfind driver functions

def parse_abfind_summary_out (summary_out, logeps=True):
    """
    This reads through a MOOG abfind summary out file and extracts the abundance information
       
    Parameters
    ----------
    summary_out : string 
        Gives the filename of the MOOG abfind summary out file    
    
    logeps : boolean
        If 'True' then it returns the abundance as logeps if 'False' then returns
        in [X/Fe] notation
        
    Returns
    -------
    abundances : ndarray 
        This array has columns: wavelength, species, ep, loggf, ew, abundance
    
    
    Notes
    -----
    
    
    Examples
    --------
    >>> abunds = parse_abfind_summary_out("summary_out.sum")
    >>>
    
    
    
    Modification History
    --------------------
    12, Jun 2013: Tim Anderton    
    13, Jun 2013: Dylan Gregersen
        - replaced original dictionary with load_Batom which does the same task

    """
    header = {'info':None,
              'teff':None,
              'logg':None,
              'feh':None,
              'vt':None}
    
    abundancestatistics = {} #mean abundance and standard deviation pairs
    linedata = {}
    infile = open(summary_out, "rb")
    
    # define the expressions to find
    paramsexp = re.compile("(\d+\.[\d+, *]) +(\d*\.\d+) +([\ ,+,-]\d\.\d+) +(\d*\.\d+)") #teff,logg,feh,vt
    elemidentexp = re.compile(r"Abundance Results for Species [A-Z][a-z]* +I+")
    lineabexp = re.compile(r" *(\d+\.\d+) +(\d+\.\d+) +")
    statlineexp = re.compile(r"average abundance = +\d\.\d\d +std\. +deviation = +\d\.\d\d")
    
    # modellineexp = re.compile(r"\d+g\d\.\d\dm-?\d\.\d+\v\d")
    currentspecies = None
    
    for i,line in enumerate(infile):
        p = paramsexp.search(line)
        if p is not None:
            teff,logg,feh,vt = [float(st) for st in p.groups()]
            header['teff'] = teff
            header['logg'] = logg 
            header['feh'] = feh 
            header['vt'] = vt 
            continue
        
        if i == 0:
            header['info'] = line.rstrip()
            continue
                                
        _l = lineabexp.match(line)
        if _l is not None:
            #print "new linedata", line
            #line is ordered like wv ep logGF EW logrw, abund, del avg
            linedatum = [float(st) for st in line.split()]
            linedata[currentspecies].append(linedatum)
            continue
        
        m = elemidentexp.search(line)
        if m is not None:
            #print "new element", line
            elemline = m.group().split()
            currentspecies = elemline[-2] + " " + elemline[-1]
            #species_id_num = float(elemline[-2]) + 0.1*(int(elemline[-1])-1)
            linedata[currentspecies] = []
            abundancestatistics[currentspecies] = []
            continue
        
        s = statlineexp.search(line)
        if s is not None:
            #print "new stat"
            spl = s.group().split()
            meanab = float(spl[3])
            sig = float(spl[-1])
            num = line.split()[-1]
            abundancestatistics[currentspecies].append([meanab, sig, num])
            continue
        
    infile.close()
    out_ldat = []
    batom = Batom()
    for cspecies in linedata.keys():
        for lidx in range(len(linedata[cspecies])):
            ldm = linedata[cspecies][lidx]
            species_parts = cspecies.split()
            species_id_num = batom[species_parts[0]][0] + 0.1*(len(species_parts[1])-1)
            #output line list format Wv, species, ep, logGF, EW, Abundance
            out_ldat.append((ldm[0], species_id_num, ldm[1], ldm[2], ldm[3], ldm[5]))
    out_ldat = np.array(out_ldat)
    
    if logeps:
        return out_ldat, header

    # convert logeps to [X/Fe]
    elements = np.round(out_ldat[:,1])
    fe_idx = (elements==26.0)
    feh = np.mean(out_ldat[:,5][fe_idx]) - batom['Fe'][-1]
    
    for z in np.unique(elements):
        out_ldat[:,5][fe_idx] = batom.to_xfe(z,out_ldat[:,5][fe_idx],feh)

    return out_ldat, header

def abundance_totals (data):
    """
    Take the data output from the function parse_abfind_summary_out and combine the abundances and perform statistics checks
    
    """
    errmsg = "data must be a 2-D numpy array (N_lines,N_columns) where columns=(wavelength, species, ep, loggf, ew, abundance)"
    if not isinstance(data,np.ndarray):
        raise TypeError(errmsg)
    
    if data.ndim != 2 or data.shape[1] != 6:
        raise ValueError(errmsg)
    
    species =  np.sort(np.unique(data[:,1]))
    # batom = Batom()
    # species_names = [batom[x][1] for x in species]
    
    abundances = []
    for spe in species:
        abunds = data[data[:,1] == spe]        
        N = len(abunds)
        abundances.append([spe,np.mean(abunds[:,5]),np.std(abunds[:,5]),N])
        
    return np.array(abundances)

default_style = 'font: name Arial; alignment: horizontal left'
yellow_background = default_style+"; pattern: pattern solid, fore_colour light_yellow;"
light_gray = default_style+'; pattern: pattern solid, fore_colour gray25;' 
dark_gray = default_style+'; pattern: pattern solid, fore_colour gray50;'              

                   
class Ablines (object):
    width = 8
    left_width = 6
    top_height = 6
    batom = Batom()
     
    def __init__ (self,summary_filelist, lines_in=None, compile_linelist=False, wl_tol=0.07,loggf_tol=.01,ep_tol=0.05, notation='l',):

        if xlwt is None:
            # TODO: make it not dependant on xlwt by changeing how the styles are done
            raise ImportError("this class needs xlwt to work")
        
        
        self.styles = {'lgray':xlwt.easyxf(light_gray,num_format_str = '#0.0'),
                      'dgray' :xlwt.easyxf(dark_gray ,num_format_str = '#0.0'),
                      
                      'yellow_int'     : xlwt.easyxf(yellow_background, num_format_str = '#0'),
                      'yellow_float00' : xlwt.easyxf(yellow_background, num_format_str = '#0.00'),
                      'yellow_float000': xlwt.easyxf(yellow_background, num_format_str = '#0.000'),
                      
                      'default_int': xlwt.easyxf(default_style, num_format_str = '#0'),
                      'default'    : xlwt.easyxf(default_style, num_format_str = '#0.00'),
                      'default_float000':  xlwt.easyxf(default_style, num_format_str = '#0.000')}       
               
    
            
        self._check_init_values()
        
        # define the tolerances
        minimum_tol = 1e-13
        self.tolerances = np.array([float(wl_tol or minimum_tol),
                                    float(ep_tol or minimum_tol),
                                    float(loggf_tol or minimum_tol)])

        # define the data dictionary based on files read in
        self.data = {}

        # if there's a linelist then read it in
        self.lines_in = lines_in
        if lines_in is not None:
            # read in the linelist
            in_linelist = read_moog_linelist(lines_in)
            in_linelist = [tuple(x) for x in in_linelist[['wl','spe','ep','loggf']]]
        
            # go through each file and parse
            self.data[lines_in] = [in_linelist,None] # included for linelist but removed later
    
        # read in summary files
        self.file_col_starts = {}
        if isinstance(summary_filelist,str):
            try: file_summaryfilelist = open(summary_filelist,'r')
            except IOError:
                raise IOError("MOOG abfind summary list file doesn't exist : '"+file_summaryfilelist+"'")
            summary_files = file_summaryfilelist.readlines()
            file_summaryfilelist.close()
        elif isinstance(summary_filelist,(list,tuple,np.ndarray)):
            summary_files = summary_filelist
            
        else:
            raise ValueError("summary_filelist must be either a file with a "\
                              "list of MOOG abfind summary files or a list of "\
                              "those paths as strings")

        for i,sumfile in enumerate(summary_files):
            if not isinstance(sumfile,str):
                raise ValueError("recieved non string when expecting filepath")
            sumfile = sumfile.rstrip()
            if not os.path.isfile(sumfile):
                print("File does not exist or is not a file : '"+sumfile+"'")
                continue
            self.data[sumfile] = parse_abfind_summary_out(sumfile) # returns lines, header
            self.file_col_starts[sumfile] = i*self.width+self.left_width
    
        
        if not compile_linelist and lines_in is None:
            raise ValueError("If you don't want to compile a linelist"\
                             " from the summary files you must provide"\
                             "a linelist file in lines_in")

        # get the file based linelist
        if compile_linelist:
            files = self.data.keys()
        else:
            files = [lines_in]
        
        # create the linelist to compare against
        linelist_list = []
        for sumfile in files:
            dat = [tuple(x[:4]) for x in self.data[sumfile][0]]
            linelist_list += dat

        # TODO (optional): make the linelist a dictionary 
        #     with the keys = (wl,spe,ep,loggf) and values = fname
        #     Then you can keep track of the filename         
        linelist = set()
        for dat in linelist_list:
            linelist.add(dat) 
        
        if lines_in is not None:
            del self.data[lines_in]

        # define the linelist and species from the linelist        
        self.linelist = np.round(np.array(list(linelist)),10)
        self.species = np.sort(np.unique(np.round(self.linelist[:,1],5)))
        self._absorption_lines = None # subset of self.linelist based on a species
        
        # get desired annotation style
        if   notation in  ('l','logeps'): abnotation = 'Logeps'
        elif notation in ('b','bracket'): abnotation = '[X/Fe]'
        else: abnotation = 'abund'
        self.abnotation = abnotation
        
        # initilaize cells
        self.cells = SpreadsheetCells()

        # variables for tracking
        self.current_row = None
        self.next_row = None    
        self._current_fname = None

    def _check_init_values (self):
        if self.top_height < 6:
            raise ValueError("top_height must be >= 6")

        if self.width < 6: 
            raise ValueError("file column width must be >= 6")
        
        if self.left_width < 5:
            raise ValueError("left width must be >= 5")
        
        if not isinstance(self.batom,Batom):
            raise TypeError("batom must be a Batom instance")
        
    def __advance_next_row (self,num):
        self.next_row = max(self.next_row,self.current_row+num)

        
    def _top_left_block (self):
        if self.current_row is not None:
            row = self.current_row
            cells = self.cells 
        else:
            row = 0
            cells = SpreadsheetCells()
        
        cells[(row,0)] = ("CREATION DATE:",self.styles['default'])
        now = datetime.now()
        cells[(row,2)] = (now.ctime(),self.styles['default'])
    
        cells[(row+1,0)] = ("linelist file:",self.styles['default'])
        cells[(row+1,1)] = (self.lines_in,self.styles['default'])
    
        cells[(row+2,0)] = ("# Files:",self.styles['default'])
        cells[(row+2,1)] = (str(len(self.data)),self.styles['default'])
    
        titles = ("#lines=>","Wavelength","EP","Loggf","NOTES")
        for j in xrange(self.left_width):
            info = ''
            if j < len(titles):
                info = titles[j]
            row = self.top_height - 1
            cells[(row,j)] = (info,self.styles['lgray'])
            
        if self.current_row is None:
            return cells
        self.next_row = self.top_height+1
         
    def _top_right_block (self):
        if self.current_row is not None:
            row = self.current_row
            cells = self.cells 
        else:
            row = 0
            cells = SpreadsheetCells()
                
        # define headers and converters
        stellar = [('Teff','teff'),
                   ('Logg','logg'),
                   ('[Fe/H]','feh'),
                   ('Vt','vt')]
        titles = ("EP","LogGF","EW","LogRW",self.abnotation,"Del Avg")                   

        # go through the files and fill in
        for k,fname in enumerate(self.data):
            col_start = self.file_col_starts[fname]
            cells[(row,col_start)] = (fname,self.styles['default'])
            
            if k % 2 == 0 and k != 0: 
                header_style = self.styles['lgray']
            else:
                header_style = self.styles['dgray'] 
                   
            # go through all the cells in the column and fill them out
            for j in xrange(self.width):
                header_info = ''
                if j < len(titles): 
                    header_info= titles[j]
                
                info,param='',''
                col = col_start+j
                
                if j < len(stellar):
                    info = stellar[j][0]
                    param = str(self.data[fname][1][stellar[j][1]])
                    
                cells[(row+2,col)] = (info,self.styles['default'])
                cells[(row+3,col)] = (param,self.styles['default'])
                cells[(self.top_height-1,col)] = (header_info,header_style)        
                                
        if self.current_row is None:
            return cells

          
    def __write_line_data (self,match,match_i,row):
        
        # if the match was internal to the data then skip
        if match_i not in match:
            return row

        # get the index which isn't the match
        idx,idx2 = match
                
        # if the match is the idx then use the other one for the data
        if match_i == idx:
            idx = idx2
        
        # get the values to fill into the cells
        wl,_spe,ep,logg,ew,abund = self._spe_data[self._current_fname][idx]
        # !IndexError: shouldn't happen, there's an error in the matching
        diff = self._abund_stats[self._current_fname][0]-abund
        rew = np.log10((ew/(wl*1000.0)))        
        values = (ep,logg,ew,rew,abund,diff)
        
        # fill in the width of the data for a match
        for m in xrange(self.width):
            col = self.file_col_starts[self._current_fname]+m
            
            if m < len(values):
                value = values[m]
            else:
                value = ''
                
            self.cells[(row,col)] = (value,self.styles['default_float000'])
            
        # record how many lines per row
        cell_id = (row,0)
        _N,_ = self.cells.get(cell_id,(0,None))
        self.cells.overwrite(cell_id,(_N + 1,self.styles['default_int']))  
        return row+1    

    def __file_blocks (self,fname,abline):
        row = deepcopy(self.current_row)
        
        # is there any data for that file
        if len(self._spe_data[fname]) == 0: 
            self.__advance_next_row(1)
            return row
        
        # store the current file name for later use
        self._current_fname = fname

        # match_i is the index for the linedata
        match_i = len(self._spe_data[fname])
        match_data = np.vstack((self._spe_data[fname][:,[0,2,3]],abline))        
        matches = get_matches(match_data,self.tolerances)
        
        # if no matches are found
        if len(matches) == 0: 
            self.__advance_next_row(1)
            return row
        
        # number of matches removing the 
        for match in matches:
            row = self.__write_line_data(match,match_i,row)
        
        self.__advance_next_row(row-self.current_row)
         
    def __linelist_right_block (self,abline):
        wl,ep,loggf=abline
        values = (wl,ep,loggf)
        style = self.styles['default']
        for i in xrange(len(values)):
            col = 1+i
            self.cells[(self.current_row,col)] = (values[i],style)
            if i == 0:
                style = self.styles['default_float000']
             
    def __linelist_loop (self,abline):
        self.next_row = deepcopy(self.current_row)                
        self.__advance_next_row(1)
        # add number of lines information, update if necessary
        cell_id = (self.current_row,0)
        _N,_ = self.cells.get(cell_id,(0,None))
        self.cells.overwrite(cell_id,(_N,self.styles['default_int']))
        
        # add the linelist header information
        self.__linelist_right_block(abline)
        
        # now add the files
        for fname in self.data:
            self.__file_blocks(fname,abline)
        
        self.current_row = deepcopy(self.next_row)
                
    def __species_header_line (self,spe,spe_idx):
        
        self._spe_data = {}
        self._abund_stats = {}
        
        # get the species name
        ionz = int(round((spe - round(spe))*10))+1
        spe_name = self.batom[round(spe)][1]+" "+int_to_roman(ionz)
        
        # cells for an element block
        values = ("Element:",spe_name,"N=",np.count_nonzero(spe_idx))
        for i in xrange(self.left_width):
            if i < len(values):
                value = values[i]
            else:
                value = ''            
            self.cells[(self.current_row,i)] = (value,self.styles['yellow_int'])
        
        
        # write out the files        
        for i,fname in enumerate(self.data):            
            col_start = self.file_col_starts[fname]
 
            # select file data for that species
            file_data = self.data[fname][0]
            self._spe_data[fname] = file_data[file_data[:,1]==spe]
             
            # get stats
            abunds = self._spe_data[fname][:,5]
            abunds = abunds[np.logical_not(np.isnan(abunds))] 
            self._abund_stats[fname] = (np.mean(abunds),np.std(abunds),len(abunds))
                       
                       
            # write out the stats for this particular file
            stats = ("Mean,Std,N=",)+self._abund_stats[fname]
            style = self.styles['yellow_float00']
 
            for j in xrange(self.width):
                col = col_start+j
                if j < len(stats): 
                    value = stats[j]
                else:
                    value = ''
                    
                if j == len(stats)-1:
                    style = self.styles['yellow_int']

                self.cells[(self.current_row,col)] = (value,style)

        self.current_row += 1
                                   
    def _species_loop (self,spe):
        if self.current_row is None:
            raise ValueError("Must do this with the build loop so that the cells are correct")
           
        # get the absorption lines for this species (sorted by wavelength)
        spe_idx = self.linelist[:,1] == spe
        lines = self.linelist[spe_idx][:,[0,2,3]]
        self._absorption_lines = lines[lines[:,0].argsort()]
        
        # add the species header line to current line
        self.__species_header_line(spe,spe_idx)
        
        # now go through all the lines for that species
        for abline in self._absorption_lines:
            self.__linelist_loop(abline)
     
        # TODO: sum up all the values in self.cells for column 0 from the row just below the header
        #   to this current row
        #   output to (self.current_row,0)
        self.current_row += 1

    def build_cells (self):
        self._check_init_values()
        
        # set up the row tracking
        self.current_row = 0
        self.next_row = 0
        self.cells = SpreadsheetCells()
        
        # fill in the top blocks
        self._top_left_block()
        self._top_right_block()
        self.cells[(self.top_height,self.left_width-1)] = ("FREEZE HERE",self.styles['default'])

        self.current_row = self.top_height+1
        
        # fill in the species blocks
        Nleft = len(self.species)
        for i,spe in enumerate(self.species):
            self._species_loop(spe)
            Nleft = len(self.species)-i
            print("Species "+format(spe,'5.1f')+" <-- "+format(Nleft,"5")+" left to compute")
        
        # reset the row tracking
        self.current_row = None
        self.next_row = None
         
    def check_output_file (self,output_xls,clobber=True):
        """ check that the output file has ".xls" extension and doesn't exist """
        if not os.path.basename(output_xls).endswith('.xls'): 
            output_xls += '.xls'
    
        if os.path.isfile(output_xls) and not clobber:
            raise IOError("File exists '"+output_xls+"'")

        return output_xls

    def save (self,output_xls,clobber=True,cells=None):
        """ save out cells to ".xls" file """
          
        # check the output file and add ".xls" if needed      
        output_xls = self.check_output_file(output_xls, clobber)

        # open up book
        book=xlwt.Workbook(encoding="utf-8")
        sheet=book.add_sheet("Line_Comp")
        
        if cells is None:
            cells = self.cells
     
        if not isinstance(cells,SpreadsheetCells):
            raise TypeError("cells must be a SpreadsheetCells object")
             
        for cell_id in cells:
            row,col = cell_id
            contents,style = cells[cell_id]
            sheet.write(row,col,contents,style)
                
        book.save(output_xls)         

def ablines (summary_filelist, output_xls, lines_in = None, clobber=True, compile_linelist=False, wl_tol=0.02,ep_tol=0.01, loggf_tol=.01, notation='l'):    
    """
    Takes a list of MOOG abfind summary files and (optionally) compares to a linelist 
    to create a excell document for line-by-line comparisons
    
    This tool creates an excel spreadsheet for the purpose of lining up lines which are 
    shared across several MOOG abfind summary files. This allows for the visual comparison
    of lines between various summary files
    
    
    Parameters
    ----------
    summary_filelist : list
        Each value of the list contains the path to a MOOG abfind summary file
    output_xls : string
        This is the path to the output file. If '.xls' is not included as the 
        exstention then it will be appended
    lines_in : string or None
        A MOOG readable linelist (see the function read_moog_linelist). If not
        given then the comparison linelist will be built from the input
        summary files
    clobber : boolean
        If 'True' and the output_xls file exists then the output file will be
        overwritten
    compile_linelist : boolean
        If 'False' then the lines from the summary files will also be included 
        in the comparison linelist along with the lines_in. If 'False' then 
        only the lines_in will be used (NOTE if 'False' and lines_in=None then
        will raise a ValueError)
    wl_tol : float
        The tolerance to compare the wavelength with
    ep_tol : float
        The tolerance to compare the excitation potential with
    loggf_tol : float
        The tolerance to compare the oscillator strength with
    notation : ('l','logeps','b','bracket')
        This determines whether the output will be in logeps format or [X/Fe]
        
    Returns
    -------
    cells : SpreadsheetCells
        This is a subclass of a dictionary with keys = (row,column) and 
        values = (value,style) for creating a spreadsheet
    
    Raises
    ------
    ValueError : If compile_linelist==False and lines_in=None because
        there is nothing to compare with
    
    Notes
    -----
    __1)__ If a line appears twice in a summary file then it will be
        included twice in the output on separate lines
    

    Examples
    --------
    >>> cells = ablines("filelist","comparison.xls", lines_in="mylinelist.txt", 
                        compile_linelist=True, wl_tol=0, loggf_tol=0, ep_tol=0)
    >>>
    
    
    
    Modification History
    --------------------
    20 July, 2013 : Dylan Gregersen
        - revised from original using an Ablines class.
        
    """
    ablines_ = Ablines(summary_filelist, lines_in, 
                       compile_linelist=compile_linelist, 
                       wl_tol=wl_tol, ep_tol=ep_tol, loggf_tol=loggf_tol, notation=notation)
    ablines_.build_cells()
    ablines_.save(output_xls,clobber)
    return ablines_.cells

def display_abfind_summary_out (summary_out,notation='l',display=True):
    """
    Displays a summary of the information in the MOOG abfind summary out file
    
    The MOOG abfind driver outputs a summary file of the element abundance for 
    particular absorption lines. This calls a function to parse that file and 
    exctract data. Then calculates mean abundances (and stdev) from all the 
    lines for a particular species. The information is displayed to the 
    screen. 
    
    
    Parameters
    ----------
    summary_out : string
        Provides the path to a MOOG abfind summary file
    notation : 'l','logeps','b','bracket'
        The notation for the abundance to be output in. Logeps is noramlized 
        to H at 10**12 and bracket is normalized to the solar abundance 
        ratio to Fe [X/Fe]
    display : boolean
        If 'True' then this function will automatically display the abundances 
        to the screen. If 'False' then will just return the string for display
    
    Returns
    -------
    display_string : string
        The single string which is printed to the string (has return characters)
    
    Raises
    ------
    TypeError : If notation is not a string
    ValueError : If variable notation is not one of the acceptable values
        
    Notes
    -----
    __1)__ 
    
    
    
    Examples
    --------
    >>> display_string = display_abfind_summary_out("summary_out.txt",notation='b')
================================================================================

Abundances from : /full_path_to_file/summary_out.txt
    5500.0      4.000     -1.900      1.000

[Fe/H] =     -1.946

--------------------------------------------------------------------------------
         Z    element     [X/Fe]       sigma     N        RMS
--------------------------------------------------------------------------------
      11.0    Na I         0.301       0.475     2      0.336
      12.0    Mg I         0.092       0.072     5      0.032
      14.0    Si I         0.536       0.000     1      0.000
      20.0    Ca I         0.209       0.071    18      0.017
      21.1    Sc II       -0.115       0.123    13      0.034
      22.0    Ti I         0.071       0.073    18      0.017
      22.1    Ti II        0.236       0.138    17      0.034
      23.0    V I         -0.210       0.021     3      0.012
      24.0    Cr I        -0.022       0.414    40      0.065
      24.1    Cr II        0.123       0.123     6      0.050
      25.0    Mn I        -0.060       0.628    14      0.168
      26.0    Fe I        -0.025       0.107    82      0.012
      26.1    Fe II        0.033       0.123    19      0.028
      28.0    Ni I        -0.067       0.149    20      0.033
      30.0    Zn I        -0.104       0.010     2      0.007
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
AVERAGES==>
         Z    element     [X/Fe]       sigma     N        RMS
--------------------------------------------------------------------------------
      24.0    Cr           0.111       0.432     2      0.305
      26.0    Fe           0.000       0.163     2      0.115
      22.0    Ti           0.106       0.156     2      0.110
================================================================================
[Fe/H] =     -1.946

Abundances for  15 elements in file: 'summary_out.txt'

    >>>
    >>>
    
    """


    batom = Batom()

    notation_err = "notation must be 'l','logeps' or 'b','bracket'"
    if not isinstance(notation,str):
        raise TypeError(notation_err)
    
    logeps=True
    if  notation in ('l','logeps'):
        notation = 'Logeps'
    elif notation in ('b','bracket'):
        notation = '[X/Fe]'
        logeps=False 
    else:
        raise ValueError(notation_err)
    
    data,header = parse_abfind_summary_out(summary_out)
    species =  np.sort(np.unique(data[:,1]))
    # species_names = [batom[x][1] for x in species]
    
    abundances = abundance_totals(data)
    averages = {}
    
    for spe in species:
        element = int(round(spe))
        mask = np.round(abundances[:,0]) == element
        if np.count_nonzero(mask) <= 1:
            continue
        abunds = abundances[mask]
        M = np.sum(abunds[:,1]/(abunds[:,2]**2))/(np.sum(1.0/abunds[:,2]**2))
        ERR = np.sqrt(np.sum(abunds[:,2]**2))
        N = len(abunds)
        averages[element] = [spe,M,ERR,N]
 
    feh_abunds = abundances[np.round(abundances[:,0])==26][:,1]
    if len(feh_abunds)==0: 
        feh = np.nan
    else:
        if 26 in averages:
            feh = averages[26][1]
        else:   
            feh = np.mean(feh_abunds)
        feh -= batom[26.0][-1]
    
    if not logeps:
        if feh == np.nan:
            raise ValueError("Can't convert to [X/Fe] because no Fe abundances were found")
        abundances[:,1]= batom.to_xfe(abundances[:,0], abundances[:,1], feh)
        for element in averages:
            averages[element][1] = batom.to_xfe(element, averages[element][1], feh)
        
    lines = []
    lines.append("="*80)
    lines.append("")
    lines.append("Abundances from : "+os.path.abspath(summary_out))
    stellar = (header['teff'],header['logg'],header['feh'],header['vt'])
    lines.append("{0:10.1f} {1:10.3f} {2:10.3f} {3:10.3f}".format(*stellar))
    lines.append("")
    lines.append("[Fe/H] = "+format(feh,'10.3f'))
    lines.append("")
    lines.append("-"*80)
    # ep, abund sigma
    titles = ('Z','element',str(notation),"sigma","N","RMS")
    format_string = "{0:>10.1f}    {1:<7} {2:>10.3f}  {3:>10.3f} {4:>5} {5:>10.3f}"
    title_string =  "{0:>10}    {1:<7} {2:>10}  {3:>10} {4:>5} {5:>10}"
    
    lines.append(title_string.format(*titles))
    lines.append("-"*80) 
    for abund in abundances:
        spe,M,STD,N = abund
        rms = STD/np.sqrt(N)
        lines.append(format_string.format(spe,batom.species_name(spe),M,STD,int(N),rms))
        
    lines.append("- "*35)
    lines.append("AVERAGES==>")
    lines.append(title_string.format(*titles))
    lines.append("-"*80)
    for element in averages:
        _,M,STD,N = averages[element]
        rms = STD/np.sqrt(N)
        lines.append(format_string.format(element,batom.species_name(element),M,STD,int(N),rms))
        
    lines.append("="*80) 
    lines.append("[Fe/H] = "+format(feh,'10.3f'))
    lines.append("")
    N = len(species)
    lines.append("Abundances for "+format(N,"3")+" elements in file: '"+summary_out+"'")
    
    output_string = "\n".join(lines)+"\n"*2
    if display:
        print(output_string)
    return output_string


# ########################################################################### #
 
def run_ablines ():
    """
    Interactively asks the user for the inputs to the ablines function and then runs it
    see help(ablines) for more information
    
    This works:
    
    * provide filenames(s) of summary file(s)
    * provide input linelist (optional to give None)
    * provide the tolerances for the matching
    * give the notation for the abundance (logeps or [X/Fe] bracket)
    * provide an output file path
    
    then the program runs abget.ablines with those inputs
    
    """
    # get the input filelist of MOOG summary files
    enter_multi = yesno("Enter multiple MOOG abfind summary files (or give a file with list of them)? ","n")
    if enter_multi:
        summary_filelist = get_filename("Enter MOOG abfind summary files :\n",enter_multi=True,find_filename=True)
    else:
        summary_filelist = get_filename("Enter file with list of MOOG abfind summary files :\n",find_filename=True)[0]
    
    # raise error if none were given
    if summary_filelist is None: 
        raise ValueError("no files give for the summary_filelist")
    
    # Get the input linelist (if wanted)    
    lines_in = None
    if yesno("Compare against a linelist?","y"):
        lines_in, = get_filename("Enter a MOOG formatted linelist:\n",find_filename=True)
    
    
    # should the summary files lines be included
    if lines_in is None:
        compile_linelist=True
    else:
        compile_linelist = yesno("Compile comparison lines from the summary files as well as the linelist?","y")
    
    # get the tolerance input
    while True:
        tols = raw_input("Please give the tolerances for wl, ep, and loggf separated by spaces (default=> 0 0 0):\n ")
        if not tols:
            wl_tol,ep_tol,loggf_tol = 0.0,0.0,0.0
            break
        try: wl_tol,ep_tol,loggf_tol = [float(x or 0.0) for x in tols.split()]
        except TypeError:
            print("Please give the tolerances wl, ep, and loggf seporated by spaces or press enter for default")
    
    # what type of notation should be given
    if yesno("Give output in logeps notation ('no' gives [X/Fe])? ",'y'):
        notation = 'l'
    else:
        notation = 'b'

    # get the output filename
    output_xls, = get_filename("Give the output excel filename :",'w',find_filename=True)
    
    # run the ablines function        
    ablines(summary_filelist, output_xls, 
            lines_in = lines_in, clobber=True, 
            compile_linelist=compile_linelist, notation= notation,
            wl_tol=wl_tol,ep_tol=ep_tol, loggf_tol=loggf_tol)
    
def run_display_abfind_summary_out ():
    """
    Takes a MOOG summary file and give a useful output format using the 
    function display_abfind_summary_out
    
    This function works in this order:
    
    * provide the summary file to view
    * provide the notation to be used (logeps or [X/Fe] bracket)
    
    then runs function display_abfind_summary_out
    
    """
    # enter the filename
    summary_out, = get_filename("Give name of MOOG abfind summary file : ",find_filename=True)
    if summary_out is None:
        raise ValueError("No file given to read")
    
    choices = ('l','logeps','b','bracket')
    notation = get_choice(choices,question="What type of abundance notation? :",default=0)    
    notation = choices[notation[0]]
    
    _ = display_abfind_summary_out(summary_out,notation)

    


