from datetime import datetime 
import numpy as np
import pandas as pd

def read_moog_linelist (fname,formatted=True, output_pandas=False, defaults=None,convert_gf=False):
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
        1) wavelength ('wv')
        2) species ID ('species')
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
       12, Mar 2014: Tim Anderton
    
    """
    
    vwdamp = np.nan
    d0 = 0.0
    ew = np.nan
    # check default values for columns
    if defaults is None:
        defaults = {}
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
              ('species',float),
              ('ep',float),
              ('loggf',float),
              ('vwdamp',float),
              ('d0',float),
              ('ew',float),
              ('notes','a200')]
    
    data = np.rec.array(data,dtype=dtypes)
    if convert_gf: data['loggf'] = np.log10(data['loggf'])
    return data


def write_moog_linelist(filename, line_data, comment=None):
    """
    write a moog readable linelist
    """
    out_file = open(filename,'w')
    
    # write the header line if desired
    if comment is None:
        comment = "#"+str(datetime.today())
    out_file.write(str(comment).rstrip()+"\n")
    
    fmt_string = "% 10.3f% 10.5f% 10.2f% 10.2f"
    for line_idx in range(len(line_data)):
        cline = line_data[line_idx]
        wv,species,ep,loggf = cline["wv"], cline["species"], cline["ep"], cline["loggf"]
        out_str = fmt_string % (wv, species, ep, loggf)
        for v_str in "vwdamp d0 ew".split():
            if cline[v_str] != np.nan:
                out_str += 10*" "
        
        out_str += "\n"
        out_file.write(out_str)    
        out_file.close()