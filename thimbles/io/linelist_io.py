from datetime import datetime 
import re
import numpy as np
import pandas as pd
import thimbles as tmb

from thimbles.stellar_atmospheres import solar_abundance as ptable

def float_or_nan(val):
    try:
        return float(val)
    except ValueError:
        return np.nan

def read_vald_linelist(fname):
    """read a vald long format linelist
    
    parameters
    ------------
    fname: string
        file name of the target file
    
    returns
    -------------
    pandas.DataFrame of line data
    """
    lines = open(fname, "r").readlines()
    ldat = {"wv":[], "species":[], "z":[], "ion":[], 
            "ep":[], "loggf":[], "solar_ab":[],
            "rad_damp":[], "stark_damp":[], "waals_damp":[]
            }
    ldat = pd.DataFrame(data=ldat)
    return ldat

def read_linelist(fname, file_type="moog"):
    lines = open(fname).readlines()
    ldat = {"wv":[], "species":[], "Z":[], "ion":[], 
            "ep":[], "loggf":[], "ew":[],
            "rad_damp":[], "stark_damp":[], "waals_damp":[],
            "moog_damp":[], "D0":[],
            }
    if file_type.lower() == "moog":
        for line in lines:
            try:
                moog_cols = [line[i*10:(i+1)*10].strip() for i in range(7)]
                wv = float(moog_cols[0])
                species = float(moog_cols[1])
                sp_split = moog_cols[1].split(".")
                z = int(sp_split[0])
                ion = int(sp_split[1][0])
                #A=sp_split[1][1:]
                ep = float(moog_cols[2])
                loggf = float(moog_cols[3])
                if moog_cols[4] != "":
                    moog_damp = float(moog_cols[4])
                else:
                    moog_damp = np.nan
                if moog_cols[5] != "":
                    d0 = float(moog_cols[5])
                else:
                    d0 = np.nan
                if moog_cols[6]:
                    ew = float(moog_cols[6])
                else:
                    ew = 0
                z = int(species)
                ion = int(10*(species-z))
                
                rad_damp = np.nan
                stark_damp = np.nan
                waals_damp = np.nan
            except ValueError as e:
                print e
                continue
            ldat["wv"].append(wv)
            ldat["species"].append(z+(ion-1)*0.1)
            ldat["Z"].append(z)
            #TODO: add a nucleon number column "A"
            ldat["ion"].append(ion)
            ldat["ep"].append(ep)
            ldat["loggf"].append(loggf)
            ldat["ew"]=ew
            ldat["rad_damp"].append(rad_damp)
            ldat["stark_damp"].append(stark_damp)
            ldat["waals_damp"].append(waals_damp)
            ldat["moog_damp"].append(moog_damp)
            ldat["D0"] = d0
    elif file_type == "vald":
        input_re = re.compile("'[A-Z][a-z] [12]', ")
        for line in lines:
            m = input_re.match(line)
            if m is None:
                continue
            spl = line.rstrip().split(",")
            species_name, ion_number = spl[0].replace("'", "").split()
            ion_number = int(ion_number)
            proton_number = ptable[species_name]["z"]
            #species_id = proton_number + 0.1*(int(ion_number)-1)
            wv, loggf, elow, jlo, eup, jup = map(float, spl[1:7])
            l_lande, u_lande, m_lande = map(float_or_nan, spl[8:11])
            rad_damp, stark_damp, waals_damp = map(float_or_nan, spl[12:15])
            ldat["wv"].append(wv)
            ldat["species"].append(proton_number+(ion_number-1)*0.1)
            ldat["Z"].append(proton_number)
            #TODO: add a nucleon number column "A"
            ldat["ion"].append(ion_number)
            ldat["ep"].append(elow)
            ldat["loggf"].append(loggf)
            ldat["rad_damp"].append(rad_damp)
            ldat["stark_damp"].append(stark_damp)
            ldat["waals_damp"].append(waals_damp)
            #and the parameters not present
            ldat["moog_damp"].append(np.nan)
            ldat["ew"].append(0.0)
            ldat["D0"].append(np.nan)
    return pd.DataFrame(data=ldat)

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
    (numpy recarray) Returns a numpy record asarray with columns associated to the short names for the columns (see NOTES 1 and 2)
    
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
    ew = 0.0
    # check default values for columns
    if defaults is None:
        defaults = {}
    if 'vwdamp' in defaults: vwdamp = defaults['vwdamp']
    if 'd0' in defaults: d0 = defaults['d0']
    if 'ew' in defaults: ew = defaults['ew']
    
    
    f = open(fname)
    lines_in = f.readlines()
    f.close()
    
    lines = []
    try:
        float(lines_in[0][:10])
        float(lines_in[0][10:20])
        lines.append(lines_in[0])
    except ValueError as e:
        pass
              
    # guess formatting is formatted is None
    for line in lines_in[1:]:
        if line.strip() == '' or line.strip()[0] == '#': 
            continue
        lines.append(line)
    
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
        
    dtypes = [('wv',float),
              ('species',float),
              ('ep',float),
              ('loggf',float),
              ('vwdamp',float),
              ('d0',float),
              ('ew',float),
              ('notes','a200')]
    
    data = np.rec.asarray(data,dtype=dtypes)
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

def write_moog_from_features(filename, features):
    llout = tmb.stellar_atmospheres.utils.moog_utils.write_moog_lines_in(filename)
    for feat in features:
        wv=feat.wv
        spe=feat.species
        loggf = feat.loggf
        ep = feat.ep
        ew = 1000*feat.eq_width
        if feat.flags["use"]:
            llout.add_line(wv, spe, ep, loggf, ew=ew, comment=feat.note)
    llout.close()
