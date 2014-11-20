from datetime import datetime 
import re
import numpy as np
import pandas as pd
import thimbles as tmb
from thimbles.tasks import task
from thimbles.linelists import LineList
from thimbles.io.moog_io import read_moog_linelist
from thimbles.io.moog_io import write_moog_linelist

from thimbles.stellar_atmospheres import solar_abundance as ptable

def float_or_nan(val):
    try:
        return float(val)
    except ValueError:
        return np.nan

def read_vald_linelist(fname):
    lines = open(fname).readlines()
    input_re = re.compile("'[A-Z][a-z] [12]', ")
    col_names = "wv species ep loggf D0 stark_damp rad_damp waals_damp".split()
    ldat = {cname:[] for cname in col_names}
    for line in lines:
        m = input_re.match(line)
        if m is None:
            continue
        spl = line.rstrip().split(",")
        species_name, ion_number = spl[0].replace("'", "").split()
        ion_number = int(ion_number) - 1
        proton_number = ptable[species_name]["z"]
        wv, loggf, elow, jlo, eup, jup = map(float, spl[1:7])
        l_lande, u_lande, m_lande = map(float_or_nan, spl[8:11])
        rad_damp, stark_damp, waals_damp = map(float_or_nan, spl[12:15])
        ldat["wv"].append(wv)
        ldat["species"].append(proton_number+ion_number*0.1)
        ldat["ep"].append(elow)
        ldat["loggf"].append(loggf)
        ldat["rad_damp"].append(rad_damp)
        ldat["stark_damp"].append(stark_damp)
        ldat["waals_damp"].append(waals_damp)
        #and the parameters not present
        ldat["moog_damp"].append(np.nan)
        ldat["D0"].append(np.nan)
    ldf = pd.DataFrame(data=ldat)
    return LineList(ldf)

@task(result_name="line_data")
def read_linelist(fname, file_type="detect"):
    """
    fname: string
      path to linelist file 
    file_type: string
      'detect' attempt to detect linelist type from extensions
      'moog' moog type linelist
      'vald' vald long format
      'hdf5' the efficient hdf5 format
    """
    if file_type is "detect":
        if ".ln" in fname:
            file_type = "moog"
        elif ".vald" in fname:
            file_type = "vald"
        elif ".h5" in fname:
            file_type = "hdf5"
    if file_type == "hdf5":
        return LineList(pd.read_hdf(fname, "ldat"))
    elif file_type.lower() == "moog":
        return read_moog_linelist(fname)
    elif file_type == "vald":
        return read_vald_linelist(fname)
    else:
        raise ValueError("file_type {} not understood".format(file_type))

@task()
def write_linelist(fname, line_data, file_type="moog", subkwargs=None):
    """write out a linelist"""
    if subkwargs is None:
        subkwargs = {}
    if file_type == "hdf5":
        line_data.to_hdf(fname, "ldat")
    elif file_type == "moog":
        write_moog_linelist(fname, line_data, **subkwargs) 
    raise ValueError("file_type not understood")

