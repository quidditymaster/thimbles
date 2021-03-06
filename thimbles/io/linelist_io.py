from datetime import datetime 
import re
import numpy as np
import pandas as pd
import thimbles as tmb
from thimbles.tasks import task
from thimbles.io.moog_io import read_moog_linelist
from thimbles.io.moog_io import write_moog_linelist
from thimbles.transitions import Transition
from thimbles import ptable, atomic_number

def float_or_nan(val):
    try:
        return float(val)
    except ValueError:
        return np.nan

def read_vald_linelist(fname, ion_dict=None):
    file = open(fname)
    lines = file.readlines()
    file.close()
    input_re = re.compile(r"'([A-Z][a-z]{0,1})([A-Z][a-z]{0,1}){0,1} {1,4}([12])'")
    ldat = []
    
    if ion_dict is None:
        ion_dict = {}
    for line in lines:
        m = input_re.match(line)
        if m is None:
            continue
        species1, species2, ion_number = m.groups()
        charge = int(ion_number) - 1
        if species2 is None:
            proton_number = atomic_number[species1]
        else:
            proton1, proton2 = sorted([atomic_number[species1], atomic_number[species2]])
            proton_number = 100*proton1+proton2
        spl = line.split(",")
        wv, loggf, elow, jlo, eup, jup = list(map(float, spl[1:7]))
        l_lande, u_lande, m_lande = list(map(float_or_nan, spl[7:10]))
        rad_damp, stark_damp, waals_damp = list(map(float_or_nan, spl[10:13]))
        cur_ion = ion_dict.get((proton_number, charge))
        if cur_ion is None:
            cur_ion = tmb.abundances.Ion(z=proton_number, charge=charge)
            ion_dict[(proton_number, charge)] = cur_ion
        trans = Transition(
            wv=wv, 
            ion=cur_ion,
            ep=elow,
            loggf=loggf,
            damp=dict(stark=stark_damp, waals=waals_damp, rad=rad_damp),
        )
        ldat.append(trans)
    return ldat


@task(
    result_name="transitions",
    sub_kwargs=dict(
        fname=dict(editor_style="file"),
        file_type=dict(),
    )
)
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
    else:
        raise ValueError("file_type not understood")

