import os

import numpy as np
import pandas as pd
from thimbles import resource_dir

def _load_lodders():
    lodderslines = open(os.path.join(resource_dir, "abundance_references", "lodders2010.txt")).readlines()
    lod_abs = {}
    for line in lodderslines[1:]:
        spl = line.split()
        if len(spl) > 0:
            #0  1   2         3    4          5               6
            #z, CI, CI_error, sun, sun_error, recommendation, recommendation_error
            vals = list(map(float, spl))
            lod_abs[(vals[0], 0)] = vals[5]
    return lod_abs

def _load_isotopes():
    atomic_number = {}
    isolines = open(os.path.join(resource_dir, "isotopes", "isotopic_weights.txt")).readlines()
    isolines = [line for line in isolines if (len(line) >  1) and (line[0] != "#")]
    weights = {}
    fracs = {}
    symbols = {}
    cur_z = None
    cur_mass_num = None
    cur_symbol = None
    for line in isolines:
        try:
            cur_z = int(line[:2])
            cur_symbol = line[4:6].strip()
        except ValueError:
            pass
        cur_mass_num = int(line[8:11])
        wstr = line[13:29].replace("(", "").replace(")", "")
        cur_weight = float(wstr)
        weights[(cur_z, cur_mass_num)] = cur_weight
        symbols[(cur_z, cur_mass_num)] = cur_symbol
        try:
            frac = float(line[32:43].replace("(", "").replace(")", ""))
        except:
            frac = 0.0
        fracs[(cur_z, cur_mass_num)] = frac
        try:
            mstand = float(line[46:61].replace("(", "").replace(")", ""))
            weights[(cur_z, 0)] = mstand
            fracs[(cur_z, 0)] = 1.0
            symbols[(cur_z, 0)] = cur_symbol
            atomic_number[cur_symbol] = cur_z
        except ValueError:
            pass
    return dict(symbol=symbols, weight=weights, fraction=fracs, atomic_number=atomic_number)

def _load_ptable():
    data_dict = _load_isotopes()
    atomic_number = data_dict.pop("atomic_number")
    data_dict["abundance"] = _load_lodders()
    #translate isotope fractions into isotopic abundances
    iso_fracs = data_dict["fraction"]
    for z, isotope in list(iso_fracs.keys()):
        if isotope == 0:
            continue
        else:
            frac = iso_fracs[(z, isotope)]
            tot_abund = data_dict["abundance"].get((z, 0))
            if tot_abund is None:
                tot_abund = -np.inf
            iso_abund = tot_abund + np.log10(frac)
            data_dict["abundance"][(z, isotope)] = iso_abund
    return pd.DataFrame(data=data_dict), atomic_number

ptable, atomic_number = _load_ptable()
atomic_symbol = {atomic_number[k]:k for k in atomic_number}

def z_to_symbol(z):
    try:
        z = int(z)
        if z < 100:
            return atomic_symbol[z]
        else:
            z1 = z//100
            z2 = z%100
            return "{}{}".format(
                atomic_symbol[z1],
                atomic_symbol[z2],
            )
    except KeyError:
        return None

def symbol_to_z(symbol):
    try:
        lsym = len(symbol)
        if lsym <= 2:
            return atomic_number[symbol]
        else:
            if symbol[-1].isupper():
                nback = 1
            else:
                nback = 2
            z1 = atomic_number[symbol[:-nback]]
            z2 = atomic_number[symbol[:lsym-nback]]
            z1, z2 = sorted(z1, z2)
            return z1*100+z2
    except KeyError:
        return None

    

#from thimbles.stellar_atmospheres import solar_abundance as ptable
