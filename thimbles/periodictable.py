import os

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
            vals = map(float, spl)
            lod_abs[(vals[0], 0)] = vals[5]
    return lod_abs

def _load_isotopes():
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
            cur_symbol = line[4:6]
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
        except ValueError:
            pass
    return dict(symbol=symbols, weight=weights, fraction=fracs)

def _load_ptable():
    data_dict = _load_isotopes()
    data_dict["abundance"] = _load_lodders()
    return pd.DataFrame(data=data_dict)

ptable = _load_ptable()
#from thimbles.stellar_atmospheres import solar_abundance as ptable
