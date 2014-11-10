import numpy as np

def verify_line_list_columns(line_data, on_missing="inject"):
    """make sure that the passed in line data has all the 
    columns we expect for a linelist.
    
    expected columns:
    wv        : transition wavelength
    species   : species identifier e.g. 26.1 for Fe II  607 for CN etc etc
    ep        : transition lower level excitation potential
    loggf     : transition likelihood log(gf)
    Z         : number of protons of species
    ion       : ionization stage of species
    ew        : an associated equivalent width
    rad_damp  : radiative damping
    stark_damp: stark damping
    waals_damp: vanderwaals damping
    moog_damp : the MOOG line list damping parameter
    D0        : the dissociation energy for molecular lines
    line_id   : a unique integer for each transition
    """
    nan_cols =[ "wv", "species", "ep", "loggf", "ew", "rad_damp",
                "stark_damp", "waals_damp", "moog_damp", "D0"]
    for col_name in nan_cols:
        try:
            line_data[col_name]
        except KeyError as e:
            if on_missing == "inject":
                line_data[col_name] = np.repeat(np.nan, len(line_data))
            elif on_missing == "raise":
                raise e
            else:
                raise Exception("on_missing option not recognized must be either 'raise' or 'inject'")
    
    #special handling for Z and ion cols derive from species column
    try:
        line_data["Z"]
        line_data["ion"]
    except KeyError as e:
        if on_missing == "inject":
            species = line_data["species"]
            Z = np.array(species, dtype=int)
            ion = np.array((species-Z)*10, dtype=int)
            line_data["Z"] =  Z
            line_data["ion"] = ion
        elif on_missing == "raise":
            raise e
        else:
            raise Exception("on_missing option not recognized must be either 'raise' or 'inject'")
    
    #put the line_id column in if not there
    try:
        line_data["line_id"]
    except KeyError as e:
        if on_missing == "inject":
            line_data["line_id"] = np.arange(len(line_data))
        elif on_missing == "raise":
            raise e
        else:
            raise Exception("on_missing option not recognized must be either 'raise' or 'inject'")

class LineList(object):
    
    def __init__(self, line_data):
        verify_line_list_columns(line_data, on_missing="inject")
        self._data = line_data
    
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, index):
        return self._data[index]
    
    def __setitem__(self, index, value):
        self._data[index] = value
    
    def iloc(self):
        return self._data.iloc
    
    def ix(self):
        return self._data.ix
    
