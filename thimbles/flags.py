import numpy as np

from thimbles.thimblesdb import Base, ThimblesTable

from sqlalchemy import create_engine, ForeignKey
from sqlalchemy import Column, Date, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import relationship, backref

class FlagSpace(object):
    """enables efficient translation of dictionaries of true and false values
    paired with specific sets of keys to and from individual integers, for efficient
    storage and checking of many boolean values at once.
    """
    
    def __init__(self):
        #keep an internal dictionary of our flag names and corresponding integers
        self.flag_bits = {} #flag name --> bit number
        self.flag_masks = {}#flag name --> 2**(bit number)
        self.default_dict = {}#flag name --> default truth value
    
    def add_dimension(self, name, bit_index=None, default=False):
        """add a flag corresponding to the integer 2**bit_index
        """
        if not self.flag_masks.get(name) is None:
            raise ValueError("the name %s is already in this flag space" % name)
        bit_nums = list(self.flag_bits.values())
        if bit_index is None:
            bit_index = 0
            #assign the lowest currently unused bit number
            while bit_index in bit_nums:
                bit_index += 1
        if bit_index in bit_nums:
            raise ValueError("bit_index %d is already taken" % bit_index)
        self.flag_bits[name] = bit_index
        self.flag_masks[name] = 2**bit_index
        self.default_dict[name] = default
    
    def int_to_dict(self, flag_int):
        """decomposes a flag integer into a dictionary of the form {name:bool, ...}
        """
        out_dict = {}
        for key in list(self.flag_masks.keys()):
            out_dict[key] = bool(self.flag_masks[key] & flag_int)
        return out_dict
    
    def dict_to_int(self, flag_dict):
        """converts a flag dictionary into a corresponding integer in flag space
        """
        int_out = 0
        for key in list(flag_dict.keys()):
            fmask = self.flag_masks.get(key)
            if fmask is None:
                raise ValueError("key %s does not belong to this flag space" % key)
            if flag_dict[key]:
                int_out += fmask
        return int_out
    
    def __getitem__(self, flag_name):
        return self.flag_masks[flag_name]
    
    def __len__(self):
        return len(self.flag_masks)
    
    @property
    def flag_names(self):
        return self.flag_bits.keys()
    
    @property
    def default_int(self):
        return self.dict_to_int(self.default_dict)

class Flags(object):
    
    def __getitem__(self, flag_name):
        return bool(self.flag_space[flag_name] & self.flag_int)
    
    def __setitem__(self, flag_name, new_flag_val):
        if new_flag_val:
            self.flag_int |= self.flag_space[flag_name]
        else:
            self.flag_int -= self.flag_space[flag_name] & self.flag_int
    
    def update(self, **kwargs):
        for flag_name in kwargs:
            self[flag_name] = kwargs[flag_name]
    
    def asdict(self):
        return self.flag_space.int_to_dict(self.flag_int)
    
    def __repr__(self):
        return repr(self.asdict())

feature_flag_space = FlagSpace()
feature_flag_space.add_dimension("fiducial", bit_index=0, default=True)


class FeatureFlags(Flags, Base, ThimblesTable):
    flag_int= Column(Integer)
    flag_space = feature_flag_space
    
    def __init__(self, flag_int=None):
        if flag_int is None:
            flag_int = self.flag_space.default_int
        self.flag_int = flag_int


spectrum_flag_space = FlagSpace()
spectrum_flag_space.add_dimension("normalized", bit_index=0)
spectrum_flag_space.add_dimension("fluxed", bit_index=1)
spectrum_flag_space.add_dimension("observed", bit_index=2, default=True)
spectrum_flag_space.add_dimension("telluric", bit_index=3)
spectrum_flag_space.add_dimension("sky", bit_index=4)


class SpectrumFlags(Flags, ThimblesTable, Base):
    flag_int= Column(Integer)    
    flag_space = spectrum_flag_space
    
    def __init__(self, flag_int=None):
        if flag_int is None:
            flag_int = self.flag_space.default_int
        self.flag_int = flag_int
