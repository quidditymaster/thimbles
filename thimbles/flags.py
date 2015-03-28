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
        self.flag_bits = {}
        self.flag_masks = {}
        self.default_dict = {}
    
    def add_dimension(self, name, bit_index=None, default=False):
        """add a flag corresponding to the integer 2**bit_index
        """
        if not self.flag_masks.get(name) is None:
            raise ValueError("the name %s is already in this flag space" % name)
        bit_nums = self.flag_bits.values()
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
        for key in self.flag_masks.keys():
            out_dict[key] = bool(self.flag_masks[key] & flag_int)
        return out_dict
    
    def dict_to_int(self, flag_dict):
        """converts a flag dictionary into a corresponding integer in flag space
        """
        int_out = 0
        for key in flag_dict.keys():
            fmask = self.flag_masks.get(key)
            if fmask is None:
                raise ValueError("key %s does not belong to this flag space" % key)
            if flag_dict[key]:
                int_out += fmask
        return int_out
    
    def __getitem__(self, flag_name):
        return self.flag_masks[flag_name]
    
    @property
    def default_int(self):
        return self.dict_to_int(self.default_dict)

class Flags(object):
    
    def __init__(self, flag_space, flag_int=None):
        """
        """
        #the internal integer representation of our flag set
        self.flag_space = flag_space
        if flag_int is None:
            flag_int = self.flag_space.default_int
        #TODO: allow flags also to be initialized with boolean dicts
        self.flag_int = int(flag_int)
    
    def __getitem__(self, flag_name):
        return bool(self.flag_space[flag_name] & self.flag_int)
    
    def __setitem__(self, flag_name, new_flag_val):
        cur_flag_val = self[flag_name]
        #the current flag value and the set value match, do nothing
        if cur_flag_val and new_flag_val:
            return
        if (not cur_flag_val) and (not new_flag_val):
            return
        #the current flag value and new value don't match, toggle
        if cur_flag_val and (not new_flag_val):
            self.flag_int -= self.flag_space.flag_masks[flag_name]
        if (not cur_flag_val) and new_flag_val:
            self.flag_int += self.flag_space.flag_masks[flag_name]
    
    def asdict(self):
        return self.flag_space.int_to_dict(self.flag_int)
    
    def __repr__(self):
        return repr(self.asdict())

#NOTE: The order of these calls it is implicitly setting the bit_indexes
feature_flag_space = FlagSpace()
feature_flag_space.add_dimension("use", default=True)
feature_flag_space.add_dimension("bad_data")
feature_flag_space.add_dimension("bad_fit")
feature_flag_space.add_dimension("viewed")

class FeatureFlags(Flags, Base, ThimblesTable):
    flag_int= Column(Integer)
    
    def __init__(self, flag_int=None):
        super(FeatureFlags, self).__init__(feature_flag_space, flag_int)


spectrum_flag_space = FlagSpace()
spectrum_flag_space.add_dimension("normalized")
spectrum_flag_space.add_dimension("fluxed")
spectrum_flag_space.add_dimension("observed", default=True)
spectrum_flag_space.add_dimension("telluric")
spectrum_flag_space.add_dimension("sky")


class SpectrumFlags(Flags, ThimblesTable, Base):
    flag_int= Column(Integer)    
    
    def __init__(self, flag_int=None):
        #ThimblesTable.__init__(self)
        Flags.__init__(self, spectrum_flag_space, flag_int)
