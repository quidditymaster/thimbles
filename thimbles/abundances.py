import pandas as pd

from thimbles import ptable
from thimbles.sqlaimports import *
from sqlalchemy import Enum
from thimbles.thimblesdb import Base, ThimblesTable
from thimbles.modeling import Parameter

class Atom(ThimblesTable, Base):
    z = Column(Integer)
    isotope = Column(Integer)
    _weight = None
    _symbol = None
    _solar_ab = None
    
    def __init__(self, z, isotope=None):
        self.z = z
        if isotope is None:
            isotope = 0
        self.isotope = isotope
    
    def __repr__(self):
        return "{} isotope={}".format(self.symbol, self.isotope)
    
    @property
    def weight(self):
        if self._weight is None:
            self._weight = ptable.ix[(self.z, self.isotope), "weight"]
        return self._weight
    
    @property
    def solar_ab(self):
        if self._solar_ab is None:
            self._solar_ab = ptable.ix[(self.z, self.isotope), "abundance"]
        return self._solar_ab
    
    @property
    def symbol(self):
        if self._symbol is None:
            self._symbol =  ptable.ix[(self.z, self.isotope)].symbol
        return self._symbol

class Molecule(ThimblesTable, Base):
    _light_id = Column(Integer, ForeignKey("Atom._id"))
    light_atom = relationship("Atom", foreign_keys=_light_id)
    _heavy_id = Column(Integer, ForeignKey("Atom._id"))
    heavy_atom = relationship("Atom", foreign_keys=_heavy_id)
    d0 = Column(Float)
    
    _weight = None
    _symbol = None
    
    def __init__(self, z, d0=None, isotopes=None):
        self.d0 = d0
        if not isinstance(z, (list, tuple)):
            z = [z]
        
        if isotopes is None:
            isotopes = [None for _ in range(len(z))]
        
        for atom_idx in range(len(z)):
            if not isinstance(z[atom_idx], Atom):
                z[atom_idx] = Atom(z[atom_idx], isotopes[atom_idx])
        
        if len(z) == 1:
            self.light_atom = z[0]
        elif len(z) == 2:        
            if z[0].z > z[1].z:
                raise ValueError("Molecules must be specified with the light element first")
            self.light_atom = z[0]
            self.heavy_atom = z[1]
    
    @property
    def z(self):
        if self.monatomic:
            return self.light_atom.z
        else:
            return self.light_atom.z*100 + self.heavy_atom.z
    
    @property
    def monatomic(self):
        return self.heavy_atom is None
    
    @property
    def solar_ab(self):
        if self.monatomic:
            return self.light_atom.solar_ab
        else:
            return min(self.light_atom.solar_ab, self.heavy_atom.solar_ab)
    
    @property
    def weight(self):
        if self._weight is None:
            if self.monatomic:
                self._weight = self.light_atom.weight
            else:
                self._weight = self.light_atom.weight + self.heavy_atom.weight
        return self._weight
    
    @property
    def symbol(self):
        if self._symbol is None:
            self._symbol = "{}{}".format(self.light_atom.symbol, self.heavy_atom.symbol)
        return self._symbol
    
    

class Ion(ThimblesTable, Base):
    _species_id = Column(Integer, ForeignKey("Molecule._id"))
    species = relationship("Molecule")
    charge = Column(Integer)
    
    def __init__(self, species, charge):
        if not isinstance(species, Molecule):
            species = Molecule(species)
        self.species = species
        self.charge = charge
    
    @property
    def d0(self):
        return self.species.d0
    
    @property
    def weight(self):
        return self.species.weight


class Abundance(Parameter):
    _id = Column(Integer, ForeignKey("Parameter._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"Abundance"
    }
    _ion_id = Column(Integer, ForeignKey("Ion._id"))
    ion = relationship("Ion")
    _stellar_parameters_id = Column(Integer, ForeignKey("StellarParameters._id"))
    _value = Column(Float) #log(epsilon)
    
    @property
    def species(self):
        return self.ion.species

