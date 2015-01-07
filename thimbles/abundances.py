import pandas as pd
from thimbles.ptable import ptable

from thimbles.sqlaimports import *
from sqlalchemy import Enum
from thimbles.thimblesdb import Base, ThimblesTable
from thimbles.modeling import Parameter

class Species(ThimblesTable, Base):
    species_class = Column(Enum("Atom Molecule".split()))
    symbol = Column(String)
    __mapper_args__{
        "polymorphic_identity":"Species",
        "polymorphic_on":species_class,
    }

class Atom(Species):
    _id = Column(Integer, ForeignKey("Species._id"), primary_key=True)
    __mapper_args__{
        "polymorphic_identity":"Molecule",
    }    
    Z = Column(Integer)
    weight = Column(Float)

class Molecule(Species):
    _id = Column(Integer, ForeignKey("Species._id"), primary_key=True)
    __mapper_args__{
        "polymorphic_identity":"Molecule",
    }
    _light_id = Column(Integer, ForeignKey("Species._id"))
    light_atom = relationship("Atom", foreign_keys=_light_id)
    _heavy_id = Column(Integer, ForeignKey("Species._id"))
    heavy_atom = relationship("Atom", foreign_keys=_heavy_id)
    
    @property
    def weight(self):
        return self.heavy_atom.weight + self.light_atom.weight

class Ion(ThimblesTable, Base):
    _atom_id = Column(Integer, ForeignKey("Atom._id"))
    atom = relationship("Atom")
    ion = Column(Integer)

class Abundance(Parameter):
    _id = Column(Integer, ForeignKey("Parameter._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"Abundance"
    }
    _ion_id = Column(Integer, ForeignKey("Ion._id"))
    ion = relationship("Ion")
    _stellar_parameters_id = Column(Integer, ForeignKey("StellarParameters._id"))
    _value = Column(Float) #log(epsilon)

#solar_abundances = AbundanceVector(ptable.logeps)
