import pandas as pd

from thimbles import ptable
from thimbles.sqlaimports import *
from sqlalchemy import Enum
from thimbles.thimblesdb import Base, ThimblesTable
from thimbles.modeling import Parameter

class Ion(ThimblesTable, Base):
    """representation of an Ion of either a single atom or a diatomic molecule.
    
    z represents proton number for an atom and light atom proton number *100 + heavier proton number for a molecule.
    isotope represents mass number for an atom and 1000*atom1.mass + atom2.mass for a molecule.
    """
    
    z = Column(Integer, nullable=False)
    isotope = Column(Integer)
    charge = Column(Integer)
    d0 = Column(Float)
    
    _weight = None
    _symbol = None
    _solar_ab = None
    
    def __init__(self, z, charge=0, isotope=0, d0=None):
        self.z = z
        self.charge=charge
        self.isotope = isotope
        self.d0 = d0
    
    @property
    def monatomic(self):
        return self.z < 100
    
    def split_z_iso(self):
        z1 = self.z//100
        z2 = self.z % 100
        iso1 = self.isotope//1000
        iso2 = self.isotope%1000
        return (z1, iso1), (z2, iso2)
    
    @property
    def solar_ab(self):
        if self._solar_ab is None:
            if self.monatomic:
                self._solar_ab = ptable.ix[(self.z, self.isotope), "abundance"]
            else:
                (z1, iso1), (z2, iso2) = self.split_z_iso()
                ab1 = ptable.ix[(z1, iso1), "abundance"]
                ab2 = ptable.ix[(z2, iso2), "abundance"]
                self._solar_ab = min(ab1, ab2)
        return self._solar_ab
    
    @property
    def weight(self):
        if self._weight is None:
            if self.monatomic:
                self._weight = ptable.ix[(self.z, self.isotope), "weight"]
            else:
                k1, k2 = self.split_z_iso()
                self._weight = ptable.ix[k1, "weight"] + ptable.ix[k2, "weight"]
        return self._weight
    
    @property
    def symbol(self):
        if self._symbol is None:
            if self.monatomic:
                symbol = ptable.ix[(self.z, self.isotope), "symbol"]
            else:
                k1, k2 = self.split_z_iso()
                symbol = "{}{}".format(*ptable.ix[[k1, k2], "symbol"])
            self._symbol = symbol
        return self._symbol


class Abundance(Parameter):
    _id = Column(Integer, ForeignKey("Parameter._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"Abundance"
    }
    _ion_id = Column(Integer, ForeignKey("Ion._id"))
    ion = relationship("Ion")
    _value = Column(Float) #log(epsilon)
    
    def __init__(self, ion, abund, bracket_notation=True):
        self._value = logeps
    
    @property
    def xonh(self):
        return self.value - ptable.ix[(self.z, self.isotope)]

