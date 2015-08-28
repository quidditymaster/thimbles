import pandas as pd

from thimbles import ptable
from thimbles.sqlaimports import *
from sqlalchemy import Enum
from thimbles.thimblesdb import Base, ThimblesTable
from thimbles.modeling import Parameter

from sqlalchemy.orm.collections import collection
from sqlalchemy.orm.collections import attribute_mapped_collection
from sqlalchemy.ext.associationproxy import association_proxy

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
    
    def __repr__(self):
        return "Ion:({} {}, iso{})".format(
            self.symbol,
            self.charge+1,
            self.isotope,
        )
    
    @property
    def monatomic(self):
        return self.z < 100
    
    def split_z_iso(self):
        z1 = self.z//100
        z2 = self.z - z1*100
        iso1 = self.isotope//100
        iso2 = self.isotope-iso1*100
        return (z1, iso1), (z2, iso2)
    
    @property
    def solar_ab(self):
        if self._solar_ab is None:
            if self.monatomic:
                self._solar_ab = ptable["abundance"].ix[(self.z, self.isotope)]
            else:
                (z1, iso1), (z2, iso2) = self.split_z_iso()
                ab1 = ptable["abundance"].ix[(z1, iso1)]
                ab2 = ptable["abundance"].ix[(z2, iso2)]
                self._solar_ab = min(ab1, ab2)
        return self._solar_ab
    
    @property
    def weight(self):
        if self._weight is None:
            if self.monatomic:
                self._weight = ptable["weight"].ix[(self.z, self.isotope)]
            else:
                k1, k2 = self.split_z_iso()
                self._weight = ptable["weight"].ix[k1] + ptable["weight"].ix[k2]
        return self._weight
    
    @property
    def symbol(self):
        if self._symbol is None:
            if self.monatomic:
                symbol = ptable["symbol"].ix[(self.z, 0)]
            else:
                (z1, i1), (z2, i2) = self.split_z_iso()
                symbol = "{}{}".format(*ptable["symbol"].ix[[(z1, 0), (z2, 0)]])
            self._symbol = symbol
        return self._symbol


class IonIndex(Base, ThimblesTable):
    _ion_id = Column(Integer, ForeignKey("Ion._id"))
    ion = relationship("Ion", foreign_keys=_ion_id)
    index = Column(Integer)
    _indexer_p_id = Column(Integer, ForeignKey("IonIndexerParameter._id"))
    
    def __init__(self, ion, index):
        self.ion = ion
        self.index = index


class IonIndexer(object):
    
    def __init__(self):
        self.ions = []
        self.ion_to_idx = {}
        self._ion_idx_list = []
    
    def extend_transitions(self, ions):
        for i in ions:
            i_i = len(self.ions)
            i_index = TransitionIndex(i, i_i)
            self._add_index(i_index)
    
    @collection.appender
    def _add_index(self, i_index):
        self._ion_idx_list.append(i_index)
        ion = i_index.ion
        idx = i_index.index
        n_current = len(self.ions)
        if n_current < idx+1:
            self.ions.extend([None for j in range(idx+1-n_current)])
        self.ions[idx] = ion
        self.ion_to_idx[ion] = idx
    
    @collection.remover
    def _remove_index(self, t_index):
        ion = t_index.ion
        idx = t_index.index
        self._ion_idx_list.remove(t_index)
        self.ions[i] = None
        self.ion_to_idx.pop(t)
    
    @collection.iterator
    def _iter_indexes(self):
        for t_idx in self._ion_idx_list:
            yield t_idx
    
    def __getitem__(self, index):
        if isinstance(index, Ion):
            return self.ion_to_idx[index]
        else:
            return self.idx_to_ion[index]
    
    def __len__(self):
        return len(self.ions)


class IonIndexerParameter(Parameter):
    _id = Column(Integer, ForeignKey("Parameter._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"IonIndexerParameter",
    }
    _value = relationship(
        "IonIndex", 
        collection_class=IonIndexer,
    )
    
    def __init__(self, ions):
        self._value.extend_ions(ions)

class IonMappedFloat(Base, ThimblesTable):
    _ion_id = Column(Integer, ForeignKey("Ion._id"))
    ion = relationship("Ion", foreign_keys=_ion_id)
    _mapper_parameter_id = Column(Integer, ForeignKey("IonMappedParameter._id"))
    value = Column(Float)
    
    def __init__(self, ion, value):
        self.ion = ion
        self.value = value


class IonMappedParameter(Parameter):
    _id = Column(Integer, ForeignKey("Parameter._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"IonMappedParameter",
    }
    _ion_map = relationship(
        "IonMappedFloat",
        collection_class=attribute_mapped_collection("ion")
    )
    _value = association_proxy("_ion_map", "value")
    
    def __init__(self, mapped_values=None):
        if mapped_values is None:
            mapped_values = {}
        self._value = mapped_values


