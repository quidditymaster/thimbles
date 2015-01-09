from thimbles.sqlaimports import *
from thimbles.thimblesdb import Base, ThimblesTable
from thimbles.modeling import Parameter, Model
from thimbles.abundances import Ion

class Damping(ThimblesTable, Base):
    stark = Column(Float)
    waals = Column(Float)
    rad   = Column(Float)
    empirical  = Column(Float)
    
    def __init__(self, stark=None, waals=None, rad=None, empirical=None):
        self.stark = stark
        self.waals = waals
        self.rad = rad
        self.empirical = empirical

class Transition(ThimblesTable, Base):
    wv = Column(Float)
    _ion_id = Column(Integer, ForeignKey("Ion._id"))
    ion = relationship("Ion")
    ep = Column(Float)
    loggf = Column(Float)
    _damp_id = Column(Integer, ForeignKey("Damping._id"))
    damp = relationship(Damping)
    
    def __init__(self, wv, ion, ep, loggf, damp=None):
        self.wv = wv
        self.ep = ep
        self.loggf = loggf
        
        if not isinstance(ion, Ion):
            if isinstance(ion, tuple):
                speciesnum, charge = ion
            else:
                speciesnum = int(ion)
                charge = int((ion-speciesnum)*10)
            ion = Ion(speciesnum, charge)
        
        self.ion = ion
        if damp is None:
            damp = Damping()
        elif isinstance(damp, dict):
            damp = Damping(**damp)
        self.damp = damp

