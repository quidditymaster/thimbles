from thimbles.sqlaimports import *
from thimbles.thimblesdb import Base, ThimblesTable
from thimbles.modeling import Parameter
from thimbles.abundances import Abundance
from thimbles.sources import Source

class TeffParameter(Parameter):
    _id = Column(Integer, ForeignKey("Parameter._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"TeffParameter",
    }
    _value = Column(Float)
    
    def __init__(self, value):
        self._value = value

class LoggParameter(Parameter):
    _id = Column(Integer, ForeignKey("Parameter._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"LoggParameter",
    }
    _value = Column(Float)
    
    def __init__(self, value):
        self._value = value

class MetalicityParameter(Parameter):
    _id = Column(Integer, ForeignKey("Parameter._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"MetalicityParameter",
    }
    _value = Column(Float)
    
    def __init__(self, value):
        self._value = value

class VmicroParameter(Parameter):
    _id = Column(Integer, ForeignKey("Parameter._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"VmicroParameter",
    }
    _value = Column(Float) #in kilometers per second
    
    def __init__(self, value):
        self._value = value

class MassParameter(Parameter):
    _id = Column(Integer, ForeignKey("Parameter._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"MassParameter",
    }
    _value = Column(Float) #in solar masses
    
    def __init__(self, value):
        self._value = value

class Star(Source):
    _id = Column(Integer, ForeignKey("Source._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"Star",
    }
    _stellar_parameters_id = Column(Integer, ForeignKey("StellarParameters._id"))
    stellar_parameters = relationship("StellarParameters", backref="star")

class StellarParameters(ThimblesTable, Base):
    _teff_id = Column(Integer, ForeignKey("TeffParameter._id"))
    teff_p = relationship("TeffParameter", foreign_keys=_teff_id)
    _logg_id = Column(Integer, ForeignKey("LoggParameter._id"))
    logg_p = relationship("LoggParameter", foreign_keys=_logg_id)
    _metalicity_id = Column(Integer, ForeignKey("MetalicityParameter._id"))
    metalicity_p = relationship("MetalicityParameter", foreign_keys=_metalicity_id)
    _vmicro_id = Column(Integer, ForeignKey("VmicroParameter._id"))
    vmicro_p = relationship("VmicroParameter", foreign_keys=_vmicro_id)
    _mass_id = Column(Integer, ForeignKey("MassParameter._id"))
    mass_p = relationship("MassParameter", foreign_keys=_mass_id)
    abundances = relationship("Abundance")
    
    def __init__(self, 
                 teff, 
                 logg, 
                 metalicity, 
                 vmicro=2.0, 
                 mass=1.0,
                 abundances=None,
                 star=None
    ):
        if not isinstance(teff, TeffParameter):
            teff = TeffParameter(teff)
        if not isinstance(logg, LoggParameter):
            logg = LoggParameter(logg)
        if not isinstance(vmicro, VmicroParameter):
            vmicro = VmicroParameter(vmicro)
        if not isinstance(metalicity, MetalicityParameter):
            metalicity = MetalicityParameter(metalicity)
        if not isinstance(mass, MassParameter):
            mass = MassParameter(mass)
        self.teff_p = teff
        self.logg_p = logg
        self.metalicity_p = metalicity
        self.vmicro_p = vmicro        
        self.mass_p = mass
        
        if abundances is None:
            abundances = []
        self.abundances = abundances
        
        self.star = star
        
    @property
    def teff(self):
        return self.teff_p.value
    
    @teff.setter
    def teff(self, value):
        self.teff_p.value
    
    @property
    def theta(self):
        return 5040.0/self.teff
    
    @theta.setter
    def theta(self, value):
        self.teff = 5040.0/value
    
    @property
    def logg(self):
        return self.logg_p.value
    
    @logg.setter
    def logg(self, value):
        self.logg_p.value = value
    
    @property
    def metalicity(self):
        return self.metalicity_p.value
    
    @metalicity.setter
    def metalicity(self, value):
        self.metalicity_p.value = value
    
    @property
    def vmicro(self):
        return self.vmicro_p.value
    
    @vmicro.setter
    def vmicro(self, value):
        self.vmicro_p.value = value
    
    @property
    def mass(self):
        return self.mass_p
    
    @mass.setter
    def mass(self, value):
        self.mass_p.value = value

