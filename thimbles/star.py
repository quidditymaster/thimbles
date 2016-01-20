from .sqlaimports import *
from .thimblesdb import Base, ThimblesTable
from .modeling import Parameter, FloatParameter
from .sources import Source

class Star(Source):
    _id = Column(Integer, ForeignKey("Source._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"Star",
    }
    
    def __init__(
            self, 
            name=None, 
            ra=None, 
            dec=None, 
            teff=5000.0, 
            logg=3.0, 
            metalicity=0.0,
            vmicro=2.0, 
            vmacro=1.0,
            vsini=5.0,
            ldark=0.6,
            mass=1.0,
            age=5.0,
            info=None
    ):
        Source.__init__(self, name=name, ra=ra, dec=dec, info=info)
        for pname, param in [
                ("teff", teff),
                ("logg", logg),
                ("metalicity", metalicity),
                ("vmicro", vmicro),
                ("vmacro", vmacro),
                ("mass", mass),
                ("age", age),
                ("vsini", vsini),
                ("ldark", ldark),
        ]:
            if not isinstance(param, Parameter):
                param = FloatParameter(param)
            self.add_parameter(pname, param)
    
    def __repr__(self):
        return "Star: {} ".format(self.name)
    
    @property
    def teff(self):
        return self["teff"].value
    
    @teff.setter
    def teff(self, value):
        self["teff"].set(value)
    
    @property
    def logg(self):
        return self["logg"].value
    
    @logg.setter
    def logg(self, value):
        self["logg"].set(value)
    
    @property
    def metalicity(self):
        return self["metalicity"].value
    
    @metalicity.setter
    def metalicity(self, value):
        self["metalicity"].set(value)
    
    @property
    def vmicro(self):
        return self["vmicro"].value
    
    @vmicro.setter
    def vmicro(self, value):
        self["vmicro"].set(value)
    
    @property
    def vmacro(self):
        return self["vmacro"].value
    
    @vmacro.setter
    def vmacro(self, value):
        self["vmacro"].set(value)
    
    @property
    def vsini(self):
        return self["vsini"].value
    
    @vsini.setter
    def vsini(self, value):
        self["vsini"].set(value)
    
    @property
    def ldark(self):
        return self["ldark"].value

    @ldark.setter
    def ldark(self, value):
        self["ldark"].set(value)
    
    @property
    def mass(self):
        return self["mass"].value
    
    @mass.setter
    def mass(self, value):
        self["mass"].set(value)
    
    @property
    def age(self):
        return self["age"].value
    
    @age.setter
    def age(self, value):
        self["age"].set(value)

