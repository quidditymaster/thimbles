from thimbles.sqlaimports import *
from thimbles.thimblesdb import Base, ThimblesTable
from thimbles.modeling import Parameter, FloatParameter
from thimbles.sources import Source


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
