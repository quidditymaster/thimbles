from thimbles.sqlaimports import *
from thimbles.thimblesdb import Base, ThimblesTable
from thimbles.modeling import Parameter, FloatParameter
from thimbles.abundances import Abundance
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
            teff=500.0, 
            logg=3.0, 
            metalicity=0.0,
            vmicro=2.0, 
            mass=1.0,
            info=None
    ):
        Source.__init__(self, name=name, ra=ra, dec=dec, info=info)
        for pname, param in [
                ("teff", teff),
                ("logg", logg),
                ("metalicity", metalicity),
                ("vmicro", vmicro),
                ("mass", mass),
        ]:
            if not isinstance(param, Parameter):
                param = FloatParameter(param)
            self.add_parameter(pname, param)
    
    def __repr__(self):
        return "Star: {} ".format(self.name)
