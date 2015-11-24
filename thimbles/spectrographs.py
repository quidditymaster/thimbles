import numpy as np
import scipy.sparse

from .modeling.associations import HasParameterContext, NamedParameters, ParameterAliasMixin
from .utils.partitioning import partitioned_polynomial_model
from .utils import piecewise_polynomial as ppol 
from .thimblesdb import Base, ThimblesTable
from .modeling import Model, Parameter
from .sqlaimports import *
from .modeling import PixelPolynomialModel
import thimbles as tmb


class SamplingModel(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"SamplingModel",
    }
    
    def __init__(
            self, 
            output_p, 
            input_wvs, 
            input_lsf,
            output_wvs,
            output_lsf,
    ):
        self.output_p = output_p
        self.add_parameter("input_wvs", input_wvs)
        self.add_parameter("input_lsf", input_lsf)
        self.add_parameter("output_wvs", output_wvs)
        self.add_parameter("output_lsf", output_lsf)  
    
    def __call__(self, vprep=None):
        vdict = self.get_vdict(vprep)
        x_in = vdict[self.inputs["input_wvs"]].coordinates
        x_out = vdict[self.inputs["output_wvs"]].coordinates
        lsf_in = vdict[self.inputs["input_lsf"]]
        lsf_out = vdict[self.inputs["output_lsf"]]
        return tmb.resampling.resampling_matrix(x_in, x_out, lsf_in, lsf_out)


class Aperture(ThimblesTable, Base):
    name = Column(String)
    info = Column(PickleType)
    
    def __init__(self, name, info=None):
        self.name = name
        if info is None:
            info = {}
        self.info = info

    def __repr__(self):
        return "Aperture: {}".format(self.name)

class Order(ThimblesTable, Base):
    number = Column(Integer)
    
    def __init__(self, number):
        self.number = number

    def __repr__(self):
        return "Order: {}".format(self.number)

class Chip(ThimblesTable, Base):
    name = Column(String)
    info = Column(PickleType)
    
    def __init__(self, name, info=None):
        self.name = name
        if info is None:
            info = {}
        self.info = info
    
    def __repr__(self):
        return "Chip: {}".format(self.name)

