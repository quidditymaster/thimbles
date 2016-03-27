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
    
    def __call__(self, override=None):
        vdict = self.get_vdict(override)
        x_in = vdict[self.inputs["input_wvs"]].coordinates
        x_out = vdict[self.inputs["output_wvs"]].coordinates
        lsf_in = vdict[self.inputs["input_lsf"]]
        lsf_out = vdict[self.inputs["output_lsf"]]
        return tmb.resampling.resampling_matrix(x_in, x_out, lsf_in, lsf_out)

class ApertureAlias(ParameterAliasMixin, ThimblesTable, Base):
    _context_id = Column(Integer, ForeignKey("Aperture._id"))
    context = relationship("Aperture", foreign_keys=_context_id, back_populates="context")

class Aperture(ThimblesTable, Base, HasParameterContext):
    name = Column(String)
    info = Column(PickleType)
    context = relationship("ApertureAlias", collection_class=NamedParameters)
    
    def __init__(self, name, info=None):
        self.name = name
        if info is None:
            info = {}
        self.info = info
    
    def __repr__(self):
        return "Aperture: {}".format(self.name)
    
    def add_parameter(self, parameter_name, parameter, is_compound=False):
        ApertureAlias(name=parameter_name, context=self, parameter=parameter, is_compound=is_compound)

class OrderAlias(ParameterAliasMixin, ThimblesTable, Base):
    _context_id = Column(Integer, ForeignKey("Order._id"))
    context = relationship("Order", foreign_keys=_context_id, back_populates="context")

class Order(ThimblesTable, Base):
    number = Column(Integer)
    context = relationship("OrderAlias", collection_class=NamedParameters)
    
    def __init__(self, number):
        self.number = number
    
    def __repr__(self):
        return "Order: {}".format(self.number)

    def add_parameter(self, parameter_name, parameter, is_compound=False):
        OrderAlias(name=parameter_name, context=self, parameter=parameter, is_compound=is_compound)
        
    
class ChipAlias(ParameterAliasMixin, ThimblesTable, Base):
    _context_id = Column(Integer, ForeignKey("Chip._id"))
    context = relationship("Chip", foreign_keys=_context_id, back_populates="context")

class Chip(ThimblesTable, Base, HasParameterContext):
    name = Column(String)
    info = Column(PickleType)
    context = relationship("ChipAlias", collection_class=NamedParameters)
    
    def __init__(self, name, info=None):
        self.name = name
        if info is None:
            info = {}
        self.info = info
    
    def __repr__(self):
        return "Chip: {}".format(self.name)

    def add_parameter(self, parameter_name, parameter, is_compound=False):
        ChipAlias(name=parameter_name, context=self, parameter=parameter, is_compound=is_compound)
