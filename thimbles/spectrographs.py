import numpy as np
import scipy.sparse

from thimbles.utils.partitioning import partitioned_polynomial_model
from thimbles.utils import piecewise_polynomial as ppol 
from thimbles.thimblesdb import Base, ThimblesTable
from thimbles.modeling import Model, Parameter
from thimbles.sqlaimports import *
from thimbles.modeling import PixelPolynomialModel
import thimbles as tmb


class SamplingModel(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"SamplingModel",
    }
    
    def __init__(
            self, 
            output_p, 
            input_wvs_p, 
            input_lsf_p,
            output_wvs_p,
            output_lsf_p,
    ):
        self.output_p = output_p
        self.add_parameter("input_wvs", input_wvs_p)
        self.add_parameter("input_lsf", input_lsf_p)
        self.add_parameter("output_wvs", output_wvs_p)
        self.add_parameter("output_lsf", output_lsf_p)  
    
    def __call__(self, vprep=None):
        vdict = self.get_vdict(vprep)
        x_in = vdict[self.inputs["input_wvs"]].coordinates
        x_out = vdict[self.inputs["output_wvs"]].coordinates
        lsf_in = vdict[self.inputs["input_lsf"]]
        lsf_out = vdict[self.inputs["output_lsf"]]
        return tmb.resampling.resampling_matrix(x_in, x_out, lsf_in, lsf_out)


class Slit(ThimblesTable, Base):
    name = Column(String)
    info = Column(PickleType)
    
    def __init__(self, name, info=None):
        self.name = name
        if info is None:
            info = {}
        self.info = info


class Order(ThimblesTable, Base):
    number = Column(Integer)
    
    def __init__(self, number):
        self.number = number


class Chip(ThimblesTable, Base):
    name = Column(String)
    info = Column(PickleType)
    
    def __init__(self, name, info=None):
        self.name = name
        if info is None:
            info = {}
        self.info = info


class Slice(ThimblesTable, Base):
    _chip_id = Column(Integer, ForeignKey("Chip._id"))
    chip = relationship("Chip")
    _order_id = Column(Integer, ForeignKey("Order._id"))
    order = relationship("Order")
    _slit_id = Column(Integer, ForeignKey("Slit._id"))
    slit = relationship("Slit")
    
    def __init__(self, chip, order, slit):
        self.chip = chip
        self.order = order
        self.slit = slit
    
