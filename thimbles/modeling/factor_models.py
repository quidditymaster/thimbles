import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import thimbles as tmb
from thimbles.modeling.models import Model, Parameter
from thimbles.thimblesdb import HasName
from thimbles.sqlaimports import *
from functools import reduce

__all__ = \
"""
FloatParameter
PickleParameter
IntegerParameter
PixelPolynomialModel
IdentityMap
IdentityMapModel
MatrixMultiplierModel
""".split()

class MultiplierModel(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"MultiplierModel",
    }
    
    def __init__(
            self, 
            parameters=None, 
            output_p=None, 
    ):
        if parameters is None:
            parameters = []
        self.parameters = parameters
        self.output_p = output_p
    
    def __call__(self, pvrep=None):
        pvd = self.get_vdict(pvrep)
        prod = reduce(lambda x, y: x*y, list(pvd.values()))
        return prod


class PickleParameter(Parameter):
    _id = Column(Integer, ForeignKey("Parameter._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"PickleParameter",
    }
    _value = Column(PickleType)
    
    def __init__(self, value=None):
        self._value = value


class IntegerParameter(Parameter):
    _id = Column(Integer, ForeignKey("Parameter._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"FloatParameter",
    }
    _value = Column(Integer)
    
    def __init__(self, value=None):
        self._value = value

class FloatParameter(Parameter):
    _id = Column(Integer, ForeignKey("Parameter._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"FloatParameter",
    }
    _value = Column(Float)
    
    def __init__(self, value=None):
        self._value = value


class MatrixMultiplierModel(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"MatrixMultiplierModel",
    }
    
    def __init__(self, output_p, matrix_p, vector_p):
        self.output_p = output_p
        self.add_input("matrix", matrix_p)
        self.add_input("vector", vector_p)
    
    def __call__(self, vprep=None):
        vdict = self.get_vdict(vprep)
        mat = vdict[self.inputs["matrix"]]
        vec = vdict[self.inputs["vector"]]
        return mat*vec


class FluxSumLogic(object):
    
    def __call__(self, pvrep=None):
        vdict = self.get_vdict(pvrep)
        if len(vdict) == 0:
            return None
        fparams = list(vdict.keys())
        output_sample = self.output_p.wv_sample
        fsum = np.zeros(len(output_sample))
        out_start = output_sample.start
        for fp in fparams:
            start, end = fp.wv_sample.start, fp.wv_sample.end
            start = start-out_start
            end = end-out_start
            fsum[start:end] += fp.value
        return fsum

class FluxSumModel(FluxSumLogic, Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"FluxSumModel",
    }
    background_level = Column(Float)
    
    def __init__(
            self, 
            parameters=None, 
            output_p=None, 
            name=None, 
            substrate=None
    ):
        if parameters is None:
            parameters = []
        self.parameters = parameters
        self.output_p = output_p


class NegativeExponentialModel(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"NegativeExponentialModel",
    }
    
    def __init__(self, input_param, output_p=None, substrate=None):
        self.parameters = [input_param]
        self.output_p = output_p
        self.substrate=substrate
    
    def __call__(self, vprep=None):
        p0 ,= self.parameters
        return np.exp(-p0.value)


class PixelPolynomialModel(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"PixelPolynomialModel",
    }
    npts = Column(Integer)
    
    def __init__(
            self, 
            output_p,
            coeffs=None,
            autofit=True,
            degree=4,
    ):
        self.output_p = output_p
        self.npts = len(self.output_p.value)
        wv_sample = output_p.wv_sample
        if coeffs is None:
            if autofit:
                x = self.get_x(wv_sample.pixels)
                y = self.output_p.value
                coeffs = np.polyfit(x, y, deg=degree)
            else:
                coeffs = np.zeros(degree)
                coeffs[-1] = 1.0
        coeffs_p = PickleParameter(coeffs)
        self.add_input("coeffs", coeffs_p)
    
    def get_x(self, pixels=None):
        if pixels is None:
            pixels = np.arange(self.npts)
        return (pixels-0.5*self.npts)/self.npts
    
    def __call__(self, vprep=None):
        vdict = self.get_vdict(vprep)
        pixels = self.output_p.wv_sample.pixels.astype(float)
        coeffs = vdict[self.inputs["coeffs"]]
        return np.polyval(coeffs, self.get_x(pixels))


class IdentityMap(object):
    """an object which when used in a binary operation returns the other object
    """
    
    def __mul__(self, other):
        return other
    
    def __div__(self, other):
        return other
    
    def __add__(self, other):
        return other
    
    def __sub__(self, other):
        return other
    
    def __rmul__(self, other):
        return other
    
    def __rdiv__(self, other):
        return other
    
    def __radd__(self, other):
        return other
    
    def __rsub__(self, other):
        return other

def IdentityMapModel(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key)
    __mapper_args__={
        "polymorphic_identity":"IdentityMatrixModel",
    }
    
    def __call__(self, vprep):
        return IdentityMap()
    
