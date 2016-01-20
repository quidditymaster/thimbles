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
MultiplierModel
LogisticModel
LinearIndexerModel
InterpolationMatrixModel
KernelWeightedMatchingInterpolatorModel
""".split()

class MultiplierModel(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"MultiplierModel",
    }
    
    def __init__(
            self, 
            output_p=None, 
            factors=None, 
    ):
        self.output_p = output_p
        if factors is None:
            factors = []
        for param in factors:
            self.add_parameter("factors", param, is_compound=True)
    
    def __call__(self, pvrep=None):
        vdict = self.get_vdict(pvrep)
        factors = self.inputs["factors"]
        product = vdict[factors[0]]
        for param in factors[1:]:
            product = product*vdict[param]
        return product
    
    def fast_deriv(self, param):
        all_factors = self.parameters
        complimentary_prod = 1.0
        param_is_dependent = False
        for factor in all_factors:
            if factor != param:
                complimentary_prod *= factor.value
            else:
                param_is_dependent = True
        if not param_is_dependent:
            raise ValueError("given parameter is not an input of this model!")
        return complimentary_prod


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
        "polymorphic_identity":"IntegerParameter",
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
    
    def __init__(self, output_p, matrix, vector):
        self.output_p = output_p
        self.add_parameter("matrix", matrix_p)
        self.add_parameter("vector", vector_p)
    
    def __call__(self, vprep=None):
        vdict = self.get_vdict(vprep)
        mat = vdict[self.inputs["matrix"]]
        vec = vdict[self.inputs["vector"]]
        return mat*vec
    
    def fast_deriv(self, param):
        if param == self.inputs["vector"]:
            return self.inputs["matrix"].value
        return None


class NegativeExponentialModel(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"NegativeExponentialModel",
    }
    
    def __init__(self, output_p, optical_depth):
        self.output_p = output_p
        self.add_parameter("optical_depth", optical_depth)
    
    def __call__(self, vprep=None):
        vdict = self.get_vdict(vprep)
        op_depth = vdict[self.inputs["optical_depth"]]
        return np.exp(-op_depth)
    
    def fast_deriv(self, param):
        opd_param = self.inputs["optical_depth"]
        if param == opd_param:
            opd_val = opd_param.value
            val_deriv = -opd_val*np.exp(-opd_val)
            return scipy.sparse.dia_matrix((val_deriv, 0), shape=(len(opd_val), len(opd_val)))
        return None


class PixelPolynomialModel(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"PixelPolynomialModel",
    }
    npts = Column(Integer)
    
    def __init__(
            self, 
            output_p,
            coeffs,
            npts,
    ):
        self.output_p = output_p
        self.npts = npts
        if not isinstance(coeffs, Parameter):
            coeffs = PickleParameter(coeffs)
        self.add_parameter("coeffs", coeffs)
    
    def get_x(self, pixels=None):
        if pixels is None:
            pixels = np.arange(self.npts)
        return (pixels-0.5*self.npts)/(0.5*self.npts)
    
    def __call__(self, vprep=None):
        vdict = self.get_vdict(vprep)
        coeffs = vdict[self.inputs["coeffs"]]
        return np.polyval(coeffs, self.get_x())


class LogisticModel(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    __mapper_args__ = {
        "polymorphic_identity":"LogisticModel",
    }
    
    def __init__(self, output_p, x_p, slope_p=None):
        self.output_p = output_p
        self.add_parameter("x", x_p)
        if slope_p is None:
            slope_p = FloatParameter(1.0)
        self.add_parameter("slope", slope_p)
    
    def __call__(self, vprep=None):
        vdict = self.get_vdict(vprep)
        x = vdict[self.inputs["x"]]
        slope = vdict[self.inputs["slope"]]
        return 1.0/(1.0 + np.exp(-slope*x))


class InterpolationMatrixModel(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    __mapper_args__ = {
        "polymorphic_identity":"InterpolationMatrixModel",
    }

    def __init__(self, output_p, coord_p, indexer_p):
        self.output_p = output_p
        self.add_parameter("coords", coord_p)
        self.add_parameter("indexer", indexer_p)
    
    def __call__(self, vprep=None):
        vdict = self.get_vdict(vprep)
        coords = vdict[self.inputs["coords"]]
        indexer = vdict[self.inputs["indexer"]]
        return indexer.interpolant_sampling_matrix(coords)

class LinearIndexerModel(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    __mapper_args__ = {
        "polymorphic_identity":"LinearIndexerModel",
    }
    min_coord = Column(Float)
    max_coord = Column(Float)
    
    def __init__(self, output_p, indexed_p, min_coord=0.0, max_coord=1.0):
        self.output_p = output_p
        self.add_parameter("indexed", indexed_p)
        self.min_coord = min_coord
        self.max_coord = max_coord
    
    def __call__(self, vprep=None):
        vdict = self.get_vdict(vprep)
        indexed = vdict[self.inputs["indexed"]]
        return tmb.coordinatization.LinearCoordinatization(min=self.min_coord, max=self.max_coord, npts=len(indexed))


class KernelWeightedMatchingInterpolatorModel(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    __mapper_args__ = {
        "polymorphic_identity":"MatchingInterpolatorModel",
    }
    kernel_gamma = Column(Float)
    kernel_sigma = Column(Float)
    matching_tolerance = Column(Float)
    
    def __init__(
            self, 
            output_p, 
            x_p, 
            y_p, 
            kernel_sigma=1.0, 
            kernel_gamma=0.1, 
            matching_tolerance=6.0
    ):
        self.output_p = output_p
        self.add_parameter("x", x_p)
        self.add_parameter("y", y_p)
        self.kernel_gamma = kernel_gamma
        self.kernel_sigma = kernel_sigma
        self.matching_tolerance = matching_tolerance
    
    def __call__(self, vprep):
        vdict = self.get_vdict(vprep)
        x = vdict[self.inputs["x"]]
        y = vdict[self.inputs["y"]]
        kernel = lambda x:np.exp(-0.5*(x/kernel_sigma)**2)/(1.0+(x/kernel_gamma)**2)
        return latbin.interpolation.KernelWeightedMatchingInterpolator(x=x, y=y, weighting_kernel=kernel, matching_tolerance=matching_tolerance)


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
    
