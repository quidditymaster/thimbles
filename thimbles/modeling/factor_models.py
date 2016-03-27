import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import thimbles as tmb
from thimbles.modeling.models import Model, Parameter
from thimbles.thimblesdb import HasName
from thimbles.sqlaimports import *
from functools import reduce
import scipy

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
    
    def __call__(self, override=None):
        vdict = self.get_vdict(override)
        factors = self.inputs["factors"]
        product = vdict[factors[0]]
        for param in factors[1:]:
            product = product*vdict[param]
        return product
    
    def fast_deriv(self, param, override=None):
        if override is None:
            override = {}
        all_factors = self.parameters
        complimentary_prod = 1.0
        param_is_dependent = False
        for factor in all_factors:
            if factor != param:
                if not (param in override):
                    complimentary_prod *= factor.value
                else:
                    complimentary_prod *= override[param]
            else:
                param_is_dependent = True
        if not param_is_dependent:
            raise ValueError("given parameter is not an input of this model!")
        out_shape = (len(complimentary_prod), len(complimentary_prod))
        der_mat = scipy.sparse.dia_matrix((complimentary_prod, 0), shape=out_shape)
        return der_mat


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
        self.add_parameter("matrix", matrix)
        self.add_parameter("vector", vector)
    
    def __call__(self, override=None):
        vdict = self.get_vdict(override)
        mat = vdict[self.inputs["matrix"]]
        vec = vdict[self.inputs["vector"]]
        return mat*vec
    
    def fast_deriv(self, param, override=None):
        if override is None:
            override = {}
        if param == self.inputs["vector"]:
            matrix_p = self.inputs["matrix"]
            if matrix_p in override:
                return override[matrix_p]
            else:
                return matrix_p.value
        return None


class NegativeExponentialModel(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"NegativeExponentialModel",
    }
    
    def __init__(self, output_p, tau):
        self.output_p = output_p
        self.add_parameter("tau", tau)
    
    def __call__(self, override=None):
        vdict = self.get_vdict(override)
        tau = vdict[self.inputs["tau"]]
        return np.exp(-tau)
    
    def fast_deriv(self, param, override=None):
        tau_param = self.inputs["tau"]
        if param == tau_param:
            if tau_param in override:
                raise ValueError("overriding derivative param")
            tau_val = tau_param.value
            val_deriv = -tau_val*np.exp(-tau_val)
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
            pixels = np.arange(self.npts, dtype=float)
        return (pixels-0.5*self.npts)/(0.5*self.npts)
    
    def __call__(self, override=None):
        vdict = self.get_vdict(override)
        coeffs = vdict[self.inputs["coeffs"]]
        return np.polyval(coeffs, self.get_x())
    
    def fast_deriv(self, param, override=None):
        coeff_p = self.inputs["coeffs"]
        if param is coeff_p:
            xvals = self.get_x()
            vdict = self.get_vdict(override)
            coeff_val = vdict[coeff_p]
            vander_mat = scipy.sparse.csr_matrix(np.vander(xvals, len(coeff_val)))
            return vander_mat


class InterpolationMatrixModel(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    __mapper_args__ = {
        "polymorphic_identity":"InterpolationMatrixModel",
    }
    
    def __init__(self, output_p, coords, indexer):
        self.output_p = output_p
        self.add_parameter("coords", coord)
        self.add_parameter("indexer", indexer)
    
    def __call__(self, override=None):
        vdict = self.get_vdict(override)
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
    
    def __call__(self, override=None):
        vdict = self.get_vdict(override)
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
            x, 
            y, 
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
    
    def __call__(self, override):
        vdict = self.get_vdict(override)
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
    
    def __call__(self, override=None):
        return IdentityMap()
    
