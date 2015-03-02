import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import thimbles as tmb
from thimbles.modeling.models import Model, Parameter
from thimbles.sqlaimports import *


class MultiplierModel(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"MultiplierModel",
    }
    
    def __init__(self, parameters, output_p=None, substrate=None):
        if parameters is None:
            parameters = []
        self.parameters = parameters
        self.substrate = substrate
        self.output_p = output_p
    
    def __call__(self, pvrep=None):
        pvd = self.get_vdict(pvrep)
        prod = reduce(lambda x, y: x*y, pvd.values())
        return prod


class MatrixParameter(Parameter):
    _id = Column(Integer, ForeignKey("Parameter._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"MatrixParameter",
    }
    
    def __init__(self, matrix=None):
        self._value = matrix


class VectorParameter(Parameter):
    _id = Column(Integer, ForeignKey("Parameter._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"VectorParameter",
    }
    _value = Column(PickleType)
    
    def __init__(self, vector=None):
        self._value = vector


class MatrixMultiplierModel(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"MatrixMultiplierModel",
    }
    
    def __init__(self, parameters=None, output_p=None, substrate=None):
        if parameters is None:
            parameters = []
        self.parameters = parameters
        self.output_p = output_p
        self.substrate = substrate
    
    @property
    def matrix_p(self):
        if self._matrix_p is None:
            for p in self.parameters:
                if isinstance(p, MatrixParameter):
                    self._matrix_p = p
        return self._matrix_p
    
    @property
    def vector_p(self):
        if self._vector_p is None:
            for p in self.parameters:
                if isinstance(p, VectorParameter):
                    self._vector_p = p
        return self._vector_p
    
    @property
    def matrix(self):
        return self.matrix_p.value
    
    @property
    def vector(self):
        return self.vector_p.value
    
    def __call__(self, pvrep=None):
        pvd = self.get_vdict(pvrep)
        matrix = pvd[self.matrix_p]
        vector = pvd[self.vector_p]
        return matrix*vector


class FluxSumModel(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"FluxSumModel",
    }
    background_level = Column(Float)
    
    def __init__(self, parameters=None, output_p=None, substrate=None):
        if parameters is None:
            parameters = []
        self.parameters = parameters
        self.output_p = output_p
        self.substrate=substrate
    
    def __call__(self, pvrep=None):
        vdict = self.get_vdict(pvrep)
        if len(vdict) == 0:
            return None
        fparams = vdict.keys()
        output_sample = self.output_p.wv_sample
        fsum = np.zeros(len(output_sample))
        out_start = output_sample.start
        for fp in fparams:
            start, end = fp.wv_sample.start, fp.wv_sample.end
            start = start-out_start
            end = end-out_start
            fsum[start:end] += fp.value
        return fsum


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
        "polymorphic_identity":"PolynomialFluxModel",
    }
    center = Column(Float)
    scale = Column(Float)
    
    def __init__(self, wv_sample, coeffs=None, center=None, scale=None):
        wv_sample = tmb.as_wavelength_sample(wv_sample)
        fp = FluxParameter(wv_sample)
        self.output_p = fp
        if coeffs is None:
            coeffs = np.zeros(3)
        vp = VectorParameter(coeffs)
        self.parameters = [vp]
        if center is None:
            center = 0.5*(wv_sample.start + wv_sample.end)
        self.center = center
        if scale is None:
            scale = max(0.25*(wv_sample.end - wv_sample.start), 1.0)
        self.scale = scale
    
    def __call__(self, vprep=None):
        vdict = self.get_vdict(vprep)
        pix = self.output_p.wv_sample.pixels
        coeffs ,= vdict.values()
        return np.polyval(coeffs, (pix-self.center)/self.scale)


class IdentityOperation(object):
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
