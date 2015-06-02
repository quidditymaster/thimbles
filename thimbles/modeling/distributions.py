import time
import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
from copy import copy

from thimbles.sqlaimports import *
from thimbles.thimblesdb import ThimblesTable, Base
from thimbles.modeling import ParameterGroup
from thimbles.modeling.associations import NamedParameters
from thimbles.modeling.associations import DistributionAlias
from sqlalchemy.orm.collections import collection


class Distribution(ThimblesTable, Base):
    parameters = relationship(
        "DistributionAlias", 
        collection_class=NamedParameters,
    )
    distribution_class = Column(String)
    __mapper_args__={
        "polymorphic_identity":"Distribution",
        "polymorphic_on":distribution_class
    }
    
    def log_likelihood(self, value):
        raise NotImplementedError("Abstract Class")

    def add_parameter(self, name, param, is_compound=False):
        dist_alias = DistributionAlias(name, self, param, is_compound=is_compound)
    
    def __getitem__(self, index):
        return self.parameters[index]
        
    def as_sog(self, center=None, radius=None, embedding_space=None, n_max=10):
        raise NotImplementedError("Abstract Class")
    
    def realize(self):
        raise NotImplementedError("Abstract Class")
    
    def __add__(self, other):
        raise NotImplementedError("Abstract Class")


class NormalDistribution(Distribution):
    _id = Column(Integer, ForeignKey("Distribution._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"NormalDistribution",
    }
    mean = Column(Float)
    ivar = Column(Float)
    
    def __init__(self, mean, ivar, parameters=None):
        if parameters is None:
            parameters = []
        elif len(parameters) > 1:
            raise ValueError("a NormalDistribution is strictly one dimensional perhaps try using VectorNormalDistribution instead")
        self.parameters=parameters
        self.mean = np.asarray(mean)
        self.ivar = np.asarray(ivar)


class VectorNormalDistribution(Distribution):
    _id = Column(Integer, ForeignKey("Distribution._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"NormalDistribution",
    }
    mean = Column(PickleType)
    var = Column(PickleType)
    
    def __init__(self, mean, var, parameters=None):
        if parameters is None:
            parameters = {}
        for pname in parameters:
            self.add_parameter(pname, parameters[pname])
        
        #self.parameters=parameters
        self.mean = np.asarray(mean)
        self.var = np.asarray(var)
    
    def realize(self):
        return np.random.normal(size=self.var.shape)*self.var


class SumOfGaussians:#(ParameterDistribution):
    _relative_probs = Column(PickleType) #(n_gauss,) numpy array
    _covariances = Column(PickleType) #(n_gauss, n_dims) numpy array for axis aligned or (n_gauss, n_dims, n_dims) for full covariance matrices
    
    
    
