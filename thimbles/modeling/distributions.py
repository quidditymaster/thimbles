import time
import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
from copy import copy

from thimbles.sqlaimports import *
from thimbles.thimblesdb import ThimblesTable, Base
from thimbles.modeling import ParameterGroup

dist_assoc = sa.Table("distribution_assoc", Base.metadata,
    Column("distribution_id", Integer, ForeignKey("ParameterDistribution._id")),
    Column("parameter_id", Integer, ForeignKey("Parameter._id")),
)

class ParameterDistribution(ParameterGroup, ThimblesTable, Base):
    relationship("Parameters", secondary=dist_assoc)
    distribution_type = Column(String)
    __mapper_args__={
        "polymorphic_identity":"ParameterDistribution",
        "polymorphic_on":distribution_type
    }
    
    def log_likelihood(self, value):
        raise NotImplementedError("Abstract Class")
    
    def as_sog(self, value=None, radius=None, embedding_space=None, n_max=10):
        raise NotImplementedError("Abstract Class")
    
    def realize(self):
        raise NotImplementedError("Abstract Class")


class SumOfGaussians:#(ParameterDistribution):
    _relative_probs = Column(PickleType) #(n_gauss,) numpy array
    _covariances = Column(PickleType) #(n_gauss, n_dims) numpy array for axis aligned or (n_gauss, n_dims, n_dims) for full covariance matrices
    

class MultivariateNormalDistribution:#(ParameterDistribution):
    _variance = Column(PickleType)
        
    def realize(self):
        return np.random.normal(size=self._variance.shape)*self._variance
    
