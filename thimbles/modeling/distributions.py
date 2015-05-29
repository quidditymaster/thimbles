import time
import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
from copy import copy

from thimbles.sqlaimports import *
from thimbles.thimblesdb import ThimblesTable, Base
from thimbles.modeling import ParameterGroup
from sqlalchemy.orm.collections import collection

class DistributionAlias(ThimblesTable, Base):
    _parameter_id = Column(Integer, ForeignKey("Parameter._id"))
    parameter = relationship("Parameter")#, back_populates="distributions")
    _distribution_id = Column(Integer, ForeignKey("Distribution._id"))
    distribution = relationship("Distribution", foreign_keys=_distribution_id, back_populates="parameters")
    name = Column(String)
    is_compound = Column(Boolean)
    
    _param_temp = None
    
    def __init__(self, name, distribution, parameter, is_compound=False):
        self.name = name
        self.is_compound=is_compound
        self._param_temp = parameter #don't trigger the back pop yet
        self.distribution = distribution
        self.parameter = parameter

class SampleSpace(ParameterGroup):
    
    def __init__(self):
        self._aliases = []
        self.groups = {}
        self.parameters = []
    
    def __getitem__(self, index):
        return self.groups[index]
    
    def __len__(self):
        return len(self.parameters)
    
    @collection.appender
    def append(self, param_alias):
        pname = param_alias.name
        if param_alias._param_temp is None:
            param = param_alias.parameter
        else:
            param = param_alias._param_temp
        is_compound = param_alias.is_compound
        pgroup = self.groups.get(pname)
        if pgroup is None:
            if is_compound:
                self.groups[pname] = [param]
            else:
                self.groups[pname] = param
        else:
            if is_compound:
                self.groups[pname].append(param)
            else:
                print("WARNING: redundant non-compound InputAlias objects for model {} and parameter {}\n former alias is unreachable by name but will still show up in the .parameters collection".format(param_alias.model, param))
                self.groups[pname] = param
        self.parameters.append(param)
        self._aliases.append(param_alias)
    
    @collection.remover
    def remove(self, param_alias):
        pname = param_alias.name
        param = param_alias.parameter
        if param_alias.is_compound:
            pgroup = self.groups[pname]
            pgroup.remove(param)
            if len(pgroup) == 0:
                self.groups.pop(pname)
        else:
            self.groups[pname].pop(pname)
        self.parameters.remove(param)
        self._aliases.remove(param_alias)
    
    @collection.iterator
    def _iter_aliases(self):
        for alias in self._aliases:
            yield alias
    
    def __iter__(self):
        for p in self.parameters:
            yield p


class Distribution(ThimblesTable, Base):
    parameters = relationship(
        "DistributionAlias", 
        collection_class=SampleSpace,
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
    
    
    
