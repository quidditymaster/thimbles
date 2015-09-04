import time
import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
from copy import copy

import thimbles as tmb
from ..sqlaimports import *
from ..thimblesdb import ThimblesTable, Base
from .associations import NamedParameters, ParameterAliasMixin
from .associations import HasParameterContext
from sqlalchemy.orm.collections import collection


class DistributionAlias(ThimblesTable, Base, ParameterAliasMixin):
    parameter = relationship("Parameter", back_populates="distributions")
    _context_id = Column(Integer, ForeignKey("Distribution._id"))
    context= relationship("Distribution", foreign_keys=_context_id, back_populates="context")
    
    def __init__(self, **kwargs):
        ParameterAliasMixin.__init__(self, **kwargs)


class Distribution(ThimblesTable, Base, HasParameterContext):
    distribution_class = Column(String)
    __mapper_args__={
        "polymorphic_identity":"Distribution",
        "polymorphic_on":distribution_class
    }
    context = relationship("DistributionAlias", collection_class=NamedParameters)
    
    def __init__(self, parameters=None):
        HasParameterContext.__init__(self, context_dict=parameters)
    
    def add_parameter(self, name, parameter, is_compound=False):
        DistributionAlias(name=name, context=self, parameter=parameter, is_compound=is_compound)


class NormalDistribution(Distribution):
    _id = Column(Integer, ForeignKey("Distribution._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"NormalDistribution",
    }
    mean = Column(PickleType)
    ivar = Column(PickleType)
    
    def __init__(self, mean, ivar, parameters=None):
        Distribution.__init__(self, parameters=parameters)
        self.mean = np.asarray(mean)
        self.ivar = np.asarray(ivar)
    
    @property
    def var(self):
        return tmb.utils.misc.inv_var_2_var(self.ivar)
    
    @var.setter
    def var(self, value):
        self.ivar = tmb.utils.misc.var_2_inv_var(value)
