import time
import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
from copy import copy

import thimbles as tmb
from thimbles.sqlaimports import *
from thimbles.thimblesdb import ThimblesTable, Base
from thimbles.modeling import ParameterGroup
from thimbles.modeling.associations import NamedParameters
from thimbles.modeling.associations import HasParameterContext
from sqlalchemy.orm.collections import collection


class Distribution(ThimblesTable, Base, HasParameterContext):
    distribution_class = Column(String)
    __mapper_args__={
        "polymorphic_identity":"Distribution",
        "polymorphic_on":distribution_class
    }
    
    def __init__(self, parameters=None):
        HasParameterContext.__init__(self, context_dict=parameters)
    
    def __getitem__(self, index):
        return self.context[index]


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
