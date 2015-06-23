import numpy as np
import scipy
from thimbles.sqlaimports import *
from thimbles.thimblesdb import ThimblesTable, Base
from thimbles.modeling.associations import HasParameterContext
from thimbles.modeling.distributions import NormalDistribution

class Estimator(ThimblesTable, Base, ):
    estimator_class = Column(String)
    __mapper_args__={
        "polymorphic_on":estimator_class,
        "polymorphic_identity":"Estimator",
    }
    _target_distribution_id = Column(Integer, ForeignKey("Distribution._id"))
    target_distribution = relationship("Distribution", foreign_keys=_target_distribution_id)
    
    
    def __init__(self, informants, informed):
        HasParameterContext.__init__(self, context_dict=informants)
        dist = NormalDistribution(parameters=informed)
    
    def __call__(self, set_current=True):
        pass
