from .derivatives import deriv

from thimbles.sqlaimports import *
from thimbles.thimblesdb import ThimblesTable, Base
from thimbles.modeling.associations import HasParameterContext, NamedParameters, ParameterAliasMixin
from thimbles.modeling.distributions import NormalDistribution

class EstimatorAlias(ThimblesTable, Base, ParameterAliasMixin):
    parameter = relationship("Parameter", back_populates="estimators")
    _context_id = Column(Integer, ForeignKey("Estimator._id"))
    context= relationship("Estimator", foreign_keys=_context_id, back_populates="context")

    def __init__(self, **kwargs):
        ParameterAliasMixin.__init__(self, **kwargs)


class Estimator(HasParameterContext, ThimblesTable, Base, ):
    estimator_class = Column(String)
    __mapper_args__={
        "polymorphic_on":estimator_class,
        "polymorphic_identity":"Estimator",
    }
    context = relationship("EstimatorAlias", collection_class=NamedParameters)
    _target_distribution_id = Column(Integer, ForeignKey("Distribution._id"))
    target_distribution = relationship("Distribution", foreign_keys=_target_distribution_id)
    
    def __init__(self, informants, informed):
        HasParameterContext.__init__(self, context_dict=informants)
        dist = NormalDistribution(parameters=informed)
        self.target_distribution = dist
    
    def add_parameter(self, name, parameter, is_compound=False):
        EstimatorAlias(name=name, context=self, parameter=parameter, is_compound=is_compound)
    
    def __call__(self, set_current=True):
        raise NotImplementedError("Estimator.__call__ should be implemented in the Estimator subclass")


class IterativeDecoupledLinearEstimator(object):
    __mapper_args__={
        "polymorphic_identity":"Estimator",
    }
    
    def __init__(
            self,
            #informant = {"pname1":[p1, p2], "pname2":p3}
            informant_distribution,
            #informant name
            
    ):
        self.informants = informants
    
    def __call__(self):
        #find the informant distributions and parameters
        
        
        #build the derivative matrix
        
        dmat = modeling.deriv(self.distribution_informants)
        

