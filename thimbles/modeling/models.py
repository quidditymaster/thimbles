import time
import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
from copy import copy

from thimbles.sqlaimports import *
from thimbles.thimblesdb import ThimblesTable, Base
from thimbles.modeling import Parameter
from thimbles.modeling.associations import NamedParameters
from thimbles.modeling.associations import ParameterAliasMixin

class InputAlias(ParameterAliasMixin, ThimblesTable, Base):
    parameter = relationship("Parameter", back_populates="models")
    _context_id = Column(Integer, ForeignKey("Model._id"))
    context= relationship("Model", foreign_keys=_context_id, back_populates="inputs")

class Model(ThimblesTable, Base):
    model_class = Column(String)
    __mapper_args__={
        "polymorphic_identity": "Model",
        "polymorphic_on": model_class
    }
    inputs = relationship(
        "InputAlias",
        collection_class=NamedParameters,
    )
    
    _output_id = Column(
        Integer, 
        ForeignKey("Parameter._id")
    )
    output_p = relationship(
        "Parameter",
        foreign_keys=_output_id,
        backref="mapped_models", 
    )
    
    def __init__(self, output_p, **kwargs):
        self.output_p=output_p
        for kw in kwargs:
            self.add_parameter(kw, kwargs[kw], is_compund=False)
    
    def add_parameter(self, name, parameter, is_compound=False):
        InputAlias(name=name, context=self, parameter=parameter, is_compound=is_compound)
    
    @property
    def parameters(self):
        return self.inputs.parameters
    
    def __call__(self, vdict=None):
        raise NotImplementedError("Model is intended strictly as a parent class. you must subclass it and implement a __call__ method.")
    
    def fire(self):
        val = self()
        self.output_p.set(val)
    
    def get_vdict(self, replacements=None):
        if replacements is None:
            replacements = {}
        vdict = {}
        for p in self.inputs:
            pval = p.value
            vdict[p] = pval
        vdict.update(replacements)
        return vdict
    
    def fast_deriv(self, param):
        """
This model should be implemented for performance sensitive or frequently used model classes.

calculate the derivative of the output parameter of this model with respect to the given input parameter in the region near the current model parameter values.
        
 if this method returns None for a particular input parameter the dervative determination harness will assume that a discrete derivative will need to be calculated for parameter influence paths containing that parameter/model pair. 
        """
        return None

