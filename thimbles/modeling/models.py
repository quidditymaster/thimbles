import time
import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
from copy import copy

from thimbles.sqlaimports import *
from thimbles.thimblesdb import ThimblesTable, Base
from thimbles.modeling import ParameterGroup, Parameter

input_assoc = sa.Table(
    "input_assoc", 
    Base.metadata,
    Column("model_id", Integer, ForeignKey("Model._id")),
    Column("parameter_id", Integer, ForeignKey("Parameter._id")),
)

class ModelLogic(ParameterGroup):
    """A convenience class which partially implements the Model API.
    
    In order to play nice with the SQLAlchemy backend objects which it is 
    convenient to make act like models. 
    
    """
    
    def __call__(self, vdict=None):
        raise NotImplementedError("ModelLogic is intended strictly as a parent class it cannot be executed. you must subclass it and implement a __call__ method for your subclass.")
    
    def fire(self):
        self.output_p.value = self()
    
    def get_vdict(self, replacements=None):
        if replacements is None:
            replacements = {}
        values_dict = {p:p.value for p in self.parameters}
        values_dict.update(replacements)
        return values_dict
    
    def parameter_expansion(self, input_vec, **kwargs):
        parameters = self.free_parameters
        deriv_vecs = []
        for p in parameters:
            if not p._expander is None:
                deriv_vecs.append(p.expand(input_vec, **kwargs))
                continue 
            pval = p.get()
            pshape = p.shape
            #TODO: use the parameters own derivative_scale scale
            if pshape == tuple():
                p.set(pval+p.derivative_scale)
                plus_d = self(input_vec, **kwargs)
                p.set(pval-p.derivative_scale)
                minus_d = self(input_vec, **kwargs)
                deriv = (plus_d-minus_d)/(2.0*p.derivative_scale)
                deriv_vecs.append(scipy.sparse.csc_matrix(deriv.reshape((-1, 1))))
                #don't forget to reset the parameters back to the start
                p.set(pval)
            else:
                flat_pval = pval.reshape((-1,))
                cur_n_param =  len(flat_pval)
                if cur_n_param > 50:
                    raise ValueError("""your parameter vector is large consider implementing your own expansion via the Parameter.expander decorator""".format(cur_n_param))
                delta_vecs = np.eye(cur_n_param)*p.derivative_scale
                for vec_idx in range(cur_n_param):
                    minus_vec = flat_pval - delta_vecs[vec_idx]
                    p.set(minus_vec.reshape(pshape))
                    minus_d = self(input_vec, **kwargs)
                    plus_vec = flat_pval + delta_vecs[vec_idx]
                    p.set(plus_vec.reshape(pshape))
                    plus_d = self(input_vec, **kwargs)
                    deriv = (plus_d - minus_d)/(2.0*delta_vecs[vec_idx, vec_idx])
                    deriv_vecs.append(scipy.sparse.csc_matrix(deriv.reshape((-1, 1))))
                    #import pdb; pdb.set_trace()
                p.set(flat_pval.reshape(pshape))
        pexp_mat = scipy.sparse.hstack(deriv_vecs)
        return pexp_mat


class Model(ModelLogic, ThimblesTable, Base):
    model_class = Column(String)
    _substrate_id = Column(Integer, ForeignKey("ModelSubstrate._id"))
    #substrate = relationship("ModelSubstrate", backref="models")
    __mapper_args__={
        "polymorphic_identity": "Model",
        "polymorphic_on": model_class
    }
    parameters = relationship(
        "Parameter", 
        secondary=input_assoc, 
        backref="models",
    )
    _output_parameter_id = Column(Integer, ForeignKey("Parameter._id"))
    output_p = relationship(
        "Parameter",
        foreign_keys=_output_parameter_id,
        backref="mapped_models", 
    )
    
    #class attributes
    _derivative_helpers = {} 
    
    def __init__(self, parameters=None, output_p=None, substrate=None):
        if parameters is None:
            parameters = []
        self.parameters = parameters
        self.output_p = output_p
        self.substrate=substrate

