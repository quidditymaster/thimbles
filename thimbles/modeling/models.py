import time
import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
from copy import copy

from thimbles.sqlaimports import *
from thimbles.thimblesdb import ThimblesTable, Base
from thimbles.modeling import ParameterGroup, Parameter


class InputAssociation(ThimblesTable, Base):
    _model_id = Column(Integer, ForeignKey("Model._id"))
    _parameter_id = Column(Integer, ForeignKey("Parameter._id"))
    model = relationship("Model")
    parameter = relationship("Parameter", backref="_models")
    name = Column(String)
    ordering_index = Column(Integer, default=-1)
    as_list = Column(Boolean, default=False)
    
    def __init__(self, model, parameter, name, ordering_index=None, as_list=None):
        self.model = model
        self.parameter = parameter
        self.name=name
        if not ordering_index is None:
            self.ordering_index = ordering_index
        if not as_list is None:
            self.as_list=as_list

class Model(ParameterGroup, ThimblesTable, Base):
    model_class = Column(String)
    __mapper_args__={
        "polymorphic_identity": "Model",
        "polymorphic_on": model_class
    }
    _inputs = relationship("InputAssociation", order_by=InputAssociation.ordering_index)
    _output_parameter_id = Column(
        Integer, 
        ForeignKey("Parameter._id")
    )
    output_p = relationship(
        "Parameter",
        foreign_keys=_output_parameter_id,
        backref="mapped_models", 
    )
    
    
    @property
    def parameters(self):
        parameters = []
        for param_assoc in self._inputs:
            parameters.append(param_assoc.parameter)
        return parameters
    
    @property
    def inputs(self):
        inp_assocs = self._inputs
        inp_dict = {}
        for assoc in inp_assocs:
            name = assoc.name
            if assoc.as_list:
                parameter = inp_dict.get(name)
                if parameter is None:
                    parameter = []
                parameter.append(assoc.parameter)
            else:
                parameter = assoc.parameter
            inp_dict[name] = parameter
        return inp_dict
    
    def add_input(
            self, 
            name,
            parameter, 
            as_list=None, 
            ordering_index=None, 
    ):
        if ordering_index is None:
            if len(self._inputs) > 0:
                last_assoc = self._inputs[-1]
                ordering_index = last_assoc.ordering_index + 1
            else:
                ordering_index = 0
        inp_assoc = InputAssociation(
            model=self, 
            parameter=parameter,
            name=name,
            ordering_index=ordering_index, 
            as_list=as_list
        )
        self._inputs.append(inp_assoc)
    
    def __call__(self, vdict=None):
        raise NotImplementedError("Model is intended strictly as a parent class. you must subclass it and implement a __call__ method.")
    
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
                p.set(flat_pval.reshape(pshape))
        pexp_mat = scipy.sparse.hstack(deriv_vecs)
        return pexp_mat
