import time
import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
from copy import copy

from thimbles.sqlaimports import *
from thimbles.thimblesdb import ThimblesTable, Base


class Model(ParameterGroup, ThimblesTable, Base):
    model_type = Column(String)
    __mapper_args__={
        "polymorphic_identity": "model",
        "polymorphic_on": model_type
    }
    parameters = relationship("Parameter", backref="model")
    
    def __init__(self):
        if len(self.parameters) == 0:
            self.attach_parameters()
        ThimblesTable.__init__(self)
    
    def attach_parameters(self):
        self.parameters = []
        for attrib in dir(self):
            try:
                val = getattr(self, attrib)
            except Exception:
                continue
            if isinstance(val, ParameterFactory):
                new_param = val.make_parameter()
                new_param.set_model(self)
                if new_param.name is None:
                    new_param.name = attrib
                new_param.validate()
                setattr(self, attrib, new_param)
    
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
    
    def as_linear_op(self, input_vec, **kwargs):
        return scipy.sparse.identity(len(input_vec))#implicitly allowing additive model passthrough


class IdentityPlaceHolder(object):
    
    def __mul__(self, other):
        return other
    
    def __rmul__(self, other):
        return other
