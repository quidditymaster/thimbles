import time
import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
from copy import copy

from thimbles.sqlaimports import *
from thimbles.thimblesdb import ThimblesTable, Base

mult_func = lambda x, y: x*y
def flat_size(shape_tup):
    if shape_tup == tuple():
        return 1
    return reduce(mult_func, shape_tup)

class ParameterGroup(object):
    
    def __init__(self, parameters):
        self._parameters = parameters
    
    @property
    def parameters(self):
        return self._parameters
    
    @property
    def free_parameters(self):
        return [param for param in self.parameters if param.is_free]
    
    def parameter_index(self, parameter):
        return self.parameters.index(parameter)
    
    def get_pvec(self, attr=None, free_only=True):
        if free_only:
            parameters = self.free_parameters
        else:
            parameters = self.parameters
        if attr is None:
            pvals = [np.asarray(p.get()).reshape((-1,)) for p in parameters]
        else:
            pshapes = [p.shape for p in parameters]
            pvals = [np.array(getattr(p, attr)) for p in parameters]
            out_vec = []
            for p_idx in range(len(parameters)):
                pshape = pshapes[p_idx]
                pval = pvals[p_idx]
                if pshape == tuple() or pval.shape != tuple():
                    out_vec.append(pval)
                else:
                    out_vec.append(np.repeat(pval, flat_size(pshape)))
            pvals = out_vec
        if len(pvals) == 0:
            return None
        return np.hstack(pvals)
    
    def set_pvec(self, pvec, attr=None, free_only=True, as_delta=False):
        if free_only:
            parameters = self.free_parameters
        else:
            parameters = self.parameters
        pshapes = [p.shape for p in parameters]
        nvals = [flat_size(pshape) for pshape in pshapes]
        break_idxs = np.cumsum(nvals)[:-1]
        if as_delta:
            pvec = pvec + self.get_pvec()
        flat_params = np.split(pvec, break_idxs)
        if attr is None:
            for p_idx in range(len(parameters)):
                param = parameters[p_idx]
                pshape = pshapes[p_idx]
                flat_val = flat_params[p_idx]
                if pshape == tuple():
                    to_set = float(flat_val)
                else:
                    to_set = flat_val.reshape(pshape)
                param.set(to_set)
        elif isinstance(attr, basestring):
            for p_idx in range(len(parameters)):
                param = parameters[p_idx]
                pshape = pshapes[p_idx]
                flat_val = flat_params[p_idx]
                if pshape == tuple():
                    setattr(param, attr, float(flat_val))
                else:
                    setattr(param, attr, flat_val.reshape(pshape))
        else:
            raise ValueError("attr must be a string if set")
    
    def get_pdict(self, attr=None, free_only=True, name_as_key=False):
        if free_only:
            parameters = self.free_parameters
        else:
            parameters = self.parameters
        if name_as_key:
            keys = [p.name for p in parameters]
        else:
            keys = parameters
        if attr is None:
            values = [p.get() for p in parameters]
        else:
            values = [getattr(p, attr) for p in parameters]
        pdict = dict(zip(keys, values)) 
        return pdict
    
    def set_pdict(self, val_dict, attr=None):
        for p in val_dict:
            if attr is None:
                p.set(val_dict[p])
            else:
                setattr(p, attr, val_dict[p])


class Parameter(ThimblesTable, Base):
    _value = Column(Float) #a handle for storing and loading our model values
    model_id = Column(Integer, ForeignKey("Model._id"))
    
    parameter_type = Column(String)
    __mapper_args__={
        "polymorphic_identity":"parameter",
        "polymorphic_on": parameter_type
    }
    
    def __init__(self, 
                 getter,
                 setter,
                 expander=None,
                 free = False,
                 name = None,
                 scale= 1.0,
                 step_scale=1.0,
                 derivative_scale=1e-4,
                 convergence_scale=1e-2,
                 min=-np.inf, 
                 max=np.inf, 
                 history_max=10,
                 ):
        self._getter=getter
        self._setter=setter
        self._expander=expander
        self.model=None
        self.name = name
        self.scale=scale
        self.step_scale = step_scale
        self.derivative_scale=derivative_scale
        self.convergence_scale = convergence_scale
        self.min = min
        self.max = max
        self._free=free
        self._dist=None
        #self._parameterizer = None
        #self._contextualizer = None
        
        self.history = ValueHistory(self, history_max)   
    
    def __repr__(self):
        val = None
        try:
            val = self.get()
        except:
            pass
        return "Parameter: name={}, value={}".format(self.name, val)
    
    @property
    def dist(self):
        return self._dist
    
    @dist.setter
    def dist(self, value):
        if isinstance(value, ParameterDistribution):
            self._dist=value
            self._dist.set_parameter(self)
        else:
            asndarr = np.asarray(value, dtype=float)
            if asndarr.shape == tuple():
                asndarr=np.ones(self.value().shape)*asndarr
            self._dist = NormalDeltaDistribution(asndarr, parameter=self)
    
    @property
    def shape(self):
        return np.asarray(self.get()).shape
    
    @property
    def is_free(self):
        return self._free
    
    def remember(self, value_id=None):
        self.history.remember(value_id=value_id)
    
    def revert(self, value_id, pop=False):
        self.history.revert(value_id=value_id, pop=pop)
        
    def set_model(self, model):
        self.model = model
    
    def set(self, value, **kwargs):
        min_respected = np.atleast_1d(value) >= self.min
        max_respected = np.atleast_1d(value) <= self.max
        self._setter(self.model, np.clip(value, self.min, self.max), **kwargs) 
        if np.all(min_respected*max_respected):
            return True
        else:
            return False
    
    def expand(self, input_vec, **kwargs):
        return self._expander(self.model, input_vec, **kwargs)
    
    def get(self):
        return self._getter(self.model)
    
    def validate(self):
        if self.model is None:
            raise ModelingError("parameter.model is None")
        if self._setter is None:
            raise ModelingError("parameter has no setter")
    
    def weight(self, offset=None):
        return self.dist.weight(offset)

class VectorParameter(Parameter):
    _id = Column(Integer, ForeignKey("Parameter._id"), primary_key=True)
    _value = Column(String)
    __mapper_args__={
        "polymorphic_identity": "vectorparameter"
    }
