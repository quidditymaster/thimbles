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
    
    def __init__(self, parameters=None):
        if parameters is None:
            parameters = []
        self.parameters = parameters
    
    @property
    def free_parameters(self):
        return [param for param in self.parameters if not param.fixed]
    
    def sliced_posterior(self, center=None, radius=None, n_max=100):
        """estimate 
        """
        distribs = [p.distributions for p in self.parameters if len(p.distributions) > 0]
        #TODO: check distributions for a modification time and cache value of sum
        sum_dist = distribs.pop(0).as_sog(center=center, radius=radius, embedding_space=self)
        while len(distribs) > 0:
            cur_dist = distribs.pop(0)
            cur_dist = cur_dist.as_sog(center=center, radius=radius, embedding_space=self)
            sum_dist = combine_sog_distributions(sum_dist, cur_dist, n_max=n_max)
        return sum_dist
    
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

class FixedParameterException(Exception):
    pass

class Parameter(ThimblesTable, Base):
    parameter_class = Column(String)
    __mapper_args__={
        "polymorphic_identity":"parameter",
        "polymorphic_on": parameter_class
    }
    
    fixed = Column(Boolean)
    propagate = Column(Boolean)
    _value = None
    
    #class attributes
    name = "base parameter class"
    scale = 1.0
    #step_scale = 1.0
    #derivative_scale = 1e-4
    #convergence_scale = 1e-2
    min=-np.inf
    max=np.inf
    
    def __init__(self, 
                 value=None,
                 fixed = False,
                 propagate=True,
                 ):
        self.free = free
        self.propagate=propagate
        #self.history = ValueHistory(self, history_max)   
    
    @property
    def value(self):
        if self._value is None:
            m_models = self.mapped_models
            if len(m_models) >= 1:
                if len(m_models) > 1:
                    print "warning parameter value regeneration is non-unique consider changing the model hierarchy to use cached paramters"
                mod = m_models[0]
                mod.fire() #should populate our self._value attribute
        return self._value
    
    @value.setter
    def value(self, value):
        self.set(value, clip=True, propagate=None)
    
    def get(self):
        return self.value
    
    def set(self, value, clip=True, propagate=None):
        if not self.free:
            raise FixedParameterException("attempted to set the value of a non-free parameter")
        if clip:
            value = np.clip(value, self.min, self.max)
        self._value = value
        if propagate is None:
            propagate = self.propagate
        if propagate:
            mods = []
            mod_set = set()
            mods.extend(self.models)
            while len(mods) > 0:
                mod = mods.pop(0)
                mod_set.add(mod)
                mod.fire()
                mods.extend([nmod for nmod in mod.output_p.models if nmod not in mod_set])
    
    def __repr__(self):
        val = None
        try:
            val = self.get()
        except:
            pass
        return "Parameter: name={}, value={}".format(self.name, val)
    
    @property
    def shape(self):
        return np.asarray(self.get()).shape
    
    def remember(self, value_id=None):
        self.history.remember(value_id=value_id)
    
    def revert(self, value_id, pop=False):
        self.history.revert(value_id=value_id, pop=pop)


#how to make a parameter subclass
#class classname(Parameter):
#    _id = Column(Integer, ForeignKey("Parameter._id"), primary_key=True)
#    _value = Column(Float)
#    __mapper_args__={
#        "polymorphic_identity": "classname"
#    }

