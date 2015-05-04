import time
import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
from copy import copy

from thimbles.sqlaimports import *
from thimbles.thimblesdb import ThimblesTable, Base
from sqlalchemy.orm.collections import attribute_mapped_collection
from functools import reduce

mult_func = lambda x, y: x*y
def flat_size(shape_tup):
    if shape_tup == tuple():
        return 1
    return reduce(mult_func, shape_tup)


class Slice(ThimblesTable, Base):
    start = Column(Integer)
    stop = Column(Integer)
    step = Column(Integer)
    
    def __init__(self, start=None, stop=None, step=None):
        self.start=start
        self.stop=stop
        self.step=step

class ParameterView(object):
    """A class that emulates a Parameter object
    and provides a sliced view into that parameter.
    
    A parameter view will hash the same as its parent parameter
    and so may be used interchangeably with its parent parameter as
    a key in a dictionary or as a unique element in a set.
    """
    
    def __init__(self, parameter, slice_):
        self.parameter = parameter        
        self.slice_ = slice_
    
    def __getattr__(self, attr):
        return getattr(self.parameter, attr)
    
    @property
    def value(self):
        if self.slice_ is None:
            return self.parameter
        else:
            return self.parameter[self.slice_]
    
    @value.setter
    def value(self, value):
        if self.slice_ is None:
            self.parameter = value
        else:
            self.parameter[self.slice_] = value
    
    def __hash__(self):
        return hash(self.parameter)
    
    def __eq__(self, other):
        return self.parameter is other
    
    def _get_sub_slice(self, slice_=None):
        if slice_ is None:
            return self.slice_
        else:
            start = slice_.start - self.slice_.start
            stop = slice_.stop - self.slice_.start
            step = slice_.step*self.slice_.step
            return slice(start, stop, step)
    
    def invalid(self, slice_=None):
        slice_ = self._get_sub_slice(slice_)
        return self.parameter.invalid(slice_=slice_)
    
    def invalidate(self, slice_=None):
        slice_ = self._get_sub_slice(slice_)
        self.parameter.invalidate(slice_=slice_)


parameter_vector_assoc = sa.Table(
    "parameter_vector_assoc", 
    Base.metadata,
    Column("vector_id", Integer, ForeignKey("ParameterVector._id")),
    Column("parameter_id", Integer, ForeignKey("Parameter._id")),
)

class ParameterVector(ThimblesTable, Base):
    parameters = relationship("Parameter", secondary=parameter_vector_assoc)
    _group_id = Column(Integer, ForeignKey("ParameterGroup._id"))
    name = Column(String)
    
    def __init__(self, name, parameters=None):
        self.name = name
        if parameters is None:
            parameters = []
        self.parameters = parameters
    
    def __getitem__(self, index):
        return self.parameters[index]

class ParameterGroup(ThimblesTable, Base):
    vectors = relationship(
        "ParameterVector",
        collection_class=attribute_mapped_collection("name"),
        backref="group",
        #cascade="all, delete-orphan",
    )
    
    def __init__(self, **parameters):
        for param_name in parameters:
            named_p = parameters[param_name]
            if isinstance(p, list):
                for p in named_p:
                    self.add_parameter(param_name, p)
            else:
                raise ValueError("unable to parse input to ParameterGroup should be instantiated as ParameterGroup(name1=[p1, p2], name2=[p3], ...)")
    
    def add_parameter(self, name, parameter,):
        pvec = self.vectors.get(name)
        if pvec is None:
            pvec = ParameterVector(name)
        pvec.parameters.append(parameter)
        self.vectors[name] = pvec
    
    @property
    def parameters(self):
        parameters = []
        for vec_name in self.vectors :
            parameters.extend(self.vectors[vec_name].parameters)
        return parameters
    
    def __getitem__(self, index):
        return self.vectors[index].parameters
    
    @property
    def free_parameters(self):
        return [param for param in self.parameters if not param.fixed]
    
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
        elif isinstance(attr, str):
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
    
    def get_pdict(self, value_replacements=None, attr=None, free_only=True, name_as_key=False):
        
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
        pdict = dict(list(zip(keys, values))) 
        return pdict
    
    def set_pdict(self, val_dict, attr=None):
        for p in val_dict:
            if attr is None:
                p.set(val_dict[p])
            else:
                setattr(p, attr, val_dict[p])

class InformedModels(object):
    
    def __init__(self):
        self.models = []
    
    def append(self, param_alias):
        self.models.append(param_alias.model)
    
    def remove(self, param_alias):
        self.models.remove(param_alias.model)
    
    def __len__(self):
        return len(self.models)
    
    def __getitem__(self, index):
        return self.models[index]
    
    def __iter__(self):
        for mod in self.models:
            yield mod


class Parameter(ThimblesTable, Base):
    parameter_class = Column(String)
    __mapper_args__={
        "polymorphic_identity":"parameter",
        "polymorphic_on": parameter_class
    }
    _value = None
    models = relationship(
        "InputAlias",
        collection_class = InformedModels
    )
    
    def __init__(self, value=None,):
        self._value = value
    
    def fire_upstream(self):
        """fire the model immediately upstream from this parameter.
        Often this will require also causing the models upstream from
        it to fire as well and so on causing a cascade.
        """
        m_models = self.mapped_models
        if len(m_models) >= 1:
            if len(m_models) > 1:
                print("warning parameter value regeneration is non-unique consider changing the model hierarchy to use cached parameter values.")
            mod = m_models[0]
            mod.fire() #should populate our self._value attribute
    
    def invalidate_downstream(self):
        """run through all parameters downstream of this one
        marking them as invalid, so that subsequent calls asking
        for their value will trigger an update cascade.
        """
        parameter_front = [self]
        while len(parameter_front) > 0:
            param = parameter_front.pop(0)
            param.invalidate()
            for mod in param.models:
                out_p = mod.output_p
                if not out_p is None:
                    if not out_p.invalid():
                        parameter_front.append(out_p)
    
    @property
    def value(self):
        return self.get()
    
    @value.setter
    def value(self, value):
        self.set(value,)
    
    def invalid(self,):
        return self._value is None
    
    def invalidate(self,):
        self._value = None
    
    def __getitem__(self, index):
        return self.value[index]
    
    def __setitem__(self, index, value):
        self._value[index] = value
    
    def get(self,):
        if self.invalid():
            self.fire_upstream()
        return self._value
    
    def set(self, value, slice_=None):
        self.invalidate_downstream()
        if slice_ is None:
            self._value = value
        else:
            self._value[slice_] = value
    
    @property
    def shape(self):
        return np.asarray(self.get()).shape
    

#how to make a parameter subclass
#class classname(Parameter):
#    _id = Column(Integer, ForeignKey("Parameter._id"), primary_key=True)
#    _value = Column(Float)
#    __mapper_args__={
#        "polymorphic_identity": "classname"
#    }

