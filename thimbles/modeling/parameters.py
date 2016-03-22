import time
import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
from copy import copy

from thimbles.sqlaimports import *
from thimbles.thimblesdb import ThimblesTable, Base
from sqlalchemy.orm.collections import attribute_mapped_collection
from sqlalchemy.orm.collections import collection
from functools import reduce
from .associations import InformedContexts, NamedContexts

mult_func = lambda x, y: x*y
def flat_size(shape_tup):
    if shape_tup == tuple():
        return 1
    return reduce(mult_func, shape_tup)


class Parameter(ThimblesTable, Base):
    parameter_class = Column(String)
    __mapper_args__={
        "polymorphic_identity":"parameter",
        "polymorphic_on": parameter_class
    }
    _value = None
    models = relationship(
        "InputAlias",
        collection_class = InformedContexts,
    )
    distributions = relationship(
        "DistributionAlias",
        collection_class = NamedContexts,
    )   
    
    _setting_callbacks = None
    _invalidation_callbacks = None
    
    def __init__(self, value=None):
        self._value = value
    
    def add_callback(self, cb_name, cb_function, cb_type="invalid"):
        """add a function to be called whenever this parameters value
        is set or invalidated depending on cb_type argument.
        """
        if cb_type == "set":
            if self._setting_callbacks is None:
                self._setting_callbacks = {}
            self._setting_callbacks[cb_name] = cb_function
        elif cb_type == "invalid":
            if self._invalidation_callbacks is None:
                self._invalidation_callbacks = {}
            self._invalidation_callbacks[cb_name] = cb_function
        else:
            raise ValueError("cb_type must be one of {}".format(["set" "invalid"]))
    
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
        parameter_front = [mod.output_p for mod in self.models if (not mod.output_p is None)]
        while len(parameter_front) > 0:
            param = parameter_front.pop(0)
            param.invalidate()
            #find all the downstream parameters
            for mod in param.models:
                out_p = mod.output_p
                #check that the model actually maps to something.
                if not out_p is None:
                    #if parameter is not already invalid add it to the queue.
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
    
    def execute_invalidation_callbacks(self):
        if self._invalidation_callbacks is None:
            self._invalidation_callbacks = {}
        for cb_name in self._invalidation_callbacks:
            cb_func = self._invalidation_callbacks[cb_name]
            cb_func()
    
    def invalidate(self,):
        if not self.invalid():
            self._value = None
            self.execute_invalidation_callbacks()
    
    def get(self,):
        if self.invalid():
            self.fire_upstream()
        return self._value
    
    def execute_setting_callbacks(self):
        if self._setting_callbacks is None:
            self._setting_callbacks = {}
        for cb_name in self._setting_callbacks:
            cb_func = self._setting_callbacks[cb_name]
            cb_func(self._value)
    
    def set(self, value):
        self.invalidate_downstream()
        self._value = value
        self.execute_setting_callbacks()
    
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

