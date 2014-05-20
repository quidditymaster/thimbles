import time
import numpy as np

class ModelingError(Exception):
    pass

def parameterize(depends_on=None, 
                 free=False, 
                 step=None,
                 epsilon=1e-8,):
    """a decorator to turn getter methods of Model class objects 
    into Parameter objects.
    """
    
    def function_to_parameter(func):
        param=Parameter(func,
                        depends_on=depends_on,
                        free=free,
                        )
                        step=step,
                        epsilon=epsilon)
        return param
    return function_to_parameter

class DeltaDistribution(object):
    
    def __init__(self, parameter=None):
        self.parameter=parameter
    
    def set_parameter(self, parameter):
        self.parameter=parameter
    
    def set_width(self, value):
        raise NotImplementedError("abstract class; implement for subclasses")
    
    def realize(self):
        """return an offset sampled from this distribution
        """
        raise NotImplementedError("abstract class; implement for subclasses")
    
    def weight(self, offset):
        """returns a weight for use in iterated 
        """
        raise NotImplementedError("abstract class; implement for subclasses")

class NormalDeltaDistribution(DeltaDistribution):
    
    def __init__(self, sigma, parameter=None):
        super(NormalDeltaDistribution, self).__init__(parameter)
        self._inv_var = sigma**-2.0
    
    def set_width(self, sigma):
        self._inv_var = sigma**-2.0
    
    def realize(self):
        return np.random.normal(size=self._inv_var.shape)*self._inv_var**-2.0
    
    def weight(self, offset):
        return self._inv_var

class Model(object):
    
    def __init__(self):
        self.attach_parameters()
    
    def attach_parameters(self):
        self._parameters = {}
        for attrib in dir(self):
            val = getattr(self, attrib)
            if isinstance(val, Parameter):
                val.set_model(self)
                self._parameters[attrib]=val
                val.validate()
    
    def calculate_derivatives(self):
        self.derivatives = {}
        for dy_param_id, dy_param in self._parameters.items():
            for dx_id, dx_param in dy_param.depends_on:
                if dx_param.is_base:
                    cpar_val = dx_param.value()

class Parameter(object):
    
    def __init__(self, getter, depends_on, free, step, epsilon):
        self._getter=getter
        if depends_on is None:
            depends_on=[]
        self.depends_on=depends_on
        self.step=step
        self.epsilon=epsilon
        self.model=None
        
        self._free=free
        self._dist=None
        self._setter = None        
        self._last_valuated = -np.inf
        self._last_value = None
    
    @property
    def dist(self):
        return self._dist
    
    @dist.setter
    def dist(self, value):
        if isinstance(value, DeltaDistribution):
            self._dist=value
            self._dist.set_parameter(self)
        else:
            asndarr = np.asarray(value, dtype=float)
            if asndarr.shape == tuple():
                asndarr=np.ones(self.value().shape)*asndarr
            self._dist = NormalDeltaDistribution(asndarr, parameter=self)
    
    @property
    def is_base(self):
        return self.is_free and self.is_independent
    
    @property
    def is_free(self):
        return self._free
    
    @property
    def is_independent(self):
        return self.depends_on == []
    
    @property
    def is_settable(self):
        return not self._setter is None
    
    def set_model(self, model):
        self.model = model
    
    def setter(self, setter):
        self._setter=setter
        return self._setter
    
    def set(self, value):
        self._setter(self.model, value)
    
    def validate(self):
        if self.model is None:
            raise ModelingError("parameter.model is None")
        if (not self.is_settable) and self.is_free:
            raise ModelingError("parameter cannot be free with no setter")
        if self.is_settable and (not self.is_independent):
            raise ModelingError("parameter must be independent if it has a set method")
    
    def value(self):
        return self._getter(self.model)
    
    def weight(self, offset=None):
        return self.dist.weight(offset)


class Anchor:
    
    def __init__(self):
        self.info_in=[]
        self.info_out=[]
    
    def add_info_in(self, param):
        self.info_in.append(predicate)
    
    def add_info_out(self, param):
        self.info_out.append(param)
    
    def value(self):
        for param in self.info_in:
            

class Fitter(object):
    
    def __init__(self):
        self._anchors={}
    
    def info_in(self, anchor_tag, parameter):
        anchor = self._anchors.get(anchor_tag)
        if anchor is None:
            anchor=Anchor()
            self._anchors[anchor_tag] = anchor
        anchor.add_info_in(parameter)
    
    def info_out(self, anchor_tag, parameter):
        anchor = self._anchors.get(anchor_tag)
        if anchor is None:
            anchor=Anchor()
            self._anchors[anchor_tag] = anchor
        anchor.add_info_out(parameter)
    
    
