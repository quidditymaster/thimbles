import time
import numpy as np
from copy import copy

class ModelingError(Exception):
    pass

factory_defaults = {
"history": ValueHistory,
"scale":ParameterScale,
"distribution":NormalDeltaDistribution,
}

def parameterize(depends_on=None, 
                 free=False, 
                 factories=None,
                 factory_kwargs=None
                 ):
    """a decorator to turn getter methods of Model class objects 
    into Parameter objects.
    """
    if factories is None:
        factories = {}
    for factory_key in factory_defaults:
        factory = factories.get(factor_key)
        if factory is None:
            factory = factory_defaults[factory_key]
    def function_to_parameter(func):
        param=Parameter(
            func,
            depends_on=depends_on,
            free=free,
            factories=factories,
            factory_kwargs=factory_kwargs,
        )
        return param
    return function_to_parameter

class DeltaDistribution(object):
    
    def __init__(self, parameter=None):
        self.set_parameter(parameter)
    
    def set_parameter(self, parameter):
        self.parameter=parameter
    
    def set_variance(self, value):
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
    
    def __init__(self, variance=None, parameter=None):
        super(NormalDeltaDistribution, self).__init__(parameter)
        self.set_variance(variance)
    
    def set_variance(self, variance):
        self._variance = variance
    
    def realize(self):
        return np.random.normal(size=self._variance.shape)*self._variance
    
    def weight(self, offset):
        return self._variance**-2.0

class ParameterScale(object):
    
    def __init__(self, small_step=None, large_step=None, epsilon=0.01):
        """step sizes for a parameter,
        small_step: float or ndarray
          the scale of the smallest meaningful differences
        large_step: float or ndarray
          the scale over which changes in the parameter have
          large effects, this is important when non-linear
          effects are in play, even if the non-linearity
          is not directly dependent on this parameter
        epsilon: float or ndarray
          the fraction of a small step to use to evaluate derivatives.
        """
        self.small_step = small_step
        self.large_step = large_step
        self.epsilon = epsilon

class ValueHistory(object):
    
    def __init__(self, parameter, max_length=10):
        self.set_parameter(parameter)
        self._vals = []
        self.max_length = max_length
    
    def __len__(self):
        return len(self._vals)
    
    def set_parameter(self, parameter):
        self.parameter = parameter
    
    def remember(self):
        current_val = copy(self.parameter.value())
        self._vals.append(current_val)
        if len(self._vals) > self.max_length:
            self._vals.pop(0)
    
    @property
    def history(self):
        return self._vals
    
    @property
    def last(self):
        if len(self) > 0:
            return self._vals[-1]
        else:
            return None

class ConvergencePolicy(object):

    def __init__(self, abs_delta=1e-5):
        #self.frac_delta = frac_delta
        self.abs_delta = abs_delta
    
    def check_converged(self, parameter):
        last_val = parameter.history.last
        if val_stack is None:
            return False
        cur_val = parameter.value()
        diff = last_val - cur_val
        abs_diff = np.abs(diff)
        if np.max(abs_diff) < self.abs_delta:
            #if np.max(abs_diff/(np.abs(cur_val))
            return True
        return False

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
    
    @property
    def parameters(self):
        return self._parameters
    
    def calculate_derivatives(self):
        self.derivatives = {}
        for dy_param_id, dy_param in self._parameters.items():
            for dx_id, dx_param in dy_param.depends_on:
                if dx_param.is_base:
                    cpar_val = dx_param.value()

class Parameter(object):
    
    def __init__(self, getter, depends_on, free, factories, factory_kwargs):
        self._getter=getter
        if depends_on is None:
            depends_on=[]
        self.depends_on=depends_on
        #self.step=step
        #self.epsilon=epsilon
        history_factory = factories["history"]
        hist_kwargs = factory_kwargs.get("history", {})
        self.history=history_factory(self, **hist_kwargs)
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
    
    def remember(self):
        self.history.remember()
    
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
    
    def __init__(self, value=None, predicted_by=None, predicts=None):
        if predicted_by is None:
            predicted_by = []
        self.predicted_by=predicted_by
        if predicts is None:
            predicts = []
        self.predicts=predicts
        self.value = value
    
    def update(self):
        pass
    
    def predict(self):
        pass
    
    def __lshift__(self, value):
        if not isinstance(value, Parameter):
            raise ModelingError("predictors must be Parameter instances")
        self.predicted_by.append(value)
    
    def __rshift__(self, value):
        if not isinstance(value, Parameter):
            raise ModelingError("predictors must be Parameter instances")
        self.predicts.append(value)

def find_base(param):
    if param.is_free:
        if param.is_base:
            return [param]
        else:
            base_ps = []
            for sub_param_attr in param.depends_on:
                sub_base = find_base(param.model.parameters[sub_param_attr])
                base_ps.extend(sub_base)
    else:
        return []

class Fitter(object):
    
    def __init__(self, anchors):
        self.anchors = anchors
    
    def add_anchor(self, anchor):
        self.anchors.append(anchors)
    
    def iterate(self):
        for anchor in self.anchors:
            anchor.update()
        #TODO: build a derivative matrix for each predicted_parameter as a function of changes in each associated free base parameter and solve.
        predict_basis = {}
        for anchor in self.anchors:
            for pred_param in anchor.predicts:
                predict_basis[(anchor, pred_param)] = find_base(pred_param)
        predict
        
    def converge(self, max_iter=100):
        #check convergence on anchors? or check convergence on parameters? 
        raise NotImplementedError()
    
