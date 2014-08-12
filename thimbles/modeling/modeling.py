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


class IdentityPlaceHolder(object):
    
    def __mul__(self, other):
        return other

class ModelChain(object):
    
    def __init__(self, models, target_data, target_inv_covar, first_input=None, kw_inputs=None, delay_execution=False):
        self.models = models
        self.n_models = len(models)
        self.target_data = target_data
        if len(target_inv_var.shape) == 1:
            npts = len(target_inv_var)
            target_inv_var = scipy.sparse.dia_matrix((target_inv_var, 0), shape=(npts, npts))
        self.target_inv_covar = target_inv_var
        if kw_inputs is None:
            kw_inputs = [{} for i in range(self.n_models)]
        self.kw_inputs = kw_inputs
        self.first_input = first_input
        
        self._result_chain = [None for i in range(self.n_models)]
        if not delay_execution:
            self()
    
    def after_matrix(self, i):
        if i == self.n_models-1:
            return IdentityPlaceHolder()
        inp, kw = self.get_model_inputs(i+1)
        lop = self.models[i+1].as_linear_op(inp, **kw)
        for model_idx in range(i+2, self.n_models):
            inp, kw = self.get_model_inputs(model_idx)
            further_lop = self.models[model_idx].as_linear_op(inp, **kw)
            lop = further_lop*lop
        return lop
    
    def model_index(self, model):
        for i in range(self.n_models):
            if model == self.models[i]
        return i
    
    def get_model_inputs(self, i):
        if i == 0:
            input_val = self.first_input
        else:
            input_val = self._result_chain[i-1]
        kwargs = self.kw_inputs[i]
        return input_val, kwargs
    
    def fit_matrix(self, model):
        md_idx = self.model_index(model)
        after_matrix = chain.after_matrix(md_idx)
        inp, kw = get_model_inputs(md_idx)
        delta_matrix = model.parameter_expansion(inp, **kw)
        return after_matrix*delta_matrix
    
    def __call__(self):
        for model_idx in range(self.n_models):
            if model_idx == 0:
                input_value = self.first_input
            else:
                input_value = self._result_chain[model_idx-1]
            kwargs= self.kw_inputs[model_idx]
            output_value = self.models[model_idx](input_value, **kwargs)
            self._result_chain[model_idx] = output_value
        return self._result_chain[-1]

class Modeler(object):
    
    def __init__(self):
        self.chains = []
        self.model_to_chains = {}
    
    def add_chain(self, chain):
        self.chains.append(chain)
        for model in chain.models:
            chain_res = self.model_to_chains.get(model)
            if chain_res is None:
                chain_res = []
            self.model_to_chains[model] = chain_res
            chain_res.append(chain)
    
    def stitched_fit_matrices(self, chains):
        fit_mats = []
        for chain in chains:
            fit_mats.append([chain.fit_matrix(model)])
        return scipy.sparse.bmat(fit_mats)
    
    def stitched_target_vectors(self, chains):
        targ_vecs = []
        for chain in chains:
            targ_vecs.append(chain.target_data)
        return np.hstack(targ_vecs)
    
    def stitched_inv_covar_matrices(self, chains):
        inv_covars = []
        for chain in chains:
            inv_covars.append(chain.target_inv_covar)
        return scipy.sparse.block_diag(inv_covars)
    
    def iterate(self, model):
        relevant_chains = self.model_to_chains(model)
        fit_mat = self.stitched_fit_matrices(relevant_chains)
        error_inv_covar = self.stitched_inv_covar_matrices(relevant_chains)
        target_vec = self.stitched_target_vectors(self, relevant_chains)
        trans_mat = fit_mat.transpose()
        ata_inv = trans_mat*(error_inv_covar*fit_mat)
        rhs = trans_mat*(error_inv_covar*target_vec)
        fit_result = scipy.sparse.lsqr(ata_inv, rhs)
        model.set_pvec(fit_result0])
