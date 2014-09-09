import time
import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
from copy import copy

class ModelingError(Exception):
    pass

#factory_defaults = {
#"history": ValueHistory,
#"scale":ParameterScale,
#"distribution":NormalDeltaDistribution,
#}

def parameter(free=False, min_step=None, max_step=None, start_damp=10.0, epsilon=0.01, min=None, max=None):
    """a decorator to turn getter methods of Model class objects 
    into Parameter objects.
    """
    #if factories is None:
    #    factories = {}
    #for factory_key in factory_defaults:
    #    factory = factories.get(factor_key)
    #    if factory is None:
    #        factory = factory_defaults[factory_key]
    def function_to_parameter(func):
        param=Parameter(
            func,
            free=free,
            min_step=min_step,
            max_step=max_step,
            start_damp=start_damp,
            epsilon=epsilon,
            min=min,
            max=max
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

mult_func = lambda x, y: x*y

def flat_size(shape_tup):
    if shape_tup == tuple():
        return 1
    return reduce(mult_func, shape_tup)

class Model(object):
    
    def __init__(self):
        self.attach_parameters()
    
    def attach_parameters(self):
        self._parameters = []
        for attrib in dir(self):
            val = getattr(self, attrib)
            if isinstance(val, Parameter):
                val.set_model(self)
                self._parameters.append(val)
                val.validate()
    
    @property
    def parameters(self):
        return self._parameters
    
    @property
    def free_parameters(self):
        return [param for param in self.parameters if param.is_free]
    
    def get_pvec(self):
        pvals = [np.asarray(p.get()).reshape((-1,)) for p in self.free_parameters]
        return np.hstack(pvals)
    
    def parameter_index(self, parameter):
        return self.parameters.index(parameter)
    
    def set_pvec(self, pvec):
        parameters = self.free_parameters
        pshapes = [p.shape for p in parameters]
        nvals = [flat_size(pshape) for pshape in pshapes]
        break_idxs = np.cumsum(nvals)[:-1]
        flat_params = np.split(pvec, break_idxs)
        for p_idx in range(len(parameters)):
            param = parameters[p_idx]
            pshape = pshapes[p_idx]
            flat_val = flat_params[p_idx]
            if pshape == tuple():
                param.set(float(flat_val))
            else:
                param.set(flat_val.reshape(pshape))
    
    def parameter_expansion(self, input_vec, **kwargs):
        parameters = self.free_parameters
        deriv_vecs = []
        pval_delta = 0.0001
        for p in parameters:
            pval = p.get()
            #TODO: use the parameters own epsilon scale
            if p.shape == tuple():
                p.set(pval+pval_delta)
                plus_d = self(input_vec, **kwargs)
                p.set(pval-pval_delta)
                minus_d = self(input_vec, **kwargs)
                deriv = (0.5*(plus_d-minus_d)/pval_delta)
                deriv_vecs.append(deriv)
                #don't forget to reset the parameters back to the start
                p.set(pval)
            else:
                raise NotImplementedError
                cur_n_param =  flat_size(pval.shape)
                delta_vals = np.eye(cur_n_param)*pval_delta
                p.set()
        pexp_mat = scipy.sparse.bmat(deriv_vecs).transpose()
        return pexp_mat
    
    def as_linear_op(self, input_vec, **kwargs):
        return IdentityPlaceHolder()
    
    def parameter_damping(self, input_vec, **kwargs):
        parameters = self.free_parameters
        damp_vecs = []
        for p in parameters:
            damp_val = np.asarray(p.damp)
            if damp_val.shape == tuple():
                pshape = p.shape
                if not pshape == tuple():
                    damp_val = np.repeat(damp_val, flat_size(pshape))*damp_val
            damp_vecs.append(damp_val)
        damp_weights = np.hstack(damp_vecs)
        return np.zeros(damp_weights.shape), damp_weights

class Parameter(object):
    
    def __init__(self, getter, free, start_damp, min_step, max_step, min, max, epsilon):#, factories, factory_kwargs):
        self._getter=getter
        #history_factory = factories["history"]
        #hist_kwargs = factory_kwargs.get("history", {})
        #self.history=history_factory(self, **hist_kwargs)
        #self.model=None
        
        self.damp=start_damp
        self.epsilon=epsilon
        self.min = min
        self.max = max
        self.max_step = max_step
        self.min_step = min_step
        self._free=free
        self._dist=None
        self._setter = None        
    
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
    def shape(self):
        return np.asarray(self.get()).shape
    
    @property
    def is_free(self):
        return self._free
    
    def remember(self):
        self.history.remember()
    
    def set_model(self, model):
        self.model = model
    
    def setter(self, setter):
        self._setter=setter
        return setter
    
    def set(self, value):
        self._setter(self.model, value)
    
    def get(self):
        return self._getter(self.model)
    
    def validate(self):
        if self.model is None:
            raise ModelingError("parameter.model is None")
    
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
        if len(target_inv_covar.shape) == 1:
            npts = len(target_inv_covar)
            target_inv_covar = scipy.sparse.dia_matrix((target_inv_covar, 0), shape=(npts, npts))
        self.target_inv_covar = target_inv_covar
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
            if model == self.models[i]:
                return i
        raise Exception("model not found!")
    
    def get_model_inputs(self, i):
        if i == 0:
            input_val = self.first_input
        else:
            input_val = self._result_chain[i-1]
        kwargs = self.kw_inputs[i]
        return input_val, kwargs
    
    def fit_matrix(self, model):
        md_idx = self.model_index(model)
        after_matrix = self.after_matrix(md_idx)
        inp, kw = self.get_model_inputs(md_idx)
        delta_matrix = model.parameter_expansion(inp, **kw)
        return after_matrix*delta_matrix
    
    def parameter_damping(self, model):
        md_idx = self.model_index(model)
        inp, kw = self.get_model_inputs(md_idx)
        return model.parameter_damping(inp, **kw)
    
    def __call__(self):
        for model_idx in range(self.n_models):
            input_value, kwargs = self.get_model_inputs(model_idx)
            output_value = self.models[model_idx](input_value, **kwargs)
            self._result_chain[model_idx] = output_value
        return self._result_chain[-1]


class FitPolicy(object):
    
    def __init__(self):
        pass

class FitState(object):
    
    def __init__(self):
        pass

class FitTransitionPolicy(object):
    
    def __self__(self):
        pass

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
    
    def stitched_fit_matrices(self, chains, model):
        fit_mats = []
        for chain in chains:
            fit_mats.append([chain.fit_matrix(model)])
        
        #stitch together a target vector
        targ_deltas = []
        for chain in chains:
            targ_dat = chain.target_data
            mod_dat = chain()
            deltas = targ_dat - mod_dat
            targ_deltas.append(deltas)
        
        #build the data inverse variance matrix
        inv_covars = []
        for chain in chains:
            inv_covars.append(chain.target_inv_covar)
        
        #add damping matrices
        n_params = fit_mats[0][0].shape[1]
        for chain in chains:
            fit_mats.append([scipy.sparse.identity(n_params, dtype=float)])
            damping_target, damping_weight = chain.parameter_damping(model)
            targ_deltas.append(damping_target)
            inv_covars.append(scipy.sparse.dia_matrix((damping_weight, 0), shape=(n_params, n_params)))
        
        fm = scipy.sparse.bmat(fit_mats)
        td = np.hstack(targ_deltas)
        ic = scipy.sparse.block_diag(inv_covars)
        return fm, td, ic
    
    def iterate(self, models):
        #accumulate the linear fit expansions for all models
        lhs_list, rhs_list = [], []
        for model_idx in range(len(models)):
            model = models[model_idx]
            relevant_chains = self.model_to_chains.get(model)
            if relevant_chains is None:
                raise Exception("model {} not part of this modeler's chain sequence".format(model))
            fit_mat, target_vec, error_inv_covar = self.stitched_fit_matrices(relevant_chains, model)
            #trans_mat = fit_mat.transpose()
            rhs = error_inv_covar*target_vec
            lhs = error_inv_covar*fit_mat
            lhs_list.append(lhs)
            rhs_list.append(rhs)
        
        #build the full fit and run it
        full_lhs = scipy.sparse.block_diag(lhs_list)
        full_rhs = np.hstack(rhs_list)
        #if full_lhs.shape == (1, 1):
        #    numerator = float(full_rhs)
        #    denominator = full_lhs.todense()[0, 0]
        #    fit_result = np.atleast_1d(numerator/denominator)
        #else:
        fit_result = scipy.sparse.linalg.lsqr(full_lhs, full_rhs)[0]
        
        #break up the results by model and assign
        lb = 0 
        ub = 0
        for model_idx in range(len(models)):
            model = models[model_idx]
            old_pvec = model.get_pvec()
            n_params = len(old_pvec)
            ub += n_params
            new_pvec = old_pvec + fit_result[lb:ub]
            model.set_pvec(new_pvec)
            lb = ub
