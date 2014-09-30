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


def parameter(free=False, scale=1.0, name=None, step_scale=1.0, derivative_scale=0.0001, convergence_scale=0.01, min=-np.inf, max=np.inf, history_max=10):
    """a decorator to turn getter methods of Model class objects 
    into Parameter objects.
    """
    def function_to_parameter(func):
        param=Parameter(
            func,
            free=free,
            name=name,
            scale=scale,
            step_scale=step_scale,
            derivative_scale=derivative_scale,
            convergence_scale=convergence_scale,
            min=min,
            max=max,
            history_max=history_max,
        )
        return param
    return function_to_parameter

class ParameterDistribution(object):
    
    def __init__(self, parameter=None):
        self.parameter = parameter
    
    def log_likelihood(self, value):
        raise NotImplementedError
    
    #def set_variance(self, value):
    #    raise NotImplementedError("abstract class; implement for subclasses")
    #
    #def realize(self):
    #    """return an offset sampled from this distribution
    #    """
    #    raise NotImplementedError("abstract class; implement for subclasses")
    #
    #def weight(self, offset):
    #    """returns a weight for use in iterated 
    #    """
    #    raise NotImplementedError("abstract class; implement for subclasses")

class NormalDeltaDistribution(ParameterDistribution):
    
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
        self.named_vals = {}
    
    def __len__(self):
        return len(self._vals)
    
    def set_parameter(self, parameter):
        self.parameter = parameter
    
    def remember(self, val_name=None):
        current_val = copy(self.parameter.get())
        if val_name is None:
            self._vals.append(current_val)
            if len(self._vals) > self.max_length:
                self._vals.pop(0)
        else:
            self.named_vals[val_name] = current_val
    
    @property
    def values(self):
        return self._vals
    
    @property
    def last(self):
        if len(self) > 0:
            return self._vals[-1]
        else:
            return None
    
    def revert(self, val_name=None, pop=False):
        if val_name is None:
            if pop:
                last_val = self._vals.pop()
            else:
                last_val = self.last
            self.parameter.set(last_val)
        else:
            self.parameter.set(self.named_vals[val_name])


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


class Model(ParameterGroup):
    
    def __init__(self):
        self.attach_parameters()
    
    def attach_parameters(self):
        self._parameters = []
        for attrib in dir(self):
            try:
                val = getattr(self, attrib)
            except Exception:
                continue
            if isinstance(val, Parameter):
                #import pdb; pdb.set_trace()
                val.set_model(self)
                self._parameters.append(val)
                if val.name is None:
                    val.name = attrib
                val.validate()
        
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
        return IdentityPlaceHolder() #implicitly allowing additive model passthrough

class ParameterPrior(ParameterGroup, ParameterDistribution):
    
    def __init__(self, parameters):
        pass
    

class Parameter(object):
    
    def __init__(self, 
                 getter, 
                 free,
                 name,
                 scale,
                 step_scale,
                 derivative_scale,
                 convergence_scale,
                 min, 
                 max, 
                 history_max,
                 ):
        self._getter=getter
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
        self._setter = None
        self._expander = None
        
        self.history = ValueHistory(self, history_max)   
    
    def __repr__(self):
        val = None
        try:
            val = self.get()
        except:
            pass
        return "Parameter: {}={}".format(self.name, val)
    
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
        min_respected = np.atleast_1d(value) >= self.min
        max_respected = np.atleast_1d(value) <= self.max
        self._setter(self.model, np.clip(value, self.min, self.max)) 
        if np.all(min_respected*max_respected):
            return True
        else:
            return False
    
    def expander(self, expander):
        self._expander=expander
        return expander
    
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

class IdentityPlaceHolder(object):
    
    def __mul__(self, other):
        return other

class DataRootedModelTree(object):
    
    def __init__(self, models, target_data, data_weight, first_input=None, kw_inputs=None, delay_execution=False):
        self.models = models
        self.n_models = len(models)
        self.target_data = target_data
        self.data_weight = data_weight
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
    
    #def parameter_damping(self, model):
    #    md_idx = self.model_index(model)
    #    inp, kw = self.get_model_inputs(md_idx)
    #    return model.parameter_damping(inp, **kw)
    
    def __call__(self):
        for model_idx in range(self.n_models):
            input_value, kwargs = self.get_model_inputs(model_idx)
            output_value = self.models[model_idx](input_value, **kwargs)
            self._result_chain[model_idx] = output_value
        return self._result_chain[-1]

class FitPolicy(object):
    
    def __init__(self, model_network=None, fit_states=None, iteration_callback=None, transition_callback=None, finish_callback=None, max_state_iter=100, max_transitions=100):
        self.max_state_iter = max_state_iter
        self.max_transitions = max_transitions
        self.iteration_callback=iteration_callback
        self.transition_callback=transition_callback
        self.finish_callback=finish_callback
        if fit_states is None:
            fit_states = [FitState(model_network)]
        self.transition_map = {}
        for fs_idx in range(len(fit_states)):
            if isinstance(fit_states[fs_idx], FitState):
                cur_state = fit_states[fs_idx]
                if fs_idx == len(fit_states)-1:
                    next_state = None
                elif isinstance(fit_states[fs_idx+1], FitState):
                    next_state = fit_states[fs_idx+1]
                else:
                    next_state = fit_states[fs_idx+1][1]
            else:
                cur_state, next_state = fit_states[fs_idx]
            transition_list = self.transition_map.get(cur_state, [])
            if not next_state is None:
                transition_list.append(next_state)
            self.transition_map[cur_state] = transition_list
        self.current_fit_state = fit_states[0] 
        self.fit_states = [self.current_fit_state]
        for available_trans in self.transition_map.values():
            self.fit_states.extend(available_trans)
        self.set_model_network(model_network)
    
    def set_fit_state(self, fit_state):
        self.current_fit_state = fit_state
    
    def set_model_network(self, model_network):
        self.model_network = model_network
        if self.model_network is None:
            return
        for fit_state in self.fit_states:
            fit_state.set_model_network(model_network)
    
    def check_model_network(self):
        if self.model_network is None:
            raise ModelingError("model network not set")
    
    def iterate(self):
        self.check_model_network()
        iter_res =  self.current_fit_state.iterate()
        if not self.iteration_callback is None:
            self.iteration_callback(self.current_fit_state)
        return iter_res
    
    def converge(self):
        self.check_model_network()
        total_iter_num = 0
        available_transitions = copy(self.transition_map)
        for trans_num in range(self.max_transitions):
            fs_converged = False
            self.current_fit_state.setup()
            for iter_idx in range(self.max_state_iter):
                #import pdb; pdb.set_trace()
                self.current_fit_state.iter_setup()
                fs_converged = self.iterate()
                self.current_fit_state.iter_cleanup()
                total_iter_num += 1
                if fs_converged:
                    print "fit state converged"
                    break
            if not fs_converged:
                print "warning max_iter exceded"
            self.current_fit_state.cleanup()
            if not self.transition_callback is None:
                self.transition_callback(self.current_fit_state)
            print "attempting transition to next fit state"
            #import pdb;pdb.set_trace()
            transition_options = available_transitions.get(self.current_fit_state, [])
            if transition_options == []:
                print "no next fit state found assuming completion"
                if not self.finish_callback is None:
                    self.finish_callback(self.current_fit_state) 
                break
            else:
                self.current_fit_state = transition_options.pop(0)

class FitState(ParameterGroup):
    
    def __init__(self, model_network=None, models=None, trees=None, clamping_factor=5.0, clamping_nu=1.4, max_clamping=1000.0, alpha=2.0, beta=2.0, max_iter=10, max_reweighting_iter=10, setup_func=None, iter_setup_func=None, iter_cleanup_func=None, cleanup_func=None):
        self.models = models
        self.trees = trees
        self.set_model_network(model_network)
        
        self.iter_num = 0
        self.max_iter = max_iter
        self.max_reweighting_iter = max_reweighting_iter
        self.clamping_factor=clamping_factor
        self.clamping_nu = clamping_nu
        assert self.clamping_nu > 1.0
        self.max_clamping = max_clamping
        self.alpha=alpha
        self.beta=beta
        self.converged = False
        self.log_likelihood = np.inf
        self.setup_func = setup_func
        self.cleanup_func = cleanup_func
        self.iter_setup_func = iter_setup_func
        self.iter_cleanup_func = iter_cleanup_func
    
    def setup(self):
        if not self.setup_func is None:
            self.setup_func(self)
    
    def iter_setup(self):
        if not self.iter_setup_func is None:    
            self.iter_setup_func(self)
    
    def iter_cleanup(self):
        if not self.iter_cleanup_func is None:
            self.iter_cleanup_func(self)
    
    def cleanup(self):
        if not self.cleanup_func is None:
            self.cleanup_func(self)
    
    def set_model_network(self, model_network):
        self.model_network = model_network
        if self.model_network is None:
            return
        if self.trees is None:
            self.trees = model_network.trees
            self.model_to_tree_idxs = {}
            for tree_idx in range(len(self.trees)):
                tree = self.trees[tree_idx]
                for model in tree.models:
                    mod_list = self.model_to_tree_idxs.get(model, [])
                    mod_list.append(tree_idx)
                    self.model_to_tree_idxs[model] = mod_list
        if self.models is None:
            models = self.model_to_tree_idxs.keys()
        self.update_grouped_parameters()
    
    def update_grouped_parameters(self):
        self._parameters = []
        if not self.models is None:
            for model in self.models:
                self._parameters.extend(model.parameters)
    
    #def get_pvec(self, attr=None, free_only=True):
    #    """get a vectorized version of parameter.attr across the models fit by this
    #    fit state.
    #    """
    #    vec_collection = []
    #    for model in self.models:
    #        cpvec = model.get_pvec(attr=attr, free_only=free_only)
    #        vec_collection.append(cpvec)
    #    return np.hstack(vec_collection)
    #
    #def set_pvec(self, value, attr=None, free_only=True, as_delta=False):
    #    pvec_lb = 0
    #    pvec_ub = 0
    #    for model in self.models:
    #        old_pvec = model.get_pvec(attr=attr, free_only=free_only)
    #        n_params = len(old_pvec)
    #        pvec_ub += n_params
    #        if as_delta:
    #            new_pvec = old_pvec + value[pvec_lb:pvec_ub]
    #        else:
    #            new_pvec = value[pvec_lb:pvec_ub]
    #        model.set_pvec(new_pvec, attr=attr, free_only=free_only)
    #        pvec_lb = pvec_ub
    # 
    #def get_pdict(self, attr=None, free_only=True):
    #    pdict = {}
    #    for model in self.models:
    #        cpdict = model.get_pvec(attr=attr, free_only=free_only)
    #        pdict[(model, attr)] = cpdict
    #    return pdict
    
    def get_expansions(self):
        expansions = [[None for i in range(len(self.models))] for j in range(len(self.trees))]
        for model_idx in range(len(self.models)):
            model = self.models[model_idx]
            relevant_tree_idxs = self.model_to_tree_idxs.get(model)
            if relevant_tree_idxs is None:
                raise Exception("model {} not part of this modeler's chain sequence".format(model))
            for tree_idx in relevant_tree_idxs:
                tree = self.trees[tree_idx]
                expansions[tree_idx][model_idx] = tree.fit_matrix(model) 
        return expansions
    
    def get_model_values(self):
        model_values = []
        for tree in self.trees:
            model_values.append(tree())
        return model_values
    
    def get_data_values(self):
        data_values = []
        for tree in self.trees:
            data_values.append(tree.target_data)
        return data_values
    
    def get_data_weights(self):
        data_weights = []
        for tree in self.trees:
            data_weights.append(tree.data_weight)
        return data_weights
    
    def iterate(self):
        #accumulate the linear fit expansions for all models
        #clamp = self.get_clamping_vector()
        data_weights = np.hstack(self.get_data_weights())
        data_values = np.hstack(self.get_data_values())
        model_values = np.hstack(self.get_model_values())
        
        fit_matrix = scipy.sparse.bmat(self.get_expansions())
        n_params = fit_matrix.shape[1]
        
        zero_vec = np.zeros(n_params)
        deltas = data_values-model_values
        old_chi_sq = np.sum(deltas**2*data_weights)
        
        #target_vec = np.hstack([deltas, zero_vec])
        param_ident_mat = scipy.sparse.identity(n_params, format="csr")
        #fit_and_damp = scipy.sparse.vstack([fit_matrix, param_ident_mat])
        
        ##now generate the weighting matrices and weight both sides
        #double_weight = np.hstack([data_weights, data_weights])
        #n_full = len(double_weight)
        weighting_mat = scipy.sparse.dia_matrix((data_weights, 0), (len(data_weights), len(data_weights)))
        
        #relax the clamping
        self.clamping_factor = max(1.0, self.clamping_factor/self.clamping_nu)
        
        #carry out the fit
        param_direction = scipy.sparse.linalg.lsqr(weighting_mat*fit_matrix, weighting_mat*deltas, atol=0, btol=0, conlim=0, iter_lim=1000)[0]
        
        for reweighting_idx in range(self.max_reweighting_iter+1):
            #set a delta on the basis of the current clamping factor
            cur_delta = param_direction/self.clamping_factor
            self.set_pvec(cur_delta, as_delta=True)
            
            #check to make sure the new parameters are actually better
            new_model_values = self.get_model_values()
            new_chi_sq = np.sum((data_values-new_model_values)**2*data_weights)
            
            if new_chi_sq < old_chi_sq:
                #end the reweighting search
                break
            else:
                self.clamping_factor *= self.clamping_nu
                if self.clamping_factor > self.max_clamping:
                    self.clamping_factor = self.max_clamping
                    break
        
        print "clamping factor", self.clamping_factor
        print "new chi sq {: 10.3f}".format(new_chi_sq)
        self.log_likelihood = -new_chi_sq
        #set the best fit delta
        self.set_pvec(cur_delta, as_delta=True)
        self.iter_num += 1
        #print "iter num", self.iter_num
        #print "fit result", fit_result
        #print "param_vals", self.get_pvec()
        if self.iter_num > self.max_iter:
            self.converged = True
            return True
        return False

class DataModelNetwork(object):
    
    def __init__(self, trees, fit_policy=None):
        self.trees = []
        self.model_to_trees = {}
        for tree in trees:
            self.add_tree(tree)
        if fit_policy is None:
            fit_policy = FitPolicy(self)
        self.set_fit_policy(fit_policy)
    
    def set_fit_policy(self, fit_policy):
        self.fit_policy = fit_policy
        self.fit_policy.set_model_network(self)
    
    def add_tree(self, tree):
        self.trees.append(tree)
        for model in tree.models:
            chain_res = self.model_to_trees.get(model)
            if chain_res is None:
                chain_res = []
            self.model_to_trees[model] = chain_res
            chain_res.append(tree)
    
    def converge(self):
        self.fit_policy.converge()
    
    def iterate(self):
        return self.fit_policy.iterate()
