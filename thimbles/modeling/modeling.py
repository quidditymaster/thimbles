import time
import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
from copy import copy

from thimbles.sqlaimports import *
from thimbles.thimblesdb import ThimblesTable, Base
from thimbles.modeling import ParameterGroup

class ModelingError(Exception):
    pass

class ValueHistory(object):
    
    def __init__(self, parameter, max_length=50):
        self.set_parameter(parameter)
        self._vals = []
        self._context = {}
        self.max_length = max_length
        self._named_values = {}
        self._named_contexts = {}
    
    def __len__(self):
        return len(self._vals)
    
    def set_parameter(self, parameter):
        self.parameter = parameter
    
    def recontextualize(self):
        self._vals = []
        self._context = self.parameter.get(attr="context")
    
    def remember(self, value_id=None):
        cur_val = copy(self.parameter.get())
        if value_id is None:
            self._vals.append(cur_val)
            if len(self._vals) > self.max_length:
                self._vals.pop(0)
                self._contexts.pop(0)
        else:
            self._named_values[value_id] = cur_val
    
    @property
    def values(self):
        return self._vals
    
    @property
    def named_values(self):
        return self._named_values
    
    @property
    def named_contexts(self):
        return self._named_contexts
    
    @property
    def last(self):
        if len(self) > 0:
            return self._vals[-1]
        else:
            return None
    
    @property
    def context(self):
        return self._context
    
    def revert(self, value_id=None, pop=False):
        if value_id is None:
            context = self.context
            if pop:
                val = self._vals.pop()
            else:
                val = self.last
        else:
            val = self._named_values[value_id]
            context = self._named_contexts[value_id]
        self.parameter.set(val, **context)


class FitPolicy(object):
    
    def __init__(self, model_network=None, fit_states=None, iteration_callback=None, transition_callback=None, finish_callback=None, max_state_iter=1000, max_transitions=1000):
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
        for available_trans in list(self.transition_map.values()):
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
                    print("fit state converged")
                    break
            if not fs_converged:
                print("warning max_iter exceded")
            self.current_fit_state.cleanup()
            if not self.transition_callback is None:
                self.transition_callback(self.current_fit_state)
            print("attempting transition to next fit state")
            #import pdb;pdb.set_trace()
            transition_options = available_transitions.get(self.current_fit_state, [])
            if transition_options == []:
                print("no next fit state found assuming completion")
                if not self.finish_callback is None:
                    self.finish_callback(self.current_fit_state) 
                break
            else:
                self.current_fit_state = transition_options.pop(0)

class FitState(ParameterGroup):
    
    def __init__(self, 
                 model_network=None, 
                 models=None, trees=None, 
                 clamping_factor=5.0, 
                 clamping_nu=1.4, 
                 max_clamping=1000.0, 
                 alpha=2.0, 
                 beta=2.0, 
                 max_iter=10, 
                 max_reweighting_iter=5, 
                 setup_func=None, 
                 iter_setup_func=None, 
                 iter_cleanup_func=None, 
                 cleanup_func=None):
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
            self.models = list(self.model_to_tree_idxs.keys())
        self.update_grouped_parameters()
    
    def update_grouped_parameters(self):
        self._parameters = []
        if not self.models is None:
            for model in self.models:
                self._parameters.extend(model.parameters)
    
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
            
            chi_improved=False
            if new_chi_sq < old_chi_sq:
                chi_improved = True
                #end the reweighting search
                break
            else:
                self.clamping_factor *= self.clamping_nu
                if self.clamping_factor > self.max_clamping:
                    self.clamping_factor = self.max_clamping
                    break
        
        print("clamping factor", self.clamping_factor)
        print("new chi sq {: 10.3f}".format(new_chi_sq))
        
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
