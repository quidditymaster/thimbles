import time
import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
from copy import copy

from thimbles.sqlaimports import *
from thimbles.thimblesdb import ThimblesTable, Base

class DataContext(ThimblesTable):
    
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
