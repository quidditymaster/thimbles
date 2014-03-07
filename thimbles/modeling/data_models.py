
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt

import thimbles as tmb

class DataModel(object):

    def __init__(self, data, inv_var, models):
        self.data = data
        self.inv_var = inv_var
        self.models = models
        self._recalculate = True
    
    def recalculate(self):
        """internally recalculate the fit matrix and cache it
        """
        lops = []
        alpha_ders = []
        n_mod = len(self.models)
        n_alphas = [len(self.models[i].alpha) for i in range(n_mod)]
        n_betas = [len(self.models[i].beta) for i in range(n_mod)]
        for model_idx in range(n_mod):
            cmod = self.models[model_idx]
            clop, calpha_der, calpha_curve = cmod.get_differential_operators()
            lops.append(clop)
            alpha_ders.append(calpha_der)
        
        resid_blocks = []
        beta_blocks = []
        alpha_blocks = []
        
        for model_idx in range(n_mod):
            for block_idx in range(len(resid_blocks)):
                cblock = resid_blocks[block_idx]
                beta_blocks[model_idx] = cblock
                resid_blocks[block_idx] = lops[model_idx]*cblock
            resid_blocks.append(alpha_ders[model_idx])
            
            for block_idx in range(len(resid_blocks)):
                pass
            
            
    
    
    def get_fit_matrix(self):
        if self._recalculate:
            self.recalculate()
        return self._fit_mat 
