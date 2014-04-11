import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt

import thimbles as tmb
from ..spaces import Dimension, Space, Vector

class Model(object):
    """a model object with an internal parameter vector object associated to
    a parameter space.
    
    constructor values,
    parameters:  spaces.Vector or dict
     the current parameter values of the model parameters is not a
     spaces.Vector then spaces.Vector(parameters) will be attempted
    model_func: function
     model_func is a function such that model_func(model) returns the value of
     this model.
    pder_funcs: dictionary of functions
     if there is an efficient way to calculate the derivative of the model f
    
    """
    
    def __init__(self, parameters, model_func, pder_funcs=None):
        if not isinstance(parameters, Vector):
            parameters = Vector(parameters)
        self.parameters = parameters
        self.model_func = model_func
        if pder_funcs is None:
            pder_funcs == {}
        self.pder_funcs = pder_funcs
        self._override=None
        self._recalculate = True
    
    def begin_override(self, override_vector):
        self._override = override_vector
    
    def end_override(self, override_vector):
        self._override = None
    
    def __getitem__(self, index):
        if isinstance(index, basestring):
            try:
                return eval("self.%s" % index)
            except AttributeError:
                if not self._override is None:
                    return self._override[index]
                return self.parameters[index]
    
    def __setitem__(self, index, value):
        self.parameters[index] = value
    
    def __call__(self, override_vector=None, ):
        return self.eval_func(self)
    
    def recalculate(self):
        """force the local operators to be recalculated
        """
        self._lop, self._alpha_der, self._alpha_curve = self.get_differential_operators(*self.get_params())
        self._recalculate = False
    
    def get_differential_operators(self, alpha=None, beta=None, pbag=None):
        """get all the differentials at once
        """
        alpha, beta, pbag = self.get_params(alpha, beta, pbag)
        delta_vecs = self.alpha_eps*np.eye(self.n_alpha)
        delta_columns = []
        curve_columns = []
        central_lop = self.get_lop(alpha, beta, pbag)
        for i in range(self.n_alpha):
            #import pdb; pdb.set_trace()
            plus_op = self.get_lop(alpha+delta_vecs[0], beta, pbag)
            minus_op = self.get_lop(alpha-delta_vecs[0], beta, pbag)
            delta_op = (plus_op - minus_op)/(2*self.alpha_eps)
            curve_op = (plus_op - 2*central_lop + minus_op)/(self.alpha_eps**2)
            delta_col = delta_op*beta
            curve_col = curve_op*beta
            delta_columns.append(delta_col)
            curve_columns.append(curve_col)
        if len(delta_columns) == 0:
            alpha_der = None
            alpha_curve = None
        else:
            alpha_der = sparse.bmat(delta_columns).transpose()
            alpha_curve = sparse.bmat(curve_columns).transpose()
        #return a tuple of locally_linear_operator, alpha_derivative_operator, alpha_curvature_operator
        return central_lop, alpha_der, alpha_curve
        