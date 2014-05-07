import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt

import thimbles as tmb
from ..spaces import Dimension, Space, Vector

class ModelingError(Exception):
    pass

class Model(object):
    """a model is a representation of a mapping between parameter spaces
    
    inputs: dict
     a dictionary of key word arguments as keys and models which supply
     them as values.
    outputs: dict
     a dictionary of output names as keys (which are potentially matched
     to the key word arguments of other models) and functions as values.
    map_var: dict
     if the mapping represented by the model has some inherent uncertainty
     in the relationship itself this dictionary provides functions which 
     map the same parameters as the function itself to variances, for
     individual outputs. (e.g. a model might predict height as a function
     of age for which the correlation is strong but imperfect and has a 
     variance which is highest around the age of puberty and is non-zero for 
     all ages.)
     if the intrinsic model variance is not specified the model is assumed
     to be exact. 
    """
    
    def __init__(self, inputs=None, outputs=None, map_var=None):
        if inputs is None:
            inputs = {}
        if outputs is None:
            outputs = {}
        if map_var is None:
            map_var = {}
        self.inputs = inputs
        self.outputs = outputs
        self.map_var = map_var
    
    def __add__(self, other):
        am = AdditiveModel([self, other])
        return am
    
    def __mul__(self, other):
        mm = MultiplicativeModel([self, other])
        return mm
    
    def __lshift__(self, other):
        output_keys = other.outputs.keys()
        input_keys = self.inputs.keys()
        overlap = set(output_keys).intersection(input_keys)
        if overlap == set():
            raise ModelingError("models have no matching inputs/outputs")
        for mkey in overlap:
            self.inputs[mkey] = other

class AdditiveModel(Model):
    
    def __init__(self, models):
        pass

class MultiplicativeModel(Model):
    
    def __init__(self, models):
        pass

class OldModel(object):
    """a model object with an internal parameter vector object associated to
    a parameter space.
    
    constructor values,
    parameters:  spaces.Vector or dict
     the current parameter values of the model parameters is not a
     spaces.Vector then spaces.Vector(parameters) will be attempted
    model_funcs: function
     model_funcs is a function such that model_funcs(model) returns the value of
     this model.
    pder_funcs: dictionary of functions
     if there is an efficient way to calculate the derivative of the model f
    
    """
    
    def __init__(self, parameters, model_funcs, pder_funcs=None):
        if not isinstance(parameters, Vector):
            parameters = Vector(parameters)
        self.parameters = parameters
        self.model_funcs = model_funcs
        if pder_funcs is None:
            pder_funcs == {}
        self.pder_funcs = pder_funcs
        self._override=Vector({})
        self._recalculate = True
        self.outputs=None
    
    def begin_override(self, override_vector):
        self._override = override_vector
    
    def end_override(self, override_vector):
        self._override = None
    
    def __getitem__(self, index):
        try:
            return self._override[index]
        except IndexError:
            pass
        except KeyError:
            pass
        if isinstance(index, basestring):
            try:
                return eval("self.%s" % index)
            except AttributeError:
                if not self._override is None:
                    return self._override[index]
                return self.parameters[index]
    
    def __setitem__(self, index, value):
        self.parameters[index] = value
    
    def __call__(self):
        for dim_key in self.model_funcs:
            pass
    
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
