import numpy as np
import matplotlib.pyplot as plt

class ParameterBag(object):
    """a dummy class for conveniently passing arguments to functions in the FactorModel.
    the parameter bag becomes a consistent object within the FactorModel and so also
    provides a convenient space for the predictor functions to store results.
    """
    pass

class FactorModel(object):
    """represents a component model of a larger vector model expression
    the component model is of the form 
    
    (A_const + A_func(alpha, param_bag))*beta + C_const + C_func(alpha, param_bag)
    
    where the param_bag argument is a ParameterBag object with accessible attributes
    including param_bag.alpha and param_bag.beta as well as any objects passed in
    to the bag_parameters argument.
    
    alpha0: ndarray or None
        the non-linear model parameters to start with
    beta0: ndarray or None
        the linear model parameters to start with
    A_const: scipy.sparse matrix or None
        the constant component of the matrix if None it is assumed to be the zero matrix
    A_func: function or None
        a function that returns a component of the model matrix dependent on the alpha
        parameters. if None there is no alpha dependence of the matrix component.
    C_const: ndarray or None
        a constant vector component of the output. If None it is assumed to be the zero vector.
    C_func: function or None
        a function which takes c_func(alpha, *alpha_args) and returns a vector output.
        If None there are no beta independent parameters in the model.
    A_derivative: function or None
        a function of the form A_derivative(alpha, beta, *alpha_args)
    bag_parameters: dictionary
        a collection of parameters to put in the ParameterBag which gets passed
        to the provided functions. 
    alpha_min: ndarray or None
        the minimum allowed values of the alphas
    alpha_max: ndarray or None
        the maximum allowed values of the alphas
    beta_min: ndarray or None
        the minimum allowed values of the betas
    beta_max: ndarray or None
        the maximum allowed values of the betas
    alpha_predictor: function or None
        an optional function of the form alpha_predictor(alpha, beta, *alpha_args)
        which returns a vector of predicted best alphas. This can be used to 
        accelerate fit convergence using good heuristics or to enforce some sort
        of constraints (e.g. smoothness) on the alpha parameters
    beta_predictor: function or none
        same as alpha_predictor but for beta vector.
    epsilon: ndarray or float
        the small offset to use in estimating the alpha derivatives if derivaitive_func is 
        not supplied.
    curvature fraction: float
        when generating a linearized expansion of the FactorModel around the current
        alpha and beta a term will be generated in the linear fit matrix to penalize
        large deltas in the alpha and beta parameters. 
    
    """
    def __init__(self, alpha0, beta0, A_const=None, A_func=None,
                 C_const=None, C_func=None,
                 A_alpha_derviative=None, C_alpha_derivative=None, 
                 alpha_args=None, beta_args=None, 
                 alpha_min=None, alpha_max=None, beta_min=None, beta_max=None, 
                 alpha_predictor=None, beta_predictor=None,
                 epsilon=1e-8, curvature_fraction=0.1):
        self.alpha = alpha0
        self.beta = beta0
        self.A_const = A_const
        self.A_func = A_func
        self.C_const = C_const
        self.C_func = C_func
        if alpha_args == None:
            alpha_args = tuple()
        self.alpha_args = alpha_args
        if beta_args == None:
            beta_args = tuple()
        self.beta_args=beta_args
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.beta_min = beta_min
        self.beta_max = beta_max
    
    def get_A(self, alpha=None, beta=None):
        if alpha == None:
            alpha = self.alpha
        if self.A_const != None:
            return self.A_const + self.A_func(alpha, *self.alpha_args)
    
    def get_C(self, alpha=None):
        if alpha == None:
            alpha = self.alpha
        if self.C_const != None:
            if self.C_func != None:
                return self.C_const + self.C_func(alpha) 
    
    def __call__(self):
        A = self.get_A()
        C = self.get_C()
        return A*self.beta + C
    
    def set_alpha(self, alpha):
        self.alpha = np.clip(alpha, self.alpha_min, self.alpha_max)
    
    def set_beta(self, beta):
        self.beta = np.clip(beta, self.beta_min, self.beta_max)
    
    def alpha_derivative(self):
        raise NotImplemented
    
class ProductDataModel(object):
    
    def __init__(self, ):