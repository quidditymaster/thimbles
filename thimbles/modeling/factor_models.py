import numpy as np
import matplotlib.pyplot as plt

class ParameterBag(object):
    """a dummy class for conveniently passing arguments to functions in the FactorModel.
    the parameter bag becomes a consistent object within the FactorModel and so also
    provides a convenient space for the predictor functions to store results.
    
    can be initialized with any keyword arguments which go into the namespace of this
    object under their passed in keywords.
    
    par_bag = ParameterBag(x=3, y=1, some_str="tada")
    par_bag.x # equals 3
    par_bag.some_str # equals "tada"
    
    or put values into the bag on the fly
    
    par_bag = ParameterBag()
    par_bag.x = 3
    par_bag.y = 1
    
    """
    def __init__(self, **kwargs):
        for key,value in kwargs.items():
            self.__dict__[key] = value

class IdentityOperation(object):
    """an object which when used in a binary operation returns the other object
    """
    
    def __mul__(self, other):
        return other
    
    def __div__(self, toher):
        return other
    
    def __add__(self, other):
        return other
    
    def __sub__(self, other):
        return other
    
    def __rmul__(self, other):
        return other
    
    def __rdiv__(self, toher):
        return other
    
    def __radd__(self, other):
        return other
    
    def __rsub__(self, other):
        return other



class FactorModel(object):
    """represents a component model of a larger vector model expression
    the component model is of the form 
    
    A_func(alpha, beta, param_bag)*beta + C_func(alpha, beta, param_bag)
    
    note: all functions called for in this model are assumed to take 3 arguments as 
    parameters in the order.
    
    f(alpha, beta, param_bag)
    
    where the param_bag argument is a ParameterBag object belonging to this FactorModel.
    
    alpha0: ndarray or None
        the non-linear model parameters to start with
    beta0: ndarray or None
        the linear model parameters to start with
    A_func: function or None
        a function that returns a component of the model matrix dependent on the alpha
        parameters. if None there is no alpha dependence of the matrix component.
    C_func: function or None
        a function which takes c_func(alpha, *alpha_args) and returns a vector output.
        If None there are no beta independent parameters in the model.
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
    def __init__(self, alpha0, beta0, 
                 A_func=None,
                 C_func=None,
                 A_alpha_der=None, 
                 C_alpha_der=None,
                 A_alpha_curvature=None, 
                 C_alpha_curvature=None, 
                 parameter_bag = None,
                 alpha_min=None, alpha_max=None, beta_min=None, beta_max=None, 
                 alpha_predictors=None,
                 beta_predictors=None,
                 epsilon=1e-7, 
                 curvature_fraction=0.1):
        self.alpha = alpha0
        self.beta = beta0
        self.A_func = A_func
        self.C_func = C_func
        self.A_alpha_der=A_alpha_der
        self.C_alpha_der=C_alpha_der
        
        if parameter_bag == None:
            parameter_bag = ParameterBag()
        self.parameter_bag = parameter_bag
        if alpha_min != None:
            raise NotImplemented
        if alpha_max != None:
            raise NotImplemented
        if beta_min != None:
            raise NotImplemented
        if beta_max != None:
            raise NotImplemented
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.beta_min = beta_min
        self.beta_max = beta_max
    
    def get_A(self, alpha=None, beta=None):
        if alpha == None:
            alpha = self.alpha
        if beta == None:
            beta = self.beta
        if self.A_func == None:
            return None
        else:
            return self.A_func(alpha, beta, self.parameter_bag)
    
    def get_C(self, alpha=None, beta=None):
        if alpha == None:
            alpha = self.alpha
        if beta == None:
            beta = self.beta
        if self.C_func == None:
            return None
        return self.C_func(alpha, beta, self.parameter_bag) 
    
    def __call__(self):
        A = self.get_A()
        C = self.get_C()
        if (A != None):
            if C != None:
                return A*self.beta + C
            else:
                return A*self.beta
        else:
            return C
    
    def set_alpha(self, alpha):
        #self.alpha = np.clip(alpha, self.alpha_min, self.alpha_max)
        self.alpha = alpha
    
    def set_beta(self, beta):
        #self.beta = np.clip(beta, self.beta_min, self.beta_max)
        self.beta = beta
    
    def alpha_derivative_matrix(self):
        if self.A_alpha_der == None:
            if self.A_func == None:
                mat_vecs_out = []
                for alpha_idx in range(len(self.alpha)):
                    #plus delta
                    raise NotImplemented
        else:
            A_der_mat = self.A_alpha_der(self.alpha, self.beta, self.par_bag)
        C_der_mat = self.C_alpha_der(self.alpha, self.beta, self.par_bag)
        return scipy.sparse.bmat([[A_der_mat, C_der_mat]])
    
    def beta_derivative_matrix(self):
        return self.get_A()
    
    def delta_fit_matrix(self):
        alpha_der = self.alpha_derivative_matrix()
        beta_der = self.beta_derivative_matrix()
        return scipy.sparse.bmat([[alpha_der, beta_der]])
    
    def delta_regularization_matrix(self):
        alpha_reg = scipy.sparse.identity(len(self.alpha))
        beta_reg = scipy.sparse.identity(len(self.beta))
        return scipy.sparse.bmat([[alpha_reg, None], [None, beta_reg]])
    
    def regularization_vector(self):
        return np.zeros(len(self.alpha) + len(self.beta))
    
    def regularization_weights(self):
        return 0.1*np.ones(len(self.alpha) + len(self.beta))
    
    def predictor_matrices(self):
        return None

class ScalarModel(FactorModel):
    """a model which simply 
    """
    def __init__(self, init_value):
        raise NotImplemented

class ScaledVectorModel(FactorModel):
    
    def __init__(self):
        raise NotImplemented

class DataModel(object):
    """a model which combines the results of factor models to output 
    a model which is related by a linear transformation to the data space.
    
    data: ndarray
        the data values
    inverse_variance: ndarray
        the inverse variances associated with the data
    factor_models: dictionary
        a dictionary of NonLinearModel objects
    model_expression: string
        the expression which says how to combine the factor models into a
        model which can be compared to data.
        
        e.g. 
        factor_models = {"x": mod1, "y": mod2}
        expression = "x**2+y*x"
    """
    
    def __init__(self, 
                 data, 
                 inverse_variance, 
                 factor_models,
                 model_expression,
                 derivative_expressions,
                 transform=None):
        self.data = data
        self.inverse_variance = inverse_variance
        self.factor_models = factor_models
        self.model

class ProductDataModel(DataModel):
    """a data model which consists of factor models multiplied together.
    """
    def __init__(self, data, inverse_variance, factor_models, transform=None):
        self.data = data
        self.inverse_variance = inverse_variance
        self.factor_models = factor_models
        self.transform = transform
    
    def eval(self):
        prod_res = self.factor_models[0]()
        for fmod in self.factor_models:
            prod_res *= fmod()
        if transform != None:
            return transform*prod_res
        else:
            return prod_res
    
    def fit_iteration(self, data, inverse_variance, factor_models):
        resid_vec = data-self.eval()
        
