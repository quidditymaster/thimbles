import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt

import thimbles as tmb

class ParameterBag(object):
    """a dummy class for conveniently passing around information.
    
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


class LocallyLinearModel(object):
    """A representation of a model whose values are a smooth
    function of its parameters both internal and input.

    the external parameter linear expansion operator and the
    internal parameter linear operator expansion are cached
    internally and a flag to recalculate them is set when
    set_params is called. For most efficient use, set the
    parameters once and then don't pass any parameters in
    to the "get_" functions. (if you pass a alpha, beta or pbag
    in it is assumed you don't want the stored alpha, beta, pbag
    and so you will need to recalculate anyway but the results 
    are not cached.)
    """
    
    def __init__(self, alpha0, beta0, pbag, eval_func, op_func, predictors=None, alpha_eps=1e-7):
        self._alpha = np.asarray(alpha0)
        self._beta = np.asarray(beta0)
        if pbag == None:
            pbag = ParameterBag()
        self._pbag = pbag
        self.n_alpha = len(self._alpha)
        self.eval_func = eval_func
        self.op_func = op_func
        self.alpha_eps = alpha_eps
        if predictors == None:
            predictors = []
        self.predictors = predictors
        self._recalculate = True
    
    def get_params(self, alpha=None, beta=None, pbag=None):
        if alpha == None:
            alpha = self._alpha
        if beta == None:
            beta = self._beta
        if pbag == None:
            pbag = self._pbag
        return alpha, beta, pbag
    
    def __call__(self, alpha=None, beta=None, pbag=None):
        return self.eval_func(*self.get_params(alpha, beta, pbag))
    
    def recalculate(self):
        """force the local operators to be recalculated
        """
        self._lop, self._alpha_der, self._alpha_curve = self.get_differential_operators(*self.get_params())
        self._recalculate = False
    
    def get_lop(self, alpha=None, beta=None, pbag=None):
        """return a linear operator expansion with respect to
        the external (beta) input parameters.
        """
        if (alpha==None) and (beta==None) and (pbag==None):
            if self._recalculate:
                self.recalculate()
            return self._lop
        else:
            abp = self.get_params(alpha, beta, pbag)
            return self.op_func(*abp)
    
    def get_alpha_der(self, alpha=None, beta=None, pbag=None):
        """the local linear operator turning delta alpha into
        a delta on the output parameters of this model, given
        the current or passed in beta.
        """
        if (alpha==None) and (beta==None) and (pbag==None):
            if self._recalculate:
                self.recalculate()
            return self._alpha_der
        else:
            abp = self.get_params(alpha, beta, pbag)
            lop, alpha_der, alpha_curve =  self.get_differential_operators(*abp)
            return alpha_der
    
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
    
    def evaluate_predictors(self, alpha, beta, pbag):
        #TODO:
        return None, None, None
    
    def set_params(self, alpha=None, beta=None, pbag=None):
        if alpha == None:
            alpha = self._alpha
        if beta == None:
            beta = self._beta
        if pbag == None:
            pbag = self._pbag
        self._alpha = alpha
        self._beta = beta
        self._pbag = pbag
        self._recalculate = True
    
    @property
    def alpha(self):
        return self._alpha
    
    @property
    def beta(self):
        return self._beta

    @property
    def pbag(self):
        return self._pbag


class MatrixModel(LocallyLinearModel):
    
    def __init__(self, alpha0, beta0, matrix_func, predictors=None, alpha_eps=None):
        raise NotImplemented

class CollapsedMatrixModel(LocallyLinearModel):
    """ 
    """
    def __init__(self, alpha0, beta0, matrix_generator, predictors=None, alpha_eps=1e-7):
        self.alpha_int = np.asarray(alpha0)
        self.beta_int = np.asarray(beta_internal0)
        self.n_alpha = len(self._alpha)
        self.n_beta = len(self.beta_int)
        
        if predictors == None:
            predictors = []
        self.predictors = predictors
        self.alpha_eps = alpha_eps
    
    @property
    def alpha(self):
        return np.hstack((self.alpha_int, self.beta_int))
    
    def split_alpha(self, alpha):
        return alpha[:self.n_alpha], alpha[self.n_alpha:] 
    
    def get_mat(self, alpha, beta, pbag):
        alpha_int, beta_int = self.split_alpha(alpha)
        mat = self.matrix_generator(alpha_int, beta_int, pbag)
        return mat
    
    def get_vec(self, alpha, beta, pbag):
        alpha_int, beta_int = self.split_alpha(alpha)
        mat = self.matrix_generator(alpha_int, beta_int, pbag)
        vec = mat*beta_int
        return vec
    
    def __call__(self, alpha, beta, pbag):
        vec = self.get_vec(alpha, beta, pbag)
        return beta*vec
    
    def get_lop(self, alpha, beta, pbag):
        vec = self.get_vec(alpha, beta, pbag)
        dmat = sparse.dia_matrix((vec, 0), shape=(len(vec), len(vec)))
        return dmat

class ProductDataModel(object):
    
    def __init__(self, data, data_sigma, data_gamma, models, parameter_bag=None):
        self.data = data
        self.data_sigma = data_sigma
        self.data_gamma
        self.sig4 = data_sigma**4
        self.gam4 = data_gamma**4
        self.rat4 = self.sig4/self.gam4
        self.models = models
        if parameter_bag == None:
            parameter_bag = ParameterBag()
        self.pbag = parameter_bag
    
    def build_fit_matrix(self):
        base_mod = self.models[-1]
        base_alpha = base_mod._alpha
        prev_eval = base_mod(base_alpha, None, self.pbag)
        clop, calpha_der, calpha_curve = base_mod.get_differential_operators(base_alpha, self.base_beta, self.pbag)
        delta_alpha_lops = [calpha_der]
        delta_alpha_curve = [calpha_curve] 
        for i in range(2, len(self.models)+1):
            cmod = self.models[-i]
            calpha = cmod._alpha
            clop, calpha_der, calpha_curve = cmod.get_differential_operators(calpha, prev_eval, self.pbag)
            prev_eval = cmod(calpha, prev_eval, self.pbag)
            for pidx in range(len(delta_alpha_lops)):
                delta_alpha_lops[pidx] = clop*delta_alpha_lops[pidx]
                delta_alpha_curve[pidx] = clop*delta_alpha_curve[pidx]
            delta_alpha_lops.insert(0, calpha_der)
            delta_alpha_curve.insert(0, calpha_curve)
        fit_mat = sparse.bmat([delta_alpha_curve])
        resids = self.data-prev_eval
        return resids, fit_mat
        #TODO: also build the predictor matrices and the damping matrices. 
    
    def iterate(self):
        resids, fit_mat = self.build_fit_matrix()
        weights = 1.0/(self.data_gamma*np.sqrt(self.rat4+resids**2))
        ftrans = fit_mat.transpose()
        ata_inv = ftrans*(weights*fit_mat)
        fit_vec = sparse.linalg.lsqr(ata_inv, ftrans*(weights*resids))
        cstart = 0
        cend = 0
        for model_idx in range(len(self.models)):
            cmod = self.models[model_idx]
            n_alpha = len(cmod._alpha)
            cend += n_alpha
            cmod.set_alpha(fit_vec[cstart:cend])
            cstart += n_alpha

class IdentityOperation(object):
    """an object which when used in a binary operation returns the other object
    """
    
    def __mul__(self, other):
        return other
    
    def __div__(self, other):
        return other
    
    def __add__(self, other):
        return other
    
    def __sub__(self, other):
        return other
    
    def __rmul__(self, other):
        return other
    
    def __rdiv__(self, other):
        return other
    
    def __radd__(self, other):
        return other
    
    def __rsub__(self, other):
        return other

    
