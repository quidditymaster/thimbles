

class Predictor(object):
    
    def __init__(self, delta_func, sigma_func, gamma_func):
        self.delta_func = delta_func
        self.sigma_func = sigma_func
        self.gamma_func = gamma_func
    
    def __call__(self, alpha, beta, pbag):
        alpha_delta = self.delta_func(alpha, beta, pbag)
        dsig = self.sigma_func(alpha, beta, pbag)
        dgamma = self.gamma_func(alpha, beta, pbag)
        return alpha_delta, dsig, dgamma
