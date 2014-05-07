

class Predictor(object):
    
    def __init__(self, val_func, sigma_func):
        self.val_func = val_func
        self.sigma_func = sigma_func
    
    def predict(self, pspace, pbag):
        return self.val_func(pspace, pbag)
    
    def sigma(self, pspace, pbag):
        return self.sigma_func(pspace, pbag)
    

class GaussianPredictor(Predictor):
    
    def __init__(self, mean , sigma):
        self.mean = mean
        self._sigma = sigma
    
    def predict(self, pspace):
        return self.mean
    
    def sigma(self, pspace):
        return self._sigma