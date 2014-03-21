import numpy as np

from .. import Spectrum
from . import factor_models
from . import data_models

class FeatureModel(factor_models.LocallyLinearModel):
    
    def __init__(self):
        raise NotImplemented

class VoigtSumModel(FeatureModel):
    
    def __init__(self, sample_wvs, features):
        """represents the spectrum as a linear sum of voigt features
        subtracted from a vector of ones.
        """
        self.sample_wvs = sample_wvs
        self.features = features

class SpectrographModel(factor_models.LocallyLinearModel):
    
    def __init__(self):
        pass

class SamplingModel(factor_models.LocallyLinearModel):
    
    def __init__(self):
        pass
    
class ReddeningModel(factor_models.LocallyLinearModel):
    
    def __init__(self):
        pass

class BroadeningModel(factor_models.LocallyLinearModel):
    
    def __init__(self):
        pass

class ContinuumModel(factor_models.LocallyLinearModel):
    
    def __init__(self):
        pass

class SpectralModel(Spectrum, factor_models.LocallyLinearModel):
    
    def __init__(self, wvs, model_wvs, models, ):