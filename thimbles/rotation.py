import numpy as np
import scipy

from thimbles.utils.misc import sparse_row_circulant_matrix
from thimbles.profiles import compound_profile

from thimbles.modeling import Model, Parameter
from thimbles.sqlaimports import *
from thimbles import speed_of_light

class BroadeningMatrixModel(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"MacroscopicDopplerBroadeningModel",
    }
    
    def __init__(
            self, 
            output_p, 
            model_wvs, 
            vmacro, 
            vsini, 
            ldark, 
    ):
        self.output_p = output_p
        self.add_parameter("model_wvs", model_wvs)
        self.add_parameter("vsini", vsini)
        self.add_parameter("vmacro", vmacro)
        self.add_parameter("ldark", ldark)
    
    def __call__(self, override=None):
        vdict = self.get_vdict(override)
        model_wv_indexer = vdict[self.inputs["model_wvs"]]
        vsini = vdict[self.inputs["vsini"]]
        vmacro = vdict[self.inputs["vmacro"]]
        ldark = vdict[self.inputs["ldark"]]
        
        wvs = model_wv_indexer.coordinates
        center_wv = wvs[len(wvs)//2]
        veff = np.sqrt(vmacro**2 + vsini**2)
        n_w = 8.0
        low_wv = center_wv*(1.0 - n_w*veff/speed_of_light)
        high_wv = center_wv*(1.0 + n_w*veff/speed_of_light)
        idx_bnds = model_wv_indexer.get_index([low_wv, high_wv], snap=True)
        lbi, ubi = idx_bnds
        if (ubi-lbi) % 2 == 1:
            ubi = ubi-1
        sample_wvs = wvs[lbi:ubi+1]
        center_wv = sample_wvs[len(sample_wvs)//2]
        prof = compound_profile(
            sample_wvs, 
            center_wv, 
            sigma=0.0,
            gamma=0.0,
            vsini=vsini,
            limb_dark=ldark,
            vmacro=vmacro,
        )
        
        #note using just a translated version of the same profile
        #only works if we are using log-linear model wavelengths  
        broadening_mat = sparse_row_circulant_matrix(prof, len(wvs), normalize=True)
        
        return broadening_mat
