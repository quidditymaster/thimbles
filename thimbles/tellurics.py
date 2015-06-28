import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.integrate as integrate
import scipy.sparse

import thimbles as tmb
from thimbles.modeling import Parameter, Model
from .sqlaimports import *
from thimbles import speed_of_light

class TelluricShiftMatrixModel(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"TelluricShiftMatrixModel",
    }
    
    def __init__(
            self,
            output_p,
            model_wvs_p,
            rv_p,
            delta_helio_p,
    ):
        self.output_p = output_p
        self.add_input("model_wvs", model_wvs_p)
        self.add_input("rv", rv_p)
        self.add_input("delta_helio", delta_helio_p)
    
    def __call__(self, vprep=None):
        vdict = self.get_vdict(vprep)
        model_wvs_indexer = vdict[self.inputs["model_wvs"]]
        model_wvs = model_wvs_indexer.coordinates
        rv = vdict[self.inputs["rv"]]
        delta_helio = vdict[self.inputs["delta_helio"]]
        
        #find the wavelengs the tellurics overlay in the star
        overlay_wvs = model_wvs*(1.0+(rv+delta_helio)/speed_of_light)
        smat = model_wvs_indexer.interpolant_sampling_matrix(overlay_wvs)
        return smat

class TransmissionModel(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"TelluricTransmissionModel",
    }
    
    def __init__(
            self, 
            output_p,
            opacity_p
    ):
        self.output_p = output_p
        self.add_input("opacity", opacity_p)        
    
    def __call__(self, vprep=None):
        vdict = self.get_vdict(vprep)
        opac = vdict[self.inputs["opacity"]]
        return np.exp(-opac)
