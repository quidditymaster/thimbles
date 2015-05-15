
import numpy as np
import scipy
from scipy.interpolate import interp1d
import thimbles as tmb
from thimbles.modeling import Model, Parameter
from thimbles.modeling.factor_models import PickleParameter
from .sqlaimports import *
from thimbles.thimblesdb import ThimblesTable, Base

class PolynomialLSFModel(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    __mapper_args__ = {
        "polymorphic_identity":"PolynomialLSFModel"
    }
    npts = Column(Integer)
    degree = Column(Integer)
    
    def __init__(self, lsf_p, degree):
        lsf_val = lsf_p.value
        npts = len(lsf_val)
        self.output_p = lsf_p
        self.npts = npts
        self.degree = degree
        
        x = (np.arange(npts, dtype=float)-npts)/npts
        coeffs = np.polyfit(x, lsf_val, deg=degree)
        coeffs_p = PickleParameter(coeffs)
        self.add_input("coeffs", coeffs_p)
    
    def __call__(self, vprep=None):
        coeffs_p ,= self.inputs["coeffs"]
        vdict = self.get_vdict(vprep)
        coeffs = vdict[coeffs_p]
        npts = self.npts
        x = (np.arange(npts, dtype=float)-npts)/npts
        return np.polyval(coeffs, x)

