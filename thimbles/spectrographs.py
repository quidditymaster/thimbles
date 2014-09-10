import numpy as np
import scipy.sparse

from thimbles.utils.partitioning import partitioned_polynomial_model
from thimbles.utils import piecewise_polynomial as ppol 
from thimbles.thimblesdb import Base, ThimblesTable
from thimbles.modeling.modeling import parameter, Model

from sqlalchemy import create_engine, ForeignKey
from sqlalchemy import Column, Date, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import relationship, backref

class Spectrograph(Base, ThimblesTable):
    name = Column(String)

class SpectrographSetup(Base, ThimblesTable):
    spectrograph_id = Column(Integer, ForeignKey("Spectrograph._id"))
    spectrograph = relationship(Spectrograph)

class PiecewisePolynomialSpectrographEfficiencyModel(Model):
    
    def __init__(self, spec_wvs, degree=3, n_max_part=5):
        Model.__init__(self)
        self.wv = spec_wvs
        self.degree = degree
        self.n_max_part = n_max_part
        self.configure_control_points()
        self.calc_basis()
        self.coefficients = np.ones(self.n_coeffs)
    
    @property
    def n_coeffs(self):
        return self._basis.shape[1]
    
    def configure_control_points(self):
        delta_pix = max(int(len(self.wv)/self.n_max_part), 2)
        self.control_points = self.wv[delta_pix/2:-delta_pix/2:delta_pix].copy()
    
    def calc_basis(self):
        self.rcppb = ppol.RCPPB(poly_order=self.degree, control_points=self.control_points, scales=np.std(self.wv)*np.ones(len(self.control_points)+1))
        self._basis = self.rcppb.get_basis(self.wv).transpose()
    
    def retrain(self, target_output, input, **kwargs):
        mult_basis = self._basis*input.reshape((-1, 1))
        new_coeffs =  np.linalg.lstsq(mult_basis, target_output)[0]
        self.coefficients = new_coeffs
    
    def blaze(self):
        return np.dot(self._basis, self.coefficients)
    
    @parameter(free=True, start_damp=0.05, min=-np.inf, max=np.inf, min_step=0.0, max_step=1000)
    def poly_coeffs_p(self):
        return self.coefficients
    
    @poly_coeffs_p.setter
    def set_poly_coeffs(self, value):
        self.coefficients = value
    
    @poly_coeffs_p.expander
    def parameter_expansion(self, input, **kwargs):
        return scipy.sparse.csc_matrix((self._basis*input.reshape((-1, 1))))
    
    def as_linear_op(self, input, **kwargs):
        return scipy.sparse.dia_matrix((self.blaze(), 0), shape = (len(self.wv), len(self.wv)))
        
    def __call__(self, input, **kwargs):
        return self.blaze()*input
