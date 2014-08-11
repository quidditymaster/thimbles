import numpy as np
import scipy.sparse

from thimbles.utils.partitioning import partitioned_polynomial_model
from thimbles.utils import piecewise_polynomial as ppol 
from thimbles.thimblesdb import Base, ThimblesTable

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


class PiecewisePolynomialSpectrographEfficiencyModel(object):
        
    def __init__(self, spectrograph_wvs, degree=3, n_max_part=10):
        self.wvs = spectrograph_wvs
        self.degree = degree
        self.min_delta_log_wv
        self.n_max_part
        self.configure_control_points()
        self.calc_basis()
        self.coefficients = np.ones(self.n_coeffs)
        
    @property
    def n_coeffs(self):
        return self._basis.shape[1]
    
    def configure_control_points(self):
        npts = len(self.wvs)
        delta_pix = max(int(npts/self.n_max_part), 2)
        self.control_points = self.wvs[delta_pix/2:-delta_pix/2:delta_pix].copy()
        #partitioned_polynomial_model(xvec, yvec, y_inv_var, poly_order=self.degree, min_delta = 0, alpha=2.0, beta=2.0, beta_epsilon=0.01)
    
    def calc_basis(self):
        self.rcppb = ppol.RCPPB(poly_order=self.degree, control_points=self.control_points)
        self._basis = self.rcppb.get_basis(self.wvs)
    
    def as_linear_op(self, input, **kwargs):
        return scipy.sparse.dia_matrix((np.dot(self._basis, self.coefficients), 0), shape = (len(self.wvs), len(self.wvs)))
    
    def __call__(self, input, **kwargs):
        return np.dot(self._basis, self.coefficients)*input