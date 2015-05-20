import numpy as np
import scipy.sparse

from thimbles.utils.partitioning import partitioned_polynomial_model
from thimbles.utils import piecewise_polynomial as ppol 
from thimbles.thimblesdb import Base, ThimblesTable
from thimbles.modeling import Model, Parameter
from thimbles.sqlaimports import *
from thimbles.spectrum import FluxParameter
from thimbles.modeling import PixelPolynomialModel
import thimbles as tmb


class PiecewisePolynomialSpectrographEfficiencyModel(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    
    def __init__(self, spec_wvs, degree=3, n_max_part=5):
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
    
    def poly_coeffs_p(self):
        return self.coefficients
    
    def set_poly_coeffs(self, value):
        self.coefficients = value
    
    def parameter_expansion(self, input, **kwargs):
        return scipy.sparse.csc_matrix((self._basis*input.reshape((-1, 1))))
    
    def as_linear_op(self, input, **kwargs):
        return scipy.sparse.dia_matrix((self.blaze(), 0), shape = (len(self.wv), len(self.wv)))
        
    def __call__(self, input, **kwargs):
        return self.blaze()*input


class SamplingModel(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"SamplingModel",
    }
    
    def __init__(self, output_p, input_wv_soln, output_wv_soln):
        self.output_p = output_p
        self.add_input("input_wvs", input_wv_soln.indexer.output_p)
        self.add_input("input_lsf", input_wv_soln.lsf_p)
        self.add_input("output_wvs", output_wv_soln.indexer.output_p)
        self.add_input("output_lsf", output_wv_soln.lsf_p)  
    
    def __call__(self, vprep=None):
        vdict = self.get_vdict(vprep)
        x_in = vdict[self.inputs["input_wvs"][0]]
        x_out = vdict[self.inputs["output_wvs"][0]]
        lsf_in = vdict[self.inputs["input_lsf"][0]]
        lsf_out = vdict[self.inputs["output_lsf"][0]]
        return tmb.resampling.resampling_matrix(x_in, x_out, lsf_in, lsf_out)


class SpectrographModel(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"SpectrographModel",
    }
    
    def __init__(
            self, 
            spectrum, 
            model_wv_soln,
            model_flux_p,
            fine_norm_p = None,
            samp_mat_p = None,
    ):
        self.output_p = spectrum.flux_p
        spec_wv_soln = spectrum.wv_sample.wv_soln
        model_wv_soln = tmb.as_wavelength_solution(model_wv_soln)
        self.add_input("model_flux", model_flux_p)
        
        if fine_norm_p is None:
            fine_norm_p = FluxParameter(spec_wv_soln, flux=spectrum.flux)
            fine_norm_mod = PixelPolynomialModel(output_p=fine_norm_p)
        self.add_input("fine_norm", fine_norm_p)
        
        if samp_mat_p is None:
            samp_mat_p = Parameter()
            samp_mod = SamplingModel(
                output_p=samp_mat_p,
                input_wv_soln=model_wv_soln, 
                output_wv_soln=spectrum.wv_sample.wv_soln
            )
        self.add_input("sampling_matrix", samp_mat_p)
    
    def __call__(self, vprep=None):
        vdict = self.get_vdict(vprep)
        fine_norm = vdict[self.inputs["fine_norm"][0]]
        samp_mat = vdict[self.inputs["sampling_matrix"][0]]
        input_model = vdict[self.inputs["model_flux"][0]]
        return fine_norm
