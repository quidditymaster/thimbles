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
        x_in = vdict[self.inputs["input_wvs"]]
        x_out = vdict[self.inputs["output_wvs"]]
        lsf_in = vdict[self.inputs["input_lsf"]]
        lsf_out = vdict[self.inputs["output_lsf"]]
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
        spectrum.add_parameter("fine_norm", fine_norm_p)
        
        if samp_mat_p is None:
            samp_mat_p = Parameter()
            samp_mod = SamplingModel(
                output_p=samp_mat_p,
                input_wv_soln=model_wv_soln, 
                output_wv_soln=spectrum.wv_sample.wv_soln
            )
        self.add_input("sampling_matrix", samp_mat_p)
        spectrum.add_parameter("sampling_matrix", samp_mat_p)
    
    def local_derivative(self, param, vprep=None):
        """the derivative of the output parameter of this matrix
        with respect to the particular specified input parameter.
        """
        vdict = self.get_vdict(vprep)
        if param == self.inputs["model_flux"]:
            fine_norm_val = vdict[self.inputs["fine_norm"]]
            npts = len(fine_norm_val)
            fnmat = scipy.sparse.dia_matrix((fine_norm_val, 0), shape=(npts, npts))
            samp_mat_p = self.inputs["sampling_matrix"]
            samp_mat = vdict[samp_mat_p]
            return fnmat*samp_mat
        elif param == self.inputs["fine_norm"]:
            mod_flux = vdict[self.inputs["model_flux"]]
            samp_mat = vdict[self.iputs["sampling_matrix"]]
            samp_flux = samp_mat*mod_flux
            npts = len(fine_norm_val)
            der_mat = scipy.sparse.dia_matrix((samp_flux, 0), shape=(npts, npts))
            return der_mat
        return None
            

    def __call__(self, vprep=None):
        vdict = self.get_vdict(vprep)
        fine_norm = vdict[self.inputs["fine_norm"]]
        samp_mat = vdict[self.inputs["sampling_matrix"]]
        input_flux = vdict[self.inputs["model_flux"]]
        return fine_norm*(samp_mat*input_flux)
