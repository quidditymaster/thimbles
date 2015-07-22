import numpy as np
import pandas as pd
import thimbles as tmb
import matplotlib.pyplot as plt
from copy import copy
import scipy
import scipy.integrate as integrate
import scipy.sparse
from thimbles import resource_dir
from .flags import FeatureFlags
from thimbles import ptable
from thimbles.profiles import voigt
from thimbles import speed_of_light
from thimbles.transitions import TransitionGroupingStandard
from thimbles.transitions import Transition
from thimbles.modeling import Model, Parameter
from thimbles.modeling import factor_models
from thimbles.utils.misc import smooth_ppol_fit
import thimbles.utils.piecewise_polynomial as ppol
from thimbles import hydrogen
from thimbles.thimblesdb import ThimblesTable, Base
from thimbles.sqlaimports import *
from thimbles.radtran import mooger


def voigt_feature_matrix(
        wavelengths,
        centers,
        sigmas,
        gammas=None,
        saturations=None,
        wv_indexer=None,
        n_gamma=60.0,
        n_sigma=6.0,
):
    indexes = []
    profiles = []
    n_features = len(centers)
    if gammas is None:
        gammas = np.zeros(n_features, dtype=float)
    if saturations is None:
        saturations = np.zeros(n_features, dtype=float)
    assert len(sigmas) == n_features
    assert len(gammas) == n_features
    gamma_eff = np.where(saturations > 1.0, gammas*saturations, gammas)
    gmax = n_gamma*gamma_eff #take into account saturation effects
    smax = n_sigma*sigmas
    wvs_grad = scipy.gradient(wavelengths)
    window_deltas = np.where(gmax > smax, gmax, smax)
    if wv_indexer is None:
        wv_indexer = tmb.coordinatization.as_coordinatization(wavelengths)
    for col_idx in range(n_features):
        ccent = centers[col_idx]
        csig = sigmas[col_idx]
        cgam = gammas[col_idx]
        cdelt = window_deltas[col_idx]
        lb, ub = wv_indexer.get_index([ccent-cdelt, ccent+cdelt], clip=True, snap=True)
        local_wvs = wavelengths[lb:ub+1]
        local_wv_grad = wvs_grad[lb:ub+1]
        prof = voigt(local_wvs, ccent, csig, cgam)
        if saturations[col_idx] > 0:
            #generate the saturated profile
            prof = np.power(10.0, -saturations[col_idx]*prof)-1.0
            #and normalize to have equivalent width == 1 angstrom
            prof /= np.sum(prof*local_wv_grad)
        indexes.append(np.array([np.arange(lb, ub+1), np.repeat(col_idx, len(prof))]))
        profiles.append(prof)
    indexes = np.hstack(indexes)
    profiles = np.hstack(profiles)
    npts = len(wavelengths)
    mat = scipy.sparse.csc_matrix((profiles, indexes), shape=(npts, n_features))
    return mat

class FeatureMatrixModel(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"FeatureMatrixModel",
    }
    
    def __init__(
            self,
            output_p,
            model_wv_p,
            centers_p,
            sigma_p,
            gamma_p,
            saturation_p,
    ):
        self.output_p = output_p
        self.add_parameter("model_wvs", model_wv_p)
        self.add_parameter("centers", centers_p)
        self.add_parameter("sigmas", sigma_p)
        self.add_parameter("gammas", gamma_p)
        self.add_parameter("saturations", saturation_p)
    
    def __call__(self, vprep=None):
        vdict = self.get_vdict(vprep)
        wv_indexer = vdict[self.inputs["model_wvs"]]
        mod_wvs = wv_indexer.coordinates
        centers = vdict[self.inputs["centers"]]
        gammas  = vdict[self.inputs["gammas"]]
        sigmas= vdict[self.inputs["sigmas"]]
        saturations = vdict[self.inputs["saturations"]]
        fmat = voigt_feature_matrix(
            mod_wvs, 
            centers=centers, 
            sigmas=sigmas, 
            gammas=gammas, 
            saturations=saturations, 
            wv_indexer=wv_indexer
        )
        return fmat

class SigmaModel(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"SigmaModel"
    }
    
    def __init__(
            self, 
            output_p, 
            teff_p, 
            vmicro_p, 
            transition_wvs_p, 
            ion_weights_p
    ):
        self.output_p = output_p
        self.add_parameter("teff", teff_p)
        self.add_parameter("vmicro", vmicro_p)
        self.add_parameter("wvs", transition_wvs_p)
        self.add_parameter("weights", ion_weights_p)
    
    def __call__(self, vprep=None):
        vdict = self.get_vdict(vprep)
        teff = vdict[self.inputs["teff"]]
        vmicro = vdict[self.inputs["vmicro"]]
        t_wvs = vdict[self.inputs["wvs"]]
        weights = vdict[self.inputs["weights"]]
        therm_widths = tmb.utils.misc.thermal_width(teff, t_wvs, weights)
        vmic_widths = t_wvs*(vmicro/tmb.speed_of_light)
        
        return np.sqrt(vmic_widths**2 + therm_widths**2)

class TransitionWavelengthVectorModel(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"TransitionWavelengthVectorModel"
    }    
    
    def __init__(self, output_p, indexer_p):
        self.output_p = output_p
        self.add_parameter("indexer", indexer_p)
    
    def __call__(self, vprep=None):
        vdict = self.get_vdict(vprep)
        indexer = vdict[self.inputs["indexer"]]
        t_wvs = np.array([t.wv for t in indexer.transitions])
        return t_wvs

class IonWeightVectorModel(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"IonWeightVectorModel"
    }
    
    def __init__(self, output_p, indexer_p):
        self.output_p = output_p
        self.add_parameter("indexer", indexer_p)
    
    def __call__(self, vprep=None):
        vdict = self.get_vdict(vprep)
        indexer = vdict[self.inputs["indexer"]]
        weights = np.array([t.ion.weight for t in indexer.transitions])
        return weights

class GammaModel(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"GammaModel"
    }
    ref_gamma_wv = Column(Float)
    
    def __init__(self, output_p, gamma_p, transition_wvs_p, ref_gamma_wv=5000.0):
        self.output_p = output_p
        self.add_parameter("gamma", gamma_p)
        self.add_parameter("transition_wvs", transition_wvs_p)
        self.ref_gamma_wv = ref_gamma_wv
    
    def __call__(self, vprep=None):
        vdict = self.get_vdict(vprep)
        gamma = vdict[self.inputs["gamma"]]
        t_wvs = vdict[self.inputs["transition_wvs"]]
        gammas = gamma*(t_wvs/self.ref_gamma_wv)**2
        return gammas


class RelativeStrengthMatrixModel(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"RelativeStrengthMatrixModel"
    }
    
    def __init__(
            self,
            output_p,
            grouping_p,
            transition_indexer_p,
            exemplar_indexer_p,
            pseudostrength_p,
            cog_p,
    ):
        self.output_p = output_p
        self.add_parameter("groups", grouping_p)
        self.add_parameter("transition_indexer", transition_indexer_p)
        self.add_parameter("exemplar_indexer", exemplar_indexer_p)
        self.add_parameter("pseudostrength", pseudostrength_p)
        self.add_parameter("cog", cog_p)
        
    def __call__(self, vprep=None):
        vdict = self.get_vdict(vprep)
        row_idxs = [] #transition indexes
        col_idxs = [] #exemplar indexes
        rel_strengths = []
        groups = vdict[self.inputs["groups"]]
        exemplar_indexer = vdict[self.inputs["exemplar_indexer"]]
        transition_indexer = vdict[self.inputs["transition_indexer"]]
        pseudostrengths = vdict[self.inputs["pseudostrength"]]
        cog = vdict[self.inputs["cog"]]
        exemplars = exemplar_indexer.transitions
        for exemplar_idx, exemplar in enumerate(exemplars):
            exemp_idx_as_trans = transition_indexer[exemplar]
            exemplar_pst = pseudostrengths[exemp_idx_as_trans]
            exemplar_mult = cog(exemplar_pst)
            grouped_transitions = groups.get(exemplar)
            if grouped_transitions is None:
                continue
            for trans in grouped_transitions:
                ctrans_idx = transition_indexer.trans_to_idx.get(trans)
                if ctrans_idx is None:
                    continue
                trans_pst = pseudostrengths[ctrans_idx]
                trans_mult = cog(trans_pst)
                delt_log_strength = trans_mult-exemplar_mult
                delt_log_strength = min(3.0, delt_log_strength)
                rel_strengths.append(np.power(10.0, delt_log_strength))
                col_idxs.append(exemplar_idx)
                row_idxs.append(ctrans_idx)
        out_shape = (len(transition_indexer), len(exemplars))
        return scipy.sparse.csr_matrix((rel_strengths, (row_idxs, col_idxs)), shape=out_shape)

class CollapsedFeatureMatrixModel(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"GroupedFeatureMatrixModel",
    }
    
    def __init__(self, output_p, feature_matrix_p, grouping_matrix_p):
        self.output_p = output_p
        self.add_parameter("feature_matrix", feature_matrix_p)
        self.add_parameter("grouping_matrix", grouping_matrix_p)
    
    def __call__(self, vprep=None):
        vdict = self.get_vdict()
        fmat = vdict[self.inputs["feature_matrix"]]
        gmat = vdict[self.inputs["grouping_matrix"]]
        return fmat*gmat

class NormalizedFluxModel(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"NormalizedFluxModel",
    }
    
    def __init__(self, output_p, feature_matrix_p, strengths_p):
        self.output_p = output_p
        self.add_parameter("feature_matrix", feature_matrix_p)
        self.add_parameter("strengths", strengths_p)
    
    def __call__(self, vprep=None):
        vdict = self.get_vdict(vprep)
        fmat = vdict[self.inputs["feature_matrix"]]
        strengths = vdict[self.inputs["strengths"]]
        return 1.0-(fmat*strengths)


class VoigtFitSingleLineSynthesis(ThimblesTable, Base):
    """table for caching the values of running MOOG to synthesize an individual
transition then fitting the resultant spectrum with a voigt profile to determine a sigma, gamma and equivalent width.
    """
    _transition_id = Column(Integer, ForeignKey("Transition._id"))
    transition = relationshp = relationship("Transition")
    
    ew = Column(Float)
    sigma = Column(Float)
    gamma = Column(Float)
    rms = Column(Float)
    
    #class attributes
    _teff_delta = 250 #grid points in teff
    _logg_delta = 0.5 #grid spacing in logg
    _metalicity_delta = 0.5 #grid spacing in [Fe/H]
    _vmicro_delta = 0.1 #grid spacing in microturbulence
    _delta_lambda = 0.01
    _opac_rad = 10.0
    
    engine_instance = mooger
    
    def __init__(self, transition, teff, logg, metalicity, vmicro, x_on_fe):
        self.transtion = transition
        abund = Abundance(self.transition.ion)
        self.stellar_parameters = StellarParameters(teff, logg, metalicity, vmicro, abundances=[abund])
        cent_wv = self.transition.wv
        min_wv = cent_wv - self._opac_rad
        max_wv = cent_wv + self._opac_rad
        targ_npts = int((max_wv-min_wv)/self._delta_lambda)
        targ_wvs = np.linspace(min_wv, max_wv, targ_npts)
        synth = self.engine_instance.spectrum(
            linelist=[self.transition],
            stellar_params=self.stellar_parameters,
            wavelengths=targ_wvs,
            sampling_mode="bounded",
            delta_wv=self._delta_lambda,
            opac_rad=self._opac_rad,
            central_intensity=False,
        )
        depths = 1.0-synth.flux
        fres = tmb.utils.misc.unweighted_voigt_fit(synth.wvs, depths)
        self.sigma, self.gamma, self.ew = fres
        #import pdb; pdb.set_trace()
