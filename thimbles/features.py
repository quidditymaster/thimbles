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
from thimbles.transitions import TransitionGroupingStandard, Transition
from thimbles.modeling import Model, Parameter
from thimbles.modeling import factor_models
from thimbles.spectrum import FluxParameter, WavelengthSample
from thimbles.utils.misc import smooth_ppol_fit
import thimbles.utils.piecewise_polynomial as ppol
from thimbles import hydrogen
from thimbles import as_wavelength_sample
from thimbles.thimblesdb import ThimblesTable, Base
from thimbles.sqlaimports import *
from thimbles.utils.misc import saturated_voigt_cog
from thimbles.sqlaimports import *
from thimbles.radtran import mooger


def voigt_feature_matrix(wavelengths, centers, sigmas, gammas=None, indexer=None):
    indexes = []
    profiles = []
    n_features = len(centers)
    if gammas is None:
        gammas = np.zeros(n_features)
    assert len(sigmas) == n_features
    assert len(gammas) == n_features
    window_deltas = 35.0*gammas
    alt_wid = 5.0*np.sqrt(sigmas**2 + gammas**2)
    window_deltas = np.where(window_deltas > alt_wid, window_deltas, alt_wid)
    if indexer is None:
        indexer = tmb.coordinatization.as_coordinatization(wavelengths)
    for col_idx in range(n_features):
        ccent = centers[col_idx]
        csig = sigmas[col_idx]
        cgam = gammas[col_idx]
        cdelt = window_deltas[col_idx]
        lb, ub = indexer.get_index([ccent-cdelt, ccent+cdelt], clip=True, snap=True)
        prof = voigt(wavelengths[lb:ub+1], ccent, csig, cgam)
        indexes.append(np.array([np.arange(lb, ub+1), np.repeat(col_idx, len(prof))]))
        profiles.append(prof)
    indexes = np.hstack(indexes)
    profiles = np.hstack(profiles)
    npts = len(wavelengths)
    mat = scipy.sparse.csc_matrix((profiles, indexes), shape=(npts, n_features))
    return mat

class SharedGammaFeatureMatrixModel(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"SharedGammaFeatureMatrixModel",
    }
    _grouping_standard_id = Column(Integer, ForeignKey("TransitionGroupingStandard._id"))
    grouping_standard = relationship("TransitionGroupingStandard")
    gamma_anchor_wv = Column(Float)
    
    def __init__(
            self,
            output_p,
            grouping_standard, 
            model_wv_soln,
            teff_p,
            vmicro_p,
            gamma_p,
            gamma_anchor_wv=5000.0
    ):
        self.output_p = output_p
        self.grouping_standard = grouping_standard
        self.add_input("model_wvs", model_wv_soln.indexer.output_p)
        self.add_input("teff", teff_p)
        self.add_input("vmicro", vmicro_p)
        self.add_input("gamma", gamma_p)
        self.gamma_anchor_wv = gamma_anchor_wv
    
    def __call__(self, vprep=None):
        vdict = self.get_vdict(vprep)
        mod_wv_param = self.inputs["model_wvs"]
        mod_wvs = vdict[mod_wv_param]
        teff = vdict[self.inputs["teff"]]
        vmicro = vdict[self.inputs["vmicro"]]
        gamma  = vdict[self.inputs["gamma"]]
        gaw = self.gamma_anchor_wv 
        fmats = []
        indexer = mod_wv_param.mapped_models[0]
        for group in self.grouping_standard:
            t_wvs = np.array([t.wv for t in group.transitions])
            mol_weight = group.transitions[0].ion.weight
            therm_widths = tmb.utils.misc.thermal_width(teff, t_wvs, mol_weight)
            vmic_widths = t_wvs*vmicro/tmb.speed_of_light
            sigmas = np.sqrt(therm_widths**2 + vmic_widths**2)
            gammas = gamma*(t_wvs/self.gamma_anchor_wv)**2
            fmats.append(voigt_feature_matrix(mod_wvs, centers=t_wvs, sigmas=sigmas, gammas=gammas, indexer=indexer))
        return scipy.sparse.hstack(fmats)


class RelativeStrengthMatrixModel(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"RelatveStrengthMatrixModel"
    }
    _grouping_standard_id = Column(Integer, ForeignKey("TransitionGroupingStandard._id"))
    grouping_standard = relationship("TransitionGroupingStandard")
    
    def __init__(
            self,
            output_p,
            grouping_standard,
            teff_p,
    ):
        self.output_p = output_p
        self.grouping_standard = grouping_standard
        self.add_input("teff", teff_p)
    
    def get_exemplars(self):
        return [g.exemplar for g in self.grouping_standard.groups]
    
    def __call__(self, vprep=None):
        vdict = self.get_vdict(vprep)
        teff = vdict[self.inputs["teff"]]
        row_idxs = [] #transition indexes
        col_idxs = [] #group indexes
        rel_strengths = []
        ctrans_idx = 0
        theta = 5040.0/teff
        groups = self.grouping_standard.groups
        for group_idx, group in enumerate(groups):
            exemp = group.exemplar
            exemplar_strength = np.log10(exemp.wv) + exemp.loggf - theta*exemp.ep
            for trans in group.transitions:
                trans_strength = np.log10(trans.wv) + trans.loggf - theta*trans.ep
                delt_log_strength = trans_strength-exemplar_strength
                rel_strengths.append(np.power(10.0, delt_log_strength))
                col_idxs.append(group_idx)
                row_idxs.append(ctrans_idx)
                ctrans_idx += 1
        out_shape = (ctrans_idx, len(groups))
        return scipy.sparse.csr_matrix((rel_strengths, (row_idxs, col_idxs)), shape=out_shape)

class GroupedFeatureMatrixModel(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"GroupedFeatureMatrixModel",
    }
    
    def __init__(self, output_p, feature_matrix_p, grouping_matrix_p):
        self.output_p = output_p
        self.add_input("feature_matrix", feature_matrix_p)
        self.add_input("grouping_matrix", grouping_matrix_p)
    
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
    
    def __init__(self, output_p, group_flux_p, residual_flux_p):
        self.output_p = output_p
        self.add_input("group_flux", group_flux_p)
        self.add_input("residual_flux", residual_flux_p)
    
    def __call__(self, vprep=None):
        vdict = self.get_vdict(vprep)
        groups_flux = vdict[self.inputs["group_flux"]]
        residual_flux = vdict[self.inputs["residual_flux"]]
        return 1.0-groups_flux-residual_flux


def fit_offsets():
    num_exemplars = sorted([(len(groups[k]), k) for k in groups.keys()], reverse=True)
    fallback_offset = 10.0 #totally arbitrary
    for species_key in range(len(groups)):
        species_df = self.fdat.ix[groups[species_key]]
        exemplars = species_df[species_df.group_exemplar > 0]
        delta_rews = np.log10(exemplars.ew/exemplars.doppler_width)
        x_deltas = exemplars.x.values - self.cog.inverse(delta_rews.values)
        offset = np.sum(x_deltas)
        if np.isnan(offset):
            offset = fallback_offset
        self.fdat.ix[groups[species_key], "x_offset"] = offset


class TransitionIndexer(object):
    pass

class TransitionVectorizer(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"TransitionMappedVectorizer"
    }
    

class HasTransition(object):
    
    @declared_attr
    def _transition_id(self):
        _transition_id = Column(Integer, ForeignKey("Transition._id"))
    
    @declared_attr
    def transition(self):
        trasition = relationship("Transition")

class EWParameter(Parameter, HasTransition):
    _id = Column(Integer, ForeignKey("Parameter._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"EWParameter",
    }
    _value = Column(Float)
    
    def __init__(self, ew, transition):
        self._value = ew
        self.transition=transition


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




