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
        mod_wv_param = self.inputs["model_wvs"][0]
        mod_wvs = vdict[mod_wv_param]
        teff = vdict[self.inputs["teff"][0]]
        vmicro = vdict[self.inputs["vmicro"][0]]
        gamma  = vdict[self.inputs["gamma"][0]]
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
        "polymorphic_identity":"GroupedFeatureMatrixModel"
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
        teff = vdict[self.inputs["teff"][0]]
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
        fmat = vdict[self.inputs["feature_matrix"][0]]
        gmat = vdict[self.inputs["grouping_matrix"][0]]
        return fmat*gmat
    
    #def fit_offsets(self):
    #    #import pdb; pdb.set_trace()
    #    species_gb = self.fdat.groupby("species_group")
    #    groups = species_gb.groups
    #    #order the species keys so we do the species with the most exemplars first
    #    #num_exemplars = sorted([(len(groups[k]), k) for k in groups.keys()], reverse=True)
    #    fallback_offset = 10.0 #totally arbitrary
    #    for species_key in range(len(groups)):
    #        species_df = self.fdat.ix[groups[species_key]]
    #        exemplars = species_df[species_df.group_exemplar > 0]
    #        delta_rews = np.log10(exemplars.ew/exemplars.doppler_width)
    #        x_deltas = exemplars.x.values - self.cog.inverse(delta_rews.values)
    #        offset = np.sum(x_deltas)
    #        if np.isnan(offset):
    #            offset = fallback_offset
    #        self.fdat.ix[groups[species_key], "x_offset"] = offset


class EWParameter(Parameter):
    _id = Column(Integer, ForeignKey("Parameter._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"EWParameter",
    }
    _value = Column(Float)
    _transition_id = Column(Integer, ForeignKey("Transition._id"))
    trasition = relationship("Transition")
    
    def __init__(self, ew, transition):
        self._value = ew
        self.transition=transition


class FeatureGroupModel(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"FeatureGroupModel",
    }
    _stellar_parameters_id = Column(Integer, ForeignKey("StellarParameters._id"))
    stellar_parameters = relationship("StellarParameters")
    _transition_group_id = Column(Integer, ForeignKey("TransitionGroup._id"))
    transition_group = relationship("TransitionGroup")
    
    def __init__(
            self,
            wv_soln,
            transition_group,
            stellar_parameters,
            ew=None,
            exemplar=None,
            share_gamma=True,
    ):
        self.transition_group = transition_group
        transitions = transition_group.transitions
        if ew is None:
            ew = 0.01
        if len(transitions) > 0:
            if exemplar is None:
                exemplar_idx = np.random.randint(len(transitions))
                exemplar = transitions[exemplar_idx]
        
        ew_p = EWParameter(ew, transition=exemplar)
        self.add_input("ew", ew_p)
        self.stellar_parameters = stellar_parameters
        profile_models = []
        for trans in transitions:
            gamma_p = None
            if len(profile_models) > 0:
                if share_gamma:
                    gamma_p ,= profile_models[-1].inputs["gamma"]
            pmod = VoigtProfileModel(
                wv_soln=wv_soln, 
                transition=trans,
                stellar_parameters=stellar_parameters,
                gamma=gamma_p
            )
            profile_models.append(pmod)
            self.add_input("profiles", pmod.output_p)
        
        min_start = np.min([pmod.output_p.wv_sample.start for pmod in profile_models])
        max_end = np.max([pmod.output_p.wv_sample.end for pmod in profile_models])
        wvsamp = WavelengthSample(wv_soln, start=min_start, end=max_end)
        self.output_p = FluxParameter(wvsamp)
    
    def __call__(self, vprep=None):
        vdict = self.get_vdict(vprep)
        profile_ps = self.inputs["profiles"]
        prof_strengths = np.zeros(len(profile_ps))
        stell_ps = self.stellar_parameters
        ew_p ,= self.inputs["ew"]
        exemplar = ew_p.transition
        exemplar_pst = exemplar.pseudo_strength(stellar_parameters=stell_ps)
        for prof_idx in range(len(profile_ps)):
            profile_param = profile_ps[prof_idx]
            transition = profile_param.mapped_models[0].transition
            pstrength = transition.pseudo_strength(stellar_parameters=stell_ps)
            pstrength -= exemplar_pst #normalize to exemplar
            prof_strengths[prof_idx] = pstrength
        prof_strengths= np.power(10.0, prof_strengths)
        output_sample = self.output_p.wv_sample
        fsum = np.zeros(len(output_sample))
        out_start = output_sample.start
        for prof_idx in range(len(profile_ps)):
            profile_p = profile_ps[prof_idx]
            start= profile_p.wv_sample.start - out_start
            end = profile_p.wv_sample.end - out_start
            profile = prof_strengths[prof_idx]*profile_p.value
            fsum[start:end] -= profile
        ew_val = vdict[ew_p]
        fsum *= ew_val
        return fsum


class VoigtProfileModel(Model):
    """a model for a voigt profile with the gaussian width determined 
    by assuming a sqrt(thermal**2 + vmicro**2) = sigma, and a free 
    gamma parameter.
    """
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"VoigtProfileModel",
    }
    _transition_id = Column(Integer, ForeignKey("Transition._id"))
    transition = relationship("Transition")
    max_width = Column(Float)
    
    def __init__(
            self, 
            wv_soln, 
            transition,
            stellar_parameters,
            max_width=None,
            gamma=0.0,
    ):
        self.transition=transition
        cent_wv = transition.wv
        if max_width is None:
            max_width = cent_wv*5e-4
        self.max_width = max_width
        if not isinstance(gamma, Parameter):
            gamma_param = gamma
        else:
            gamma_param = FloatParameter(float(gamma))
        self.add_input("gamma", gamma_param)
        teff_p = stellar_parameters.teff_p
        self.add_input("teff", teff_p)
        vmicro_p = stellar_parameters.vmicro_p
        self.add_input("vmicro", vmicro_p)
        start, end = wv_soln.get_index([cent_wv-self.max_width, cent_wv+self.max_width], clip=True, snap=True)
        wvsamp = WavelengthSample(wv_soln, start=start, end=end)
        self.output_p = FluxParameter(wvsamp)
    
    def __call__(self, vprep=None):
        vdict = self.get_vdict(vprep)
        teff_p ,= self.inputs["teff"]
        teff_val = vdict[teff_p]
        vmicro_p ,= self.inputs["vmicro"]
        vmicro_val = vdict[vmicro_p]
        twv = self.transition.wv
        weight = self.transition.ion.weight
        therm_sig = tmb.utils.misc.thermal_width(teff_val, twv, weight)
        vmicro_sig = (vmicro_val/speed_of_light)*twv
        sigma_val = np.sqrt(therm_sig**2 + vmicro_sig**2)
        gamma_p ,= self.inputs["gamma"]
        gamma_val = vdict[gamma_p]
        wvs = self.output_p.wv_sample.wvs
        try:
            pflux = voigt(wvs, twv, sigma_val, gamma_val)
        except:
            import pdb; pdb.set_trace()
        return pflux


class GroupedFeaturesModel(tmb.modeling.factor_models.FluxSumLogic, Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"GroupedFeaturesModel",
    }
    _star_id = Column(Integer, ForeignKey("Source._id"))
    star = relationship("Star")
        
    def __init__(self, star, grouping_standard, wv_soln):
        self.star = star
        wv_soln = tmb.as_wavelength_solution(wv_soln)
        self.output_p = FluxParameter(wv_soln)
        for group in grouping_standard.groups:
            fgm = FeatureGroupModel(
                wv_soln=wv_soln,
                transition_group=group,
                stellar_parameters=star.stellar_parameters,
            )
            self.add_input("group_fluxes", fgm.output_p)


class VoigtFitSingleLineSynthesis(ThimblesTable, Base):
    """table for caching the values of running MOOG to synthesize an individual
transition then fitting the resultant spectrum with a voigt profile to determine a sigma, gamma and equivalent width.
    """
    _transition_id = Column(Integer, ForeignKey("Transition._id"))
    transition = relationshp = relationship("Transition")
    _stellar_parameters_id = Column(Integer, ForeignKey("StellarParameters._id"))
    stellar_parameters = relationship("StellarParameters")
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




