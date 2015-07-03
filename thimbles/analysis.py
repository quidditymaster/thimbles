import numpy as np
import scipy.sparse

import thimbles as tmb
from thimbles.sqlaimports import *
from thimbles.thimblesdb import ThimblesTable, Base
from thimbles.modeling.associations import HasParameterContext
from thimbles.modeling import FloatParameter, Parameter

class SharedParameterSpace(ThimblesTable, Base, HasParameterContext):
    name = Column(String, unique=True)
    
    def __init__(self, name):
        HasParameterContext.__init__(self)
        self.name = name

def make_shared_parameters(
        space_name,
        min_wv,
        max_wv,
        linelist,
        grouping_dict,
        model_resolution=2e5,
        n_tellurics=1,
):
    sps = SharedParameterSpace(name=space_name)
    
    n_model_pts = int(np.log(max_wv/min_wv)*model_resolution)+1
    model_wvs = np.exp(np.linspace(np.log(min_wv), np.log(max_wv), n_model_pts))
    model_wv_indexer_p = tmb.modeling.Parameter()
    model_wv_indexer_mod = tmb.coordinatization.LogLinearCoordinatizationModel(
        output_p=model_wv_indexer_p,
        min_p=FloatParameter(min_wv),
        max_p=FloatParameter(max_wv),
        npts=n_model_pts
    )
    sps.add_parameter("model_wvs", model_wv_indexer_p)
    
    #indexer parameter
    tidx_p = tmb.transitions.TransitionIndexerParameter(linelist)
    sps.add_parameter("transition_indexer", tidx_p)
    
    #transition wavelengths vector
    twvs_p = tmb.modeling.Parameter()
    twvs_mod = tmb.features.TransitionWavelengthVectorModel(
        output_p = twvs_p,
        indexer_p = tidx_p,
    )
    sps.add_parameter("transition_wvs", twvs_p)
    
    #transition molecular weights vector
    tmolw_p = tmb.modeling.Parameter()
    tmolw_mod = tmb.features.IonWeightVectorModel(
        output_p = tmolw_p,
        indexer_p = tidx_p,
    )
    sps.add_parameter("ion_weights", tmolw_p)
    
    #exemplar grouping parameter
    grouping_p = tmb.transitions.ExemplarGroupingParameter(
        groups=grouping_dict
    )
    sps.add_parameter("grouping", grouping_p)
    
    #exemplar indexer
    sorter_func = lambda t: (t.ion.z, t.ion.charge, t.wv)
    sorted_exemplars = sorted(list(grouping_p.value.exemplars()), key=sorter_func)
    exemp_index_p = tmb.transitions.TransitionIndexerParameter(sorted_exemplars)
    sps.add_parameter("exemplar_indexer", exemp_index_p)
    
    
    #telluric opacity vectors
    telluric_opac_basis = scipy.sparse.csc_matrix(np.zeros((len(model_wvs), n_tellurics)))
    telluric_opac_basis_p = tmb.modeling.PickleParameter(telluric_opac_basis)
    sps.add_parameter("telluric_opacity_basis", telluric_opac_basis_p)
    
    return sps

def star_modeler(
        star, 
        database, 
        shared_parameters,
):
    teff_p = star.context["teff"]
    vmicro_p = star.context["vmicro"]
    vmacro_p = star.context["vmacro"]
    vsini_p = star.context["vsini"]
    ldark_p = star.context["ldark"]
    transition_indexer_p = shared_parameters.context["transition_indexer"]
    exemplar_indexer_p = shared_parameters.context["exemplar_indexer"]
    grouping_p = shared_parameters.context["grouping"]
    model_wvs_p = shared_parameters.context["model_wvs"]
    transition_wvs_p = shared_parameters.context["transition_wvs"]
    ion_weights_p = shared_parameters.context["ion_weights"]
    
    #sigma model
    sigma_vec_p = tmb.modeling.Parameter()
    sig_mod = tmb.features.SigmaModel(
        output_p = sigma_vec_p,
        vmicro_p = vmicro_p,
        transition_wvs_p=transition_wvs_p,
        ion_weights_p=ion_weights_p,
        teff_p=teff_p,
    )
    star.add_parameter("transition_sigma", sigma_vec_p)
    
    #gamma model
    vec_gamma_p = tmb.modeling.Parameter()
    ref_gamma_p = tmb.modeling.FloatParameter(0.01)
    gam_mod = tmb.features.GammaModel(
        output_p = vec_gamma_p,
        gamma_p=ref_gamma_p,
        transition_wvs_p=transition_wvs_p,
    )
    star.add_parameter("transition_gamma", vec_gamma_p)
    star.add_parameter("reference_gamma", ref_gamma_p)
    
    ion_correction_p = tmb.abundances.IonMappedParameter()
    star.add_parameter("ion_correction", ion_correction_p)
    
    pseudostrength_p = tmb.modeling.Parameter()
    tmb.cog.PseudoStrengthModel(
        output_p=pseudostrength_p,
        transition_indexer_p=transition_indexer_p,
        ion_correction_p=ion_correction_p,
        teff_p=teff_p,
    )
    star.add_parameter("pseudostrength", pseudostrength_p)
    
    #saturation model
    saturation_vec_p = tmb.modeling.Parameter()
    saturation_offset_p = tmb.modeling.FloatParameter(0.5)
    tmb.cog.SaturationModel(
        output_p=saturation_vec_p,
        pseudostrength_p=pseudostrength_p,
        offset_p=saturation_offset_p
    )
    star.add_parameter("transition_saturation", saturation_vec_p)
    star.add_parameter("saturation_offset", saturation_offset_p)
    
    #feature matrix model
    feature_mat_p = tmb.modeling.Parameter()
    feature_mat_mod = tmb.features.FeatureMatrixModel(
        output_p=feature_mat_p,
        model_wv_p=model_wvs_p,
        centers_p=transition_wvs_p,
        sigma_p=sigma_vec_p,
        gamma_p=vec_gamma_p,
        saturation_p=saturation_vec_p,
    )
    star.add_parameter("feature_matrix", feature_mat_p)
    
    #grouping matrix model
    grouping_matrix_p = tmb.modeling.Parameter()
    grouping_matrix_mod = tmb.features.RelativeStrengthMatrixModel(
        output_p = grouping_matrix_p,
        grouping_p = grouping_p,
        transition_indexer_p = transition_indexer_p,
        exemplar_indexer_p = exemplar_indexer_p,
        pseudostrength_p=pseudostrength_p,
    )
    star.add_parameter("grouping_matrix", grouping_matrix_p)
    
    #collapse the feature matrix
    cfm_p = tmb.modeling.Parameter()
    cfm_mod = tmb.features.CollapsedFeatureMatrixModel(
        output_p = cfm_p,
        feature_matrix_p=feature_mat_p,
        grouping_matrix_p=grouping_matrix_p,
    )
    star.add_parameter("cfm", cfm_p)
    
    #equivalent width parameters
    ew_dict_p = tmb.transitions.TransitionMappedParameter()
    ew_vec_p = tmb.modeling.Parameter()
    ew_vec_mod = tmb.transitions.TransitionMappedVectorizerModel(
        output_p = ew_vec_p,
        transition_mapped_p=ew_dict_p,
        indexer_p=exemplar_indexer_p,
        fill_value=0.0
    )
    star.add_parameter("ew_vec", ew_vec_p)
    star.add_parameter("ew_dict", ew_dict_p)
    
    #subtract ews from 1.0
    pre_broadened_flux_p = tmb.modeling.Parameter()
    tmb.features.NormalizedFluxModel(
        output_p = pre_broadened_flux_p,
        feature_matrix_p=cfm_p,
        strengths_p = ew_vec_p,
    )
    star.add_parameter("pre_broadened_flux", pre_broadened_flux_p)
    
    #macroturbulent + rotational broadening matrix model
    bmat_p = tmb.modeling.Parameter()
    bmat_mod = tmb.rotation.BroadeningMatrixModel(
        output_p=bmat_p,
        model_wvs_p=model_wvs_p,
        vmacro_p=vmacro_p,
        vsini_p=vsini_p,
        ldark_p=ldark_p,
    )
    star.add_parameter("broadening_matrix", bmat_p)
    
    model_flux_p = tmb.modeling.Parameter()
    flux_mod = tmb.modeling.MatrixMultiplierModel(
        output_p = model_flux_p,
        matrix_p = bmat_p,
        vector_p = pre_broadened_flux_p,
    )
    star.add_parameter("model_flux", model_flux_p)
    

def pointing_modeler(pointing, database, shared_parameters):
    opac_basis_p = shared_parameters.context["telluric_opacity_basis"]
    ntell = opac_basis_p.value.shape[1]
    telluric_coeffs_p = tmb.modeling.PickleParameter(np.zeros((ntell,)))
    telluric_opacity_p = tmb.modeling.Parameter()
    telluric_opacity_mod = tmb.modeling.MatrixMultiplierModel(
        output_p = telluric_opacity_p,
        matrix_p = opac_basis_p,
        vector_p = telluric_coeffs_p,
    )
    pointing.add_parameter("telluric_opacity", telluric_opacity_p)
    
    telluric_transmission_p = tmb.modeling.Parameter()
    telluric_transmission_mod = tmb.tellurics.TransmissionModel(
        output_p = telluric_transmission_p,
        opacity_p = telluric_opacity_p,
    )
    pointing.add_parameter("telluric_transmission", telluric_transmission_p)


def observation_modeler(observation, database, shared_parameters):
    source = observation.source
    pointing = observation.pointing
    rv_p = source.context["rv"]
    delta_helio_p = pointing.context["delta_helio"]
    model_wvs_p = shared_parameters.context["model_wvs"]
    
    shift_matrix_p = tmb.modeling.Parameter()
    telluric_shift_matrix_mod = tmb.tellurics.TelluricShiftMatrixModel(
        output_p=shift_matrix_p,
        model_wvs_p = model_wvs_p,
        rv_p = rv_p,
        delta_helio_p = delta_helio_p
    )
    observation.add_parameter("telluric_shift_matrix", shift_matrix_p)


def spectrum_modeler(spectrum, database, shared_parameters):
    flux_p = spectrum.flux_p
    observation = spectrum.observation
    source = observation.source
    pointing = observation.pointing
    
    model_wvs_p = shared_parameters.context["model_wvs"]
    telluric_transmission_p = pointing.context["telluric_transmission"]
    source_model_flux_p = source.context["model_flux"]
    telluric_shift_matrix_p = observation.context["telluric_shift_matrix"]
    
    shifted_telluric_p = tmb.modeling.Parameter()
    shift_matrix_mod = tmb.modeling.MatrixMultiplierModel(
        output_p = shifted_telluric_p,
        matrix_p = telluric_shift_matrix_p,
        vector_p = telluric_transmission_p,
    )
    spectrum.add_parameter("shifted_telluric", shifted_telluric_p)
    
    post_atmosphere_p = tmb.modeling.Parameter()
    post_atmosphere_mod = tmb.modeling.MultiplierModel(
        output_p = post_atmosphere_p,
        factors=[
            source_model_flux_p,
            shifted_telluric_p,
        ]
    )
    spectrum.add_parameter("post_atmosphere_flux", post_atmosphere_p)
    
    resampling_matrix_p = tmb.modeling.Parameter()
    model_wv_lsf_p = tmb.modeling.FloatParameter(1.0)
    tmb.spectrographs.SamplingModel(
        output_p = resampling_matrix_p,
        input_wvs_p=model_wvs_p,
        input_lsf_p=model_wv_lsf_p,
        output_wvs_p=spectrum.context["rest_wvs"],
        output_lsf_p=spectrum.context["lsf"],
    )
    spectrum.add_parameter("sampling_matrix", resampling_matrix_p)
    
    resampled_flux_p = tmb.modeling.Parameter()
    tmb.modeling.MatrixMultiplierModel(
        output_p=resampled_flux_p,
        matrix_p=resampling_matrix_p,
        vector_p=post_atmosphere_p,
    )
    spectrum.add_parameter("sampled_flux", resampled_flux_p)
    
    pseudo_norm = tmb.pseudonorms.sorting_norm(spectrum)
    final_norm_p = tmb.modeling.Parameter(pseudo_norm)
    final_norm_model = tmb.modeling.PixelPolynomialModel(
        output_p=final_norm_p,
    )
    spectrum.add_parameter("norm", final_norm_p)
    
    comparison_mod = tmb.modeling.MultiplierModel(
        output_p = spectrum.flux_p,
        factors=[
            resampled_flux_p,
            final_norm_p,
        ]
    )
    spectrum.add_parameter("obs_flux", spectrum.flux_p)


def make_all_models(
        spectra,
        database, 
        shared_parameters,
        source_modeler=None,
        pointing_modeler=None,
        observation_modeler=None,
        spectrum_modeler=None,
        estimator_builder=None,
):
    database.add(shared_parameters)
    
    source_set = set()
    pointing_set = set()
    obs_set = set()
    
    for spec in spectra:
        cobs = spec.observation
        obs_set.add(cobs)
        source_set.add(cobs.source)
        pointing_set.add(cobs.pointing)
    
    sources = list(source_set)
    pointings = list(pointing_set)
    observations = list(obs_set)
    
    for source in sources:
        source_modeler(source, database, shared_parameters)
    
    for pointing in pointings:
        pointing_modeler(pointing, database, shared_parameters)
    
    for obs in observations:
        observation_modeler(obs, database, shared_parameters)
    
    for spec in spectra:
        spectrum_modeler(spec, database, shared_parameters)
    
    database.commit()

