
import numpy as np
import thimbles as tmb
from thimbles.modeling import Parameter, FloatParameter, PickleParameter
from thimbles.analysis import ModelComponentTemplate, ParameterSpec, ModelSpec 

lsf_degree = 1
normalization_degree=5
min_model_wv = 15161
max_model_wv = 16932
model_resolution = 2e5
npts_model = int(model_resolution*np.log(max_model_wv/min_model_wv))+1


def get_spectrum_len(contexts):
    return len(contexts["spectrum"])

def get_npts_chip(contexts):
    return contexts["chip"]["wvs"].value.npts

def dirac_vec(n, i):
    vec = np.zeros(n, dtype=float)
    vec[i] = 1.0
    return vec


model_wvs_mct = ModelComponentTemplate(
    parameter_specs={
        "min_wv":ParameterSpec(
            factory=FloatParameter,
            kwargs={"value":min_model_wv},
        ),
        "max_wv":ParameterSpec(
            factory=FloatParameter,
            kwargs={"value":max_model_wv}
        ),
        "model_wvs":ParameterSpec(
            factory=Parameter,
            push_to="global.model_wvs"
        ),
        "model_lsf":ParameterSpec(FloatParameter, kwargs={"value":0.0}, push_to="global.model_lsf")
    },
    model_specs = {
        "wavelength_solution":ModelSpec(
            factory=tmb.coordinatization.LogLinearCoordinatizationModel,
            inputs = {
                "min":"min_wv",
                "max":"max_wv",
            },
            output="model_wvs",
            kwargs={"npts":npts_model},
        ),
    }
)


chip_sampling_mct = ModelComponentTemplate(
    parameter_specs={
        "lsf":ParameterSpec(factory=Parameter, push_to="chip.lsf"),
        "lsf_coeffs":ParameterSpec(
            PickleParameter, 
            kwargs={"value":lambda x: dirac_vec(lsf_degree+1, -1)}, 
            push_to="chip.lsf_coeffs"
        ),
        "spectrum_wvs":ParameterSpec("chip.wvs"),
        "model_wvs":ParameterSpec("global.model_wvs"),
        "model_lsf":ParameterSpec("global.model_lsf"),
        "sampling_matrix":ParameterSpec(Parameter, push_to="chip.sampling_matrix")
    },
    model_specs={
        "lsf_poly":ModelSpec(
            tmb.modeling.PixelPolynomialModel,
            inputs={"coeffs":"lsf_coeffs"},
            kwargs={"npts":get_npts_chip},
            output="lsf",
        ),
        "sampling_matrix":ModelSpec(
            tmb.spectrographs.SamplingModel,
            inputs={
                "input_wvs":"model_wvs",
                "input_lsf":"model_lsf",
                "output_wvs":"spectrum_wvs",
                "output_lsf":"lsf",
            },
            output="sampling_matrix",
        )
    }
)


def extract_pseudonorm_coeffs(spec_context):
    spec = spec_context['spectrum']
    psnorm = spec.pseudonorm()
    npts = len(spec)
    xvals = (np.arange(npts) - 0.5*npts)/(0.5*npts)
    pseudonorm_coeffs = np.polyfit(xvals, psnorm, deg=normalization_degree)
    return pseudonorm_coeffs


normalization_mct = ModelComponentTemplate(
    parameter_specs={
        "norm_coeffs":ParameterSpec(
            factory=PickleParameter, 
            kwargs={"value":extract_pseudonorm_coeffs}, 
            push_to="spectrum.norm_coeffs"
        ),
        "norm":ParameterSpec(
            Parameter, 
            push_to="spectrum.norm"
        ),
    },
    model_specs={
        "norm":ModelSpec(
            tmb.modeling.PixelPolynomialModel,
            inputs={
                "coeffs":"norm_coeffs",
            },
            kwargs={"npts":get_spectrum_len,},
            output="norm"
        ),
    }
)


star_mct = ModelComponentTemplate(
    parameter_specs = {
        "feature_flux":ParameterSpec(factory=Parameter, push_to="star.feature_flux"),
        "transition_indexer":ParameterSpec("global.transition_indexer"),
        "model_wvs":ParameterSpec("global.model_wvs"),
        "exemplar_map":ParameterSpec("global.exemplar_map"),
        "exemplar_indexer":ParameterSpec("global.exemplar_indexer"),
        "transition_wvs":ParameterSpec("global.transition_wvs"),
        "transition_molecular_weights":ParameterSpec("global.transition_molecular_weights"),
        "transition_ep":ParameterSpec("global.transition_ep"),
        "profile_matrix":ParameterSpec(factory=Parameter, push_to="star.profile_matrix"),
        "feature_matrix":ParameterSpec(factory=Parameter, push_to="star.feature_matrix"),
        "strength_matrix":ParameterSpec(factory=Parameter, push_to="star.strength_matrix"),
        "sigmas":ParameterSpec(factory=Parameter, push_to="star.sigmas"),
        "gammas":ParameterSpec(factory=Parameter, push_to="star.gammas"),
        "pseudostrengths":ParameterSpec(factory=Parameter, push_to="star.pseudostrengths"),
        "saturations":ParameterSpec(factory=Parameter, push_to="star.saturations"),
        "cog_ew_adjust":ParameterSpec(FloatParameter, kwargs={"value":0.0}, push_to="star.cog_ew_adjust"),
        "saturation_offset":ParameterSpec(
            factory=FloatParameter,
            kwargs={"value":2.0},
            push_to="star.saturation_offset",
        ),
        "teff":ParameterSpec("star.teff"),
        "logg":ParameterSpec("star.logg"),
        "vmicro":ParameterSpec("star.vmicro"),
        "thermalized_widths_vec":ParameterSpec(
            PickleParameter,
            push_to="star.thermalized_widths_vec",
            kwargs={"value":lambda x: np.zeros(len(x["global"]["exemplar_indexer"].value))},
        ),
        "ion_shifts":ParameterSpec(tmb.abundances.IonMappedParameter, push_to="star.ion_shifts"),
        "cog":ParameterSpec(Parameter, push_to="star.cog"),
        "gamma_coeff_dict":ParameterSpec("global.gamma_coeff_dict")
    },
    model_specs={
        "profile_matrix":ModelSpec(
            tmb.features.ProfileMatrixModel,
            inputs={
                "model_wvs":"model_wvs",
                "centers":"transition_wvs",
                "sigmas":"sigmas",
                "gammas":"gammas",
                "saturations":"saturations",
            },
            output="profile_matrix",
        ),
        "sigmas":ModelSpec(
            tmb.features.SigmaModel,
            inputs={
                "teff":"teff",
                "vmicro":"vmicro",
                "transition_wvs":"transition_wvs",
                "molecular_weights":"transition_molecular_weights",
            },
            output="sigmas",
        ),
        "gammas":ModelSpec(
            tmb.features.GammaModel,
            inputs={
                "logg":"logg",
                "teff":"teff",
                "transition_ep":"transition_ep",
                "coeff_dict":"gamma_coeff_dict"
            },
            output="gammas"
        ),
        "cog":ModelSpec(
            tmb.cog.SaturationCurveModel,
            inputs={
                "sigmas":"sigmas",
                "gammas":"gammas",
            },
            output="cog",
        ),
        "pseudostrengths":ModelSpec(
            tmb.cog.PseudoStrengthModel,
            inputs={
                "transition_indexer":"transition_indexer",
                "ion_correction":"ion_shifts",
                "teff":"teff",
            },
            output="pseudostrengths"
        ),
        "saturations":ModelSpec(
            tmb.cog.SaturationModel,
            inputs={
                "pseudostrengths":"pseudostrengths",
                "saturation_curve":"cog",
                "offset":"saturation_offset",
            },
            output="saturations"
        ),
        "strength_matrix":ModelSpec(
            tmb.features.RelativeStrengthMatrixModel,
            inputs={
                "grouping":"exemplar_map",
                "transition_indexer":"transition_indexer",
                "pseudostrength":"pseudostrengths",
                "row_indexer":"transition_indexer",
                "col_indexer":"exemplar_indexer",
                "cog":"cog",
            },
            output="strength_matrix",
        ),
        "feature_matrix":ModelSpec(
            tmb.features.CollapsedFeatureMatrixModel,
            inputs={
                "feature_matrix":"profile_matrix",
                "grouping_matrix":"strength_matrix",
            },
            output="feature_matrix",
        ),
        "feature_flux":ModelSpec(
            tmb.features.NormalizedFluxModel,
            inputs={
                "feature_matrix":"feature_matrix",
                "strengths":"thermalized_widths_vec",
            },
            output="feature_flux",
        ),
    },
)


source_spec_linker_mct = ModelComponentTemplate(
    parameter_specs={
        "norm":ParameterSpec("spectrum.norm"),
        "sampling_matrix":ParameterSpec("chip.sampling_matrix"),
        "sampled_flux":ParameterSpec(Parameter),
        "obs_flux":ParameterSpec("spectrum.obs_flux"),
        "feature_flux":ParameterSpec("source.feature_flux"),
    },
    model_specs={
        "matrix_multiplier":ModelSpec(
            tmb.modeling.MatrixMultiplierModel,
            inputs={
                "matrix":"sampling_matrix",
                "vector":"feature_flux",
            },
            output="sampled_flux",
        ),
        "norm_multiplier":ModelSpec(
            tmb.modeling.MultiplierModel,
            inputs={
                "factors":["norm", "sampled_flux"]
            },
            output="obs_flux"
        ),
    }
)
