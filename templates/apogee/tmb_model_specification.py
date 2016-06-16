
import numpy as np
import thimbles as tmb
import thimbles.contexts
from thimbles.modeling import Parameter, FloatParameter, PickleParameter
from thimbles.analysis import ModelComponentTemplate, ParameterSpec, ModelSpec
import re
from collections import OrderedDict

lsf_degree = 1
normalization_degree=5
min_model_wv = 15140
max_model_wv = 16970
model_resolution = 3e5
npts_model = int(model_resolution*np.log(max_model_wv/min_model_wv))+1

gamma_coeff_dict = {
    "offset":-2.5,
    "ep":[5.085e-2, 6.5e-3],
    "logg":[1.3722e-1, 2.256e-2],
    "teff":[-1.21e-4],
}

saturation_coeff_dict = {
    "offset":-0.0864,
    "teff":[-8.117e-5]
}

chip_wvs = []
for chip_idx in range(3):
    cwvs = np.loadtxt("coadd_wvs_chip{}.txt".format(chip_idx))
    chip_wvs.append(cwvs)

chip_bounds =[(cw[0], cw[-1]) for cw in chip_wvs]
chip_lsf_widths = [0.9, 0.85, 0.8]#in angstroms
chip_npts = [len(cw) for cw in chip_wvs]

#helper functions
def get_spectrum_len(contexts):
    return len(contexts["spectrum"])

def get_npts_chip(contexts):
    return contexts["chip"]["wvs"].value.npts

def dirac_vec(n, i, val=1.0):
    vec = np.zeros(n, dtype=float)
    vec[i] = val
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
    },
    application_spine = tmb.contexts.global_spine
)

transition_vectorizer_mct = ModelComponentTemplate(
    parameter_specs = {
        "transition_wvs":ParameterSpec(Parameter, push_to="global.transition_wvs"),
        "transition_ep":ParameterSpec(Parameter, push_to="global.transition_ep"),
        "transition_molecular_weights":ParameterSpec(Parameter, push_to="global.transition_molecular_weights"),
        "transition_indexer":ParameterSpec("global.transition_indexer"),
    },
    model_specs={
        "wv_vectorizer":ModelSpec(
            tmb.features.TransitionWavelengthVectorModel,
            inputs={
                "indexer":"transition_indexer"
            },
            output="transition_wvs"
        ),
        "mol_weight_vectorizer":ModelSpec(
            tmb.features.IonWeightVectorModel,
            inputs={
                "indexer":"transition_indexer"
            },
            output="transition_molecular_weights"
        ),
        "ep_vectorizer":ModelSpec(
            tmb.features.TransitionEPVectorModel,
            inputs={
                "indexer":"transition_indexer"
            },
            output="transition_ep"
        ),
    },
    application_spine = tmb.contexts.global_spine
)

gcog_mct = ModelComponentTemplate(
    parameter_specs={
        "gamma_coeff_dict":ParameterSpec(
            PickleParameter,
            push_to="global.gamma_coeff_dict",
            kwargs={"value":gamma_coeff_dict},
        ),
        "saturation_coeff_dict":ParameterSpec(
            PickleParameter,
            push_to="global.saturation_coeff_dict",
            kwargs={"value":saturation_coeff_dict},
        ),
    },
    model_specs={},
    application_spine=tmb.contexts.global_spine
)

ap_pspecs = OrderedDict()
ap_mspecs = OrderedDict()
for i in range(3):
    lsf_pname = "lsf_{}".format(i)
    ap_pspecs[lsf_pname] = ParameterSpec(
        factory=Parameter,
        push_to="aperture.[lsf]",
    )
    coeff_pname = "lsf_coeffs_{}".format(i)
    ap_pspecs[coeff_pname] = ParameterSpec(
        factory=PickleParameter,
        push_to="aperture.[lsf_coeffs]",
        kwargs = {
            "value":lambda x: dirac_vec(lsf_degree+1, -1, chip_lsf_widths[i])
        },
    )
    
    lsf_mod_name = "lsf_poly_{}".format(i)
    ap_mspecs[lsf_mod_name] = ModelSpec(
        tmb.modeling.PixelPolynomialModel,
        inputs={"coeffs":coeff_pname},
        kwargs={"npts":chip_npts[i]},
        output=lsf_pname,
    )


aperture_mct = ModelComponentTemplate(
    parameter_specs=ap_pspecs,
    model_specs=ap_mspecs,
    application_spine=tmb.contexts.aperture_spine
)

def fetch_chip_lsf_coeffs(spec_context):
    spec = spec_context["spectrum"]
    ap = spec.aperture
    chip = spec.chip
    chip_name = chip.name
    chip_idx = int(chip_name[-1])
    coeff_p = ap["lsf_coeffs"][chip_idx]
    return coeff_p

def fetch_chip_lsf(spec_context):
    spec = spec_context["spectrum"]
    ap = spec.aperture
    chip = spec.chip
    chip_name = chip.name
    chip_idx = int(chip_name[-1])
    coeff_p = ap["lsf"][chip_idx]
    return coeff_p

sampling_mct = ModelComponentTemplate(
    {
        "model_wvs":ParameterSpec("global.model_wvs"),
        "model_lsf":ParameterSpec("global.model_lsf"),
        "spectrum_wvs":ParameterSpec("spectrum.rest_wvs"),
        "lsf_coeffs":ParameterSpec(
            fetch_chip_lsf_coeffs,
            push_to="spectrum.lsf_coeffs",
            factory_context_arg=True,
        ),
        "lsf":ParameterSpec(
            fetch_chip_lsf,
            push_to="spectrum.lsf",
            factory_context_arg=True,
        ),
        "sampling_matrix":ParameterSpec(Parameter, push_to="spectrum.sampling_matrix")
    },
    model_specs={
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
    },
    application_spine=tmb.contexts.spectrum_spine
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
    },
    application_spine = tmb.contexts.spectrum_spine
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
        "saturation_offset":ParameterSpec(
            factory=Parameter,
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
        "gamma_coeff_dict":ParameterSpec("global.gamma_coeff_dict"),
        "saturation_coeff_dict":ParameterSpec("global.saturation_coeff_dict"),
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
        "saturation_offset":ModelSpec(
            tmb.cog.SaturationOffsetModel,
            inputs={
                "teff":"teff",
                "coeff_dict":"saturation_coeff_dict",
            },
            output="saturation_offset",
        ),
        "saturations":ModelSpec(
            tmb.cog.SaturationModel,
            inputs={
                "offset":"saturation_offset",
                "pseudostrengths":"pseudostrengths",
                "saturation_curve":"cog",
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
    application_spine = tmb.contexts.star_spine
)


source_spec_linker_mct = ModelComponentTemplate(
    parameter_specs={
        "norm":ParameterSpec("spectrum.norm"),
        "sampling_matrix":ParameterSpec("spectrum.sampling_matrix"),
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
    },
    application_spine=tmb.contexts.source_spectra_pairs,
)


model_network = tmb.analysis.ModelNetworkTemplate(
    components=[
        model_wvs_mct, #set up the model wavelength grid
        transition_vectorizer_mct,
        gcog_mct,
        aperture_mct, #set up the lsf models and parameters
        sampling_mct, #set up the sampling matrix models
        normalization_mct, 
        star_mct,
        source_spec_linker_mct
    ]
)
