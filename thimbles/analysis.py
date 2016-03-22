import os
from copy import copy

import numpy as np
import scipy.sparse

import thimbles as tmb
from .options import opts
from .sqlaimports import *
from .thimblesdb import ThimblesTable, Base
from .modeling.associations import HasParameterContext
from .modeling.associations import NamedParameters, ParameterAliasMixin
from .modeling import Parameter, FloatParameter, PickleParameter
from .tasks import task

class SharedParametersAlias(ParameterAliasMixin, ThimblesTable, Base):
    _context_id = Column(Integer, ForeignKey("SharedParameterSpace._id"))
    context= relationship("SharedParameterSpace", foreign_keys=_context_id, back_populates="context")

class SharedParameterSpace(ThimblesTable, Base, HasParameterContext):
    name = Column(String, unique=True)
    context = relationship("SharedParametersAlias", collection_class=NamedParameters)
    
    def __init__(self, name):
        HasParameterContext.__init__(self)
        self.name = name

    def add_parameter(self, name, parameter, is_compound=False):
        SharedParametersAlias(name=name, context=self, parameter=parameter, is_compound=is_compound)


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
    
    #cog model
    cog_p = tmb.modeling.Parameter()
    cog_model = tmb.cog.SaturationCurveModel(
        output_p = cog_p,
        gamma_p=vec_gamma_p,
        sigma_p=sigma_vec_p,
    )
    star.add_parameter("cog", cog_p)
    
    #grouping matrix model
    grouping_matrix_p = tmb.modeling.Parameter()
    grouping_matrix_mod = tmb.features.RelativeStrengthMatrixModel(
        output_p = grouping_matrix_p,
        grouping_p = grouping_p,
        transition_indexer_p = transition_indexer_p,
        exemplar_indexer_p = exemplar_indexer_p,
        pseudostrength_p=pseudostrength_p,
        cog_p=cog_p,
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
        degree=6,
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


def _load_db_ob(fname, cls, call_signature, converters=None, key_func=lambda x: x.name):
    if converters is None:
        converters = {}
    obs = {}
    with open(fname) as f:
        for line in f:
            lstr = line.strip()
            if len(lstr) > 0:
                if lstr[0]=="#":
                    continue
                lspl = lstr.split()
                kwargs = dict(list(zip(call_signature, lspl)))
                for conv_key in converters:
                    kwargs[conv_key] = converters[conv_key](kwargs[conv_key])
                new_instance = cls(**kwargs)
                obs[key_func(new_instance)] = new_instance
    return obs

def load_sources(fname):
    return _load_db_ob(
        fname, 
        cls=tmb.sources.Source, 
        call_signature = "name ra dec".split(),
        converters=dict(ra=float, dec=float),
    )


def load_exposures(fname):
    return _load_db_ob(
        fname, 
        cls=tmb.observations.Exposure, 
        call_signature = "name time duration".split(),
        converters=dict(time=float, duration=float),
    )

def load_chips(fname):
    return _load_db_ob(
        fname, 
        cls=tmb.spectrographs.Chip, 
        call_signature = "name".split(),
        converters={}
    )

def load_slits(fname):
    return _load_db_ob(
        fname, 
        cls=tmb.spectrographs.Slit, 
        call_signature = "name".split(),
        converters=dict(),
    )

def load_orders(fname):
    return _load_db_ob(
        fname, 
        cls=tmb.spectrographs.Order, 
        call_signature = "number".split(),
        converters=dict(number=int),
        key_func=lambda x: x.number
    )


def component_template_gv(mct):
    parameter_classes = mct.parameter_classes
    fetched_params = mct.fetched_parameters
    pushed_params = mct.pushed_parameters
    model_classes = mct.model_classes
    minp_edges = mct.input_edges
    mout_edges = mct.output_edges
    
    gv = ["digraph graphname{"]
    
    for param_name in parameter_classes:
        node_id = "param.{}".format(param_name)
        style="solid"
        if param_name in fetched_params:
            style="dashed"
        gv.append("param_{param_name} [shape=oval, style={style}, label={param_name}];".format(style=style, param_name=param_name))
    
    context_node_str = "context_{context_name} [shape=polygon, sides=7, label={context_name}];"
    for context_name in pushed_params.keys():
        gv.append(context_node_str.format(context_name=context_name))
    
    for param_name in fetched_params:
        context_name, pnwc = fetched_params[param_name]
        gv.append(context_node_str.format(context_name=context_name))
    
    for model_name in model_classes:
        gv.append("model_{model_name} [shape=box, label={model_name}];".format(model_name=model_name))
    
    for model_name in minp_edges:
        for input_edge in minp_edges[model_name]:
            param_name, model_port, compoundness = input_edge
            if compoundness:
                compound_indicator = "[]"
            else:
                compound_indicator = ""
            gv.append('param_{param_name} -> model_{model_name} [label="{model_port}{compound_indicator}"];'.format(param_name=param_name, model_name=model_name, model_port=model_port, compound_indicator=compound_indicator))
    
    for model_name in mout_edges:
        param_name = mout_edges[model_name]
        gv.append("model_{model_name} -> param_{param_name};".format(param_name=param_name, model_name=model_name))
    
    for param_name in fetched_params:
        context_name, in_context_pname = fetched_params[param_name]
        gv.append("context_{context_name} -> param_{param_name} [label={in_context_pname}];".format(context_name=context_name, param_name=param_name, in_context_pname=in_context_pname))
    
    for context_name in pushed_params:
        for context_edge in pushed_params[context_name]:
            param_name, in_context_pname, is_compound = context_edge
            gv.append("param_{param_name} -> context_{context_name} [label={in_context_pname}];".format(context_name=context_name, param_name=param_name, in_context_pname=in_context_pname))
    
    gv.append("}")
    return "\n".join(gv)

class ParameterSpec(object):
    
    def __init__(
            self,
            factory,
            kwargs=None,
            push_to=None,
    ):
        self.factory = factory
        if kwargs is None:
            kwargs = {}
        self.kwargs = kwargs
        self.push_to = push_to
    
    @property
    def is_fetched(self):
        return isinstance(self.factory, str)
    
    def evaluate_kwargs(self, contexts):
        pkwargs = copy(self.kwargs)
        for pk in pkwargs:
            raw_arg = pkwargs[pk]
            if hasattr(raw_arg, "__call__"):
                pkwargs[pk] = raw_arg(contexts)
        return pkwargs
    
    def fetch(self, contexts):
        if self.is_fetched:
            context_name, ct_param_name = self.factory.split(".")
            param = contexts[context_name][ct_param_name]
        else:
            pkwargs = self.evaluate_kwargs(contexts)
            param = self.factory(**pkwargs)
        return param
    
    def push(self, param, contexts):
        if not self.push_to is None:
            ctext_name, param_name = self.push_to.split(".")
            if "[" == param_name[0]:
                is_compound=True
            else:
                is_compound=False
            ctext = contexts[ctext_name]
            ctext.add_parameter(param_name, param, is_compound=is_compound)


class ModelSpec(object):
    
    def __init__(
            self,
            factory,
            inputs,
            output,
            kwargs=None,
    ):
        self.factory=factory
        self.inputs = inputs
        self.output = output
        if kwargs is None:
            kwargs = {}
        self.kwargs = kwargs
    
    def evaluate_kwargs(self, contexts):
        pkwargs = copy(self.kwargs)
        for pk in pkwargs:
            raw_arg = pkwargs[pk]
            if hasattr(raw_arg, "__call__"):
                pkwargs[pk] = raw_arg(contexts)
        return pkwargs
    
    def link(self, parameters, contexts):
        pkwargs = self.evaluate_kwargs(contexts)
        inputs = self.inputs
        for input_name in inputs:
            param_name = inputs[input_name]
            if isinstance(param_name, list):
                plist = []
                for pname in param_name:
                    plist.append(parameters[pname])
                pkwargs[input_name] = plist
            else:
                pkwargs[input_name] = parameters[param_name]
        
        output_p = parameters[self.output]
        return self.factory(output_p=output_p, **pkwargs)


class ModelComponentTemplate(object):
    
    def __init__(
            self,
            parameter_specs,
            model_specs,
    ):
        self.param_specs = parameter_specs
        self.model_specs = model_specs
    
    def search_and_apply(
            self,
            context_engine,
            tag=None,
            db=None,
    ):
        modeling_spine = context_engine.find(tag=tag, db=db)
        for backbone_instance in modeling_spine:
            self.apply(backbone_instance, context_engine)
    
    def apply(
            self,
            backbone_instance,
            context_engine,
    ):
        #a dictionary of named external contexts
        #{"context_name":context_instance}
        #e.g. {"spectrum": spec, "global":shared_params}
        contexts = context_engine.contextualize(backbone_instance)
        
        instantiated_params = {}
        for param_name in self.param_specs:
            pspec = self.param_specs[param_name]
            param = pspec.fetch(contexts)
            instantiated_params[param_name] = param
            pspec.push(param, contexts)
        
        ###
        for model_name in self.model_specs:
           mspec = self.model_specs[model_name]
           mspec.link(instantiated_params, contexts)

######

class ComponentRegistry(object):
    
    def __init__(self):
        self.spines = {}
    
    def register_template(
            self,
            spine_name,
            template_name,
            template_instance,
    ):
        copts = self.spines.get(spine_name, {})
        copts[template_name] = template_instance
        self.spines[spine_name] = copts

component_templates = ComponentRegistry()

class ModelNetworkRecipe(object):
    
    def __init__(
            self,
            #a list of model components and their unique context specification
            #[[context_identity_tuple, template_instance,]]
            model_templates,
            context_extractor,
    ):
        self.model_templates = model_templates
        self.extractor = context_extractors
        self.incorporated_context_tuples = set()
    
    def get_data_context(self, data_instance):
        contexts = {}
        for cont_name in self.context_extractors:
            cext = self.context_extractors[cont_name]
            contexts[cont_name] = cext(data_instance)
        return contexts
    
    def incorporate_datum(
            self,
            data_instance,
    ):
        data_contexts = self.get_data_context(data_instance)
        for context_id_tuple, mct in self.model_templates:
            cur_context_tup = tuple([data_contexts[cname] for cname in context_id_tuple])
            if not (cur_context_tup in self.incorporated_context_tuples):
                mct.apply_to_contexts(data_contexts)
                self.incorporated_context_tuples.add(cur_context_tup)
    
    def incorporate_data(
            self,
            data_list,
    ):
        for datum in data_list:
            self.incorporate_datum(datum)
        return

class JModelComponentTemplate(object):

    def __init__(self, *args, **kwargs):
        pass

global_mct = JModelComponentTemplate(
    parameter_classes = {
        "min_wv":FloatParameter,
        "max_wv":FloatParameter,
        "model_wvs":Parameter,
        "molecular_weights":Parameter,
        "model_lsf":FloatParameter,
        "transition_indexer":tmb.transitions.TransitionIndexerParameter,
        "measurement_indexer":tmb.transitions.TransitionIndexerParameter,
        "primary_indexer":tmb.transitions.TransitionIndexerParameter,
        "transition_wvs":Parameter,
        "measurement_grouping":tmb.transitions.ExemplarGroupingParameter,
        "primary_grouping":tmb.transitions.ExemplarGroupingParameter,
    },
    parameter_kwargs = {
        "min_wv":{"value":lambda x: tmb.opts["modeling.min_wv"]},
        "max_wv":{"value":lambda x: tmb.opts["modeling.max_wv"]},
        "model_lsf":{"value":0.0},
    },
    model_classes={
        "wavelength_solution":tmb.coordinatization.LogLinearCoordinatizationModel,
        "wv_vectorizer":tmb.features.TransitionWavelengthVectorModel,
        "molecular_weight_vectorizer":tmb.features.IonWeightVectorModel,
    },
    input_edges = {
        "wavelength_solution":[
            ["min_wv", "min", False],
            ["max_wv", "max", False],
        ],
        "wv_vectorizer":[
            ["transition_indexer", "indexer", False],
        ],
        "molecular_weight_vectorizer":[
            ["transition_indexer", "indexer", False]
        ],
    },
    model_kwargs = {
        "wavelength_solution":{
            "npts":lambda x: int(np.log(opts["modeling.max_wv"]/opts["modeling.min_wv"])*opts["modeling.resolution"])+1
        }
    },
    output_edges = {
        "wavelength_solution":"model_wvs",
        "wv_vectorizer":"transition_wvs",
        "molecular_weight_vectorizer":"molecular_weights"
    },
    fetched_parameters={},
    pushed_parameters={
        "global":[
            ["min_wv", "min_wv", False],
            ["max_wv", "max_wv", False],
            ["model_wvs", "model_wvs", False],
            ["model_lsf", "model_lsf", False],
            ["transition_wvs", "transition_wvs", False],
            ["molecular_weights", "molecular_weights", False],
            ["transition_indexer", "transition_indexer", False],
            ["measurement_indexer", "measurement_indexer", False],
            ["primary_indexer", "primary_indexer", False],
        ],
    }
)
#component_templates.register_template("global", "ew_base", global_mct)

def dirac_vec(n, i):
    vec = np.zeros(n, dtype=float)
    vec[i] = 1.0
    return vec

default_sampling_mct = JModelComponentTemplate(
    parameter_classes = {
        "lsf":None, 
        "lsf_coeffs":PickleParameter, 
        "sampling_matrix":Parameter, 
        "model_wvs":None,
        "model_lsf":None,
        "spectrum_wvs":None
    },
    parameter_kwargs={
        "lsf_coeffs":{"value":lambda x: dirac_vec(4, -1)}
    },
    model_classes={
        "lsf_poly":tmb.modeling.PixelPolynomialModel, 
        "sampling_matrix":tmb.spectrographs.SamplingModel,
    },
    input_edges = {
        "lsf_poly":[["lsf_coeffs", "coeffs", False]],
        "sampling_matrix":[
            ["model_wvs", "input_wvs", False],
            ["model_lsf", "input_lsf", False],
            ["spectrum_wvs", "output_wvs", False],
            ["lsf", "output_lsf", False],
        ]
    },
    model_kwargs={
        "lsf_poly":{
            "npts":lambda x: len(x["spectrum"].flux)
        },
    },
    output_edges = {
        "lsf_poly":"lsf",
        "sampling_matrix":"sampling_matrix",
    },
    fetched_parameters = {
        "lsf":["spectrum", "lsf"],
        "spectrum_wvs":["spectrum", "rest_wvs"],
        "model_wvs":["global", "model_wvs"],
        "model_lsf":["global", "model_lsf"],
    },
    pushed_parameters = {
        "spectrum":[
            ["lsf_coeffs", "lsf_coeffs", False],
            ["sampling_matrix", "sampling_matrix", False],
        ],
    },
)
#component_templates.register_template(
#    "sampling",
#    "default",
#    default_sampling_mct
#)

def extract_pseudonorm_coeffs(spec_context):
    spec = spec_context['spectrum']
    psnorm = spec.pseudonorm()
    npts = len(spec)
    xvals = (np.arange(npts) - 0.5*npts)/(0.5*npts)
    pseudonorm_coeffs = np.polyfit(xvals, psnorm)
    return pseudonorm_coeffs


default_normalization_mct = JModelComponentTemplate(
    parameter_classes = {
        "norm":Parameter, 
        "norm_coeffs":PickleParameter, 
    },
    parameter_kwargs={
        "norm_coeffs":{"value":extract_pseudonorm_coeffs}
    },
    model_classes={
        "norm_poly":tmb.modeling.PixelPolynomialModel, 
    },
    input_edges = {
        "norm_poly":[["lsf_coeffs", "coeffs", False]],
    },
    model_kwargs={
        "norm_poly":{
            "npts":lambda x: len(x["spectrum"])
        },
    },
    output_edges = {
        "norm_poly":"norm",
    },
    fetched_parameters = {
    },
    pushed_parameters = {
        "spectrum":[
            ["norm", "norm", False],
            #["norm_coeffs", "norm_coeffs", False],
        ],
    },
)
#component_templates.register_template("normalization", "default", default_normalization_mct)


fluxed_source_mct = JModelComponentTemplate(
    parameter_classes = {
        "source_features":None,
        "continuum_shape":None,
        "central_flux":Parameter,
        "broadening_matrix":None,
        "source_flux":Parameter,
    },
    parameter_kwargs = {},
    model_classes={
        "flux_multiplier":tmb.modeling.MultiplierModel,
        "broadener":tmb.modeling.MatrixMultiplierModel
    },
    input_edges={
        "flux_multiplier":[
            ["features", "factors", True],
            ["continuum_shape", "factors", True],
        ],
        "broadener":[
            ["central_flux", "vector", False],
            ["broadening_matrix", "matrix", True],
        ],
    },
    model_kwargs={},
    output_edges={
        "flux_multiplier":"central_flux",
        "broadener":"source_flux",
    },
    fetched_parameters = {
        "continuum_shape":["source", "continuum_shape"],
        "source_features":["source", "source_features"],
        "broadengin_matrix":["source", "broadening_matrix"],
    },
    pushed_parameters = {
        "source":[
            ["central_flux", "central_flux", False],
            ["source_flux", "source_flux", True],
        ],
    }
)
#component_templates.register_template("source", "fluxed source", fluxed_source_mct)


bare_source_mct = JModelComponentTemplate(
    parameter_classes = {
        "source_flux":Parameter,
    },
    parameter_kwargs = {},
    model_classes={
        "flux_multiplier":tmb.modeling.MultiplierModel,
        "broadener":tmb.modeling.MatrixMultiplierModel
    },
    input_edges={
        "flux_multiplier":[
            ["features", "factors", True],
            ["continuum_shape", "factors", True],
        ],
        "broadener":[
            ["central_flux", "vector", False],
            ["broadening_matrix", "matrix", True],
        ],
    },
    model_kwargs={},
    output_edges={
        "flux_multiplier":"central_flux",
        "broadener":"source_flux",
    },
    fetched_parameters = {
        "continuum_shape":["source", "continuum_shape"],
        "source_features":["source", "source_features"],
        "broadengin_matrix":["source", "broadening_matrix"],
    },
    pushed_parameters = {
        "source":[
            ["source_flux", "source_flux", True],
        ],
    }
)
#component_templates.register_template("source", "source flux", fluxed_source_mct)


star_mct = JModelComponentTemplate(
    parameter_classes = {
        "central_features":Parameter,
        "features":Parameter,
        "broadening_matrix":Parameter,
        "sigma_vec":Parameter,
        "gamma_vec":Parameter,
        "pseudostrength":Parameter,
        "reference_gamma":FloatParameter,
        "relative_ion_density":tmb.abundances.IonMappedParameter,
        "cog":Parameter,
        "teff":None,
        "vmicro":None,
        "vsini":None,
        "vmacro":None,
        "ldark":None,
        "model_wvs":None,
        "transition_wvs":None,
        "molecular_weights":None,
        "transition_indexer":None,
        "measurement_indexer":None,
        "measurement_grouping":None,
        "primary_indexer":None,
        "feature_matrix":Parameter,
        "measurement_rsm":Parameter,
        "primary_rsm":Parameter,
        "condensed_feature_matrix":Parameter,
        "saturation_offset":FloatParameter,
        "saturation":Parameter,
        "measured_ews":tmb.transitions.TransitionMappedParameter,
        "measured_ew_vec":Parameter,
    },
    parameter_kwargs={
        "reference_gamma":{"value":0.02},
        "saturation_offset":{"value":-1.0},
    },
    model_classes={
        "broadening_matrix_model":tmb.rotation.BroadeningMatrixModel,
        "broadener":tmb.modeling.MatrixMultiplierModel,
        "sigma_model":tmb.features.SigmaModel,
        "gamma_model":tmb.features.GammaModel,
        "pseudostrength_model":tmb.cog.PseudoStrengthModel,
        "cog_model":tmb.cog.SaturationCurveModel,
        "saturation_model":tmb.cog.SaturationModel,
        "feature_matrix_model":tmb.features.ProfileMatrixModel,
        #"primary_grouping_matrix_model":tmb.features.RelativeStrengthMatrixModel,
        "measurement_rsm_model":tmb.features.RelativeStrengthMatrixModel,
        "feature_condenser":tmb.features.CollapsedFeatureMatrixModel,
        "feature_model":tmb.modeling.MatrixMultiplierModel,
        "ew_vectorizer":tmb.transitions.TransitionMappedVectorizerModel,
    },
    input_edges = {
        "sigma_model":[
            ["teff", "teff", False],
            ["vmicro", "vmicro", False],
            ["transition_wvs", "transition_wvs", False],
            ["molecular_weights", "molecular_weights", False],
        ],
        "gamma_model":[
            ["reference_gamma", "gamma", False],
            ["transition_wvs", "transition_wvs", False],
        ],
        "pseudostrength_model":[
            ["transition_indexer", "transition_indexer", False,],
            ["relative_ion_density", "ion_correction", False,],
            ["teff", "teff", False],
        ],
        "saturation_model":[
            ["pseudostrength", "pseudostrength", False,],
            ["saturation_offset", "offset", False],
        ],
        "feature_matrix_model":[
            ["model_wvs", "model_wvs", False],
            ["transition_wvs", "centers", False],
            ["sigma_vec", "sigma", False],
            ["gamma_vec", "gamma", False],
            ["saturation", "saturation", False],
        ],
        "measurement_rsm_model":[
            ["measurement_grouping", "measurement_grouping", False],
            ["transition_indexer", "transition_indexer", False],
            ["measurement_indexer","exemplar_indexer", False],
            ["pseudostrength", "pseudostrength", False],
            ["cog", "cog", False],
        ],
        "cog_model":[
            ["gamma_vec", "gamma", False],
            ["sigma_vec", "sigma", False],
        ],
        "broadening_matrix_model":[
            ["model_wvs", "model_wvs", False],
            ["vsini", "vsini", False],
            ["vmacro", "vmacro", False],
            ["ldark", "ldark", False],
        ],
        "broadener":[
            ["broadening_matrix", "matrix", False],
            ["central_features", "vector", False],
        ],
        "feature_condenser":[
            ["feature_matrix", "feature_matrix", False],
            ["measurement_rsm", "grouping_matrix", False],
        ],
        "feature_model":[
            ["condensed_feature_matrix", "matrix", False],
            ["measured_ew_vec", "vector", False],
        ],
        "ew_vectorizer":[
            ["measured_ews", "mapped_values", False],
            ["measurement_indexer", "indexer", False],
        ],
    },
    model_kwargs={},
    output_edges = {
        "sigma_model":"sigma_vec",
        "gamma_model":"gamma_vec",
        "pseudostrength_model":"pseudostrength",
        "saturation_model":"saturation",
        "broadener":"features",
        "broadening_matrix_model":"broadening_matrix",
        "cog_model":"cog",
        "measurement_rsm_model":"measurement_rsm",
        "primary_rsm_model":"primary_rsm",
        "feature_condenser":"condensed_feature_matrix",
        "feature_model":"central_features",
        "feature_matrix_model":"feature_matrix",
        "ew_vectorizer":"measured_ew_vec",
    },
    fetched_parameters = {
        "teff":["star", "teff"],
        "vmicro":["star", "vmicro"],
        "vmacro":["star", "vmacro"],
        "vsini":["star", "vsini"],
        "ldark":["star", "ldark"],
        "measurement_indexer":["global", "measurement_indexer"],
        "measurement_grouping":["global", "measurement_grouping"],
        "primary_grouping":["global", "primary_grouping"],
        "transition_indexer": ["global", "transition_indexer"],
        "primary_indexer":["global", "primary_indexer"],
        "transition_wvs":["global", "transition_wvs"],
        #"primary_associator": ["global", "primary_map"],
        #"measurement_associator": ["global"
        "molecular_weights":["global", "full_molecular_weights"],
        "model_wvs":["global", "model_wvs"],
    },
    pushed_parameters = {
        "star":[
            #["features", "features", False],
            #["full_ews", "full_ews", False],
            #["measured_ews", "measured_ews", False],
            #["sigma_vec", "sigma_vec", False],
            #["gamma_vec", "gamma_vec", False],
            #["relative_ion_density", "relative_ion_density", False],
            #["broadening_matrix", "broadening_matrix", False],
            #["feature_matrix", "feature_matrix", False],
            #["pseudostrengths", "pseudostrengths", False],
        ],
    },
)
#component_templates.register_template("stars", "default", star_mct)


plain_ew_mct = JModelComponentTemplate(
    parameter_classes = {
        "synthetic_ews":tmb.transitions.TransitionMappedParameter,
        "measurement_ews":Parameter,
        "spectrum_ews":Parameter,
        "lsf_coeffs":PickleParameter, 
        "sampling_matrix":Parameter, 
        "model_wvs":None,
        "model_lsf":None,
        "spectrum_wvs":None
    },
    parameter_kwargs={
        "lsf_coeffs":{"value":lambda x: dirac_vec(4, -1)}
    },
    model_classes={
        "lsf_poly":tmb.modeling.PixelPolynomialModel, 
        "sampling_matrix":tmb.spectrographs.SamplingModel,
    },
    input_edges = {
        "lsf_poly":[["lsf_coeffs", "coeffs", False]],
        "sampling_matrix":[
            ["model_wvs", "input_wvs", False],
            ["model_lsf", "input_lsf", False],
            ["spectrum_wvs", "output_wvs", False],
            ["lsf", "output_lsf", False],
        ]
    },
    model_kwargs={
        "lsf_poly":{
            "npts":lambda x: len(x["spectrum"].flux)
        },
    },
    output_edges = {
        "lsf_poly":"lsf",
        "sampling_matrix":"sampling_matrix",
    },
    fetched_parameters = {
        "lsf":["spectrum", "lsf"],
        "spectrum_wvs":["spectrum", "rest_wvs"],
        "model_wvs":["global", "model_wvs"],
        "model_lsf":["global", "model_lsf"],
    },
    pushed_parameters = {
        "spectrum":[
            ["lsf_coeffs", "lsf_coeffs", False],
            ["sampling_matrix", "sampling_matrix", False],
        ],
    },
)


standard_config_head = \
"""
import thimbles as tmb
opts = tmb.opts
from thimbles import workingdataspace as wds

#moog stuff
opts["moog.working_dir"] = "{moogwdir}"
opts["moog.executable"] = "{moogsilent}"

#instantiate our probject database
db = tmb.ThimblesDB("{project_db_path}")
#put a reference to our project db into the working data space
wds.db = db
#make our default database our project db
opts["default_db"] = db

#spectra options
opts["spectra.io.read_default"] = tmb.io.read_spec
opts["spectra.io.write_default"] = tmb.io.write_spec

opts["wavelengths.medium"] = "vaccuum"
"""

@task()
def newproject(projname, flavor="standard", create_moog_wdir=False):
    cur_dir = os.getcwd()
    moogwdir = os.path.join(cur_dir, "moogdir")
    if create_moog_wdir:
        os.system("mkdir {}".format(moogwdir))
    
    project_db_path = os.path.join(cur_dir, "{}.db".format(projname))
    config_str = standard_config_head.format(
        moogwdir=moogwdir,
        moogsilent=tmb.opts["moog.executable"],
        project_db_path=project_db_path,
    )
    if os.path.isfile("tmb_config.py"):
        print("pre-existing tmb-config.py found. quitting")
    else:
        print("writing configuration into local file tmb_config.py")
        with open("tmb_config.py", "w") as f:
            f.write(config_str)

