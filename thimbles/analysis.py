import os
from copy import copy
import re

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
            factory_context_arg=False,
    ):
        self.factory = factory
        if kwargs is None:
            kwargs = {}
        self.kwargs = kwargs
        self.push_to = push_to
        self.factory_context_arg=factory_context_arg
    
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
            compound_match = re.match("(.*)\[(\d*)\]$", ct_param_name)
            if not compound_match is None:
                ct_param_name, param_index = compound_match.groups()
                param_index = int(param_index)
                param = contexts[context_name][ct_param_name][param_index]
            else:
                param = contexts[context_name][ct_param_name]
        else:
            pkwargs = self.evaluate_kwargs(contexts)
            if self.factory_context_arg:
                arg = (contexts,)
            else:
                arg = tuple()
            param = self.factory(*arg, **pkwargs)
        return param
    
    def push(self, param, contexts):
        if not self.push_to is None:
            ctext_name, param_name = self.push_to.split(".")
            if ("[" == param_name[0])and ("]" == param_name[-1]):
                is_compound = True
                param_name = param_name[1:-1]
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
            application_spine,
    ):
        self.param_specs = parameter_specs
        self.model_specs = model_specs
        self.spine = application_spine
    
    def search_and_apply(
            self,
            db,
            context_engine=None,
            tag=None,
    ):
        if context_engine is None:
            context_engine = self.spine
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

class ModelNetworkTemplate(object):
    
    def __init__(
            self,
            components,
    ):
        self.components = components
    
    def initialize_network(
            self,
            db,
            context_engine=None,
            tag=None
    ):
        for mct in self.components:
            mct.search_and_apply(
                db=db,
                context_engine=context_engine,
                tag=tag
            )



def dirac_vec(n, i):
    vec = np.zeros(n, dtype=float)
    vec[i] = 1.0
    return vec


def extract_pseudonorm_coeffs(spec_context):
    spec = spec_context['spectrum']
    psnorm = spec.pseudonorm()
    npts = len(spec)
    xvals = (np.arange(npts) - 0.5*npts)/(0.5*npts)
    pseudonorm_coeffs = np.polyfit(xvals, psnorm)
    return pseudonorm_coeffs


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


def mct_to_gv(
        mct,
        graph_name,
        join=True,
):
    gv = ["digraph " + graph_name + "{"]
    param_nodes = sorted(mct.param_specs.keys())
    model_nodes = sorted(mct.model_specs.keys())
    
    param_to_idx = {param_nodes[i]:i for i in range(len(param_nodes))}
    
    #write the nodes out
    nodestr = '{node_id} [shape={shape} label="{label}"];'
    for p_idx in range(len(param_nodes)):
        p_name = param_nodes[p_idx]
        gv.append(nodestr.format(shape="oval", node_id="p_{}".format(p_idx), label=p_name))
    
    for m_idx in range(len(model_nodes)):
        m_name = model_nodes[m_idx]
        gv.append(nodestr.format(shape="box", node_id="m_{}".format(m_idx), label=m_name))
    
    for m_idx in range(len(model_nodes)):
        mod_name = model_nodes[m_idx]
        cmspec = mct.model_specs[mod_name]
        #write the input edges to models
        for param in cmspec.inputs.values():
            if isinstance(param, list):
                for compound_param in param:
                    p_idx = param_to_idx[compound_param]
                    gv.append("p_{} -> m_{};".format(p_idx, m_idx))
            else:
                p_idx = param_to_idx[param]
                gv.append("p_{} -> m_{};".format(p_idx, m_idx))
        #write output edge
        gv.append("m_{} -> p_{};".format(m_idx, param_to_idx[cmspec.output]))
    
    gv.append("}")
    
    if join:
        gv = "\n".join(gv)
    return gv


def model_net_template_to_gv(
        net_template,
):
    gv_list = []
    for i, component in enumerate(net_template.components):
        cname = "component_{}".format(i)
        gv_list.append(mct_to_gv(component, graph_name=cname))
    
    return "\n".join(gv_list)

