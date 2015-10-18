import numpy as np
import scipy.sparse

import os
import thimbles as tmb
from .options import opts
from .sqlaimports import *
from .thimblesdb import ThimblesTable, Base
from .modeling.associations import HasParameterContext
from .modeling.associations import NamedParameters, ParameterAliasMixin
from .modeling import FloatParameter, Parameter
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


class ModelComponentTemplate(object):
    
    def __init__(
            self,
            #what parameter class to use for each parameter node
            #{"parameter_name":ParameterClass} or {parameter_name":[ParameterClass]} in the case of compound parameters
            #e.g. {"coeffs":PickleParameter, "lsf":Parameter}
            parameter_classes,
            
            #extra arguments to pass to the parameter factory
            #{"parameter_name":{"kw":val}}
            #e.g. {"rv":{"value":32.1}}
            parameter_kwargs,
            
            #what model class to use for each model edge
            #{"model_edge_name":ModelClass}
            #e.g. {"lsf_poly":PixelPolynomialModel}
            model_classes,
            
            #how to connect model inputs to parameters
            #{"model_name":[["parameter_name", "model_input_name", is_compound], ...] 
            #e.g. {"lsf_poly":[["coeffs", "coeffs", False]]
            input_edges,
            
            #extra non-Parameter class arguments to pass models
            #{"model_name":{"kwarg_name": values}
            #e.g. {"model_wvs":{"npts":4096}}
            model_kwargs,
            
            #how to connect model outputs to parameters
            #{"model_name":"parameter_name"}
            #e.g. {"lsf_poly":"lsf"}
            output_edges, 
            
            #how to find pre-existing parameters in external contexts
            #{parameter_name:["context_name", "parameter_name_within_context"]}
            #e.g. {"lsf":["spectrum", "lsf"]}
            fetched_parameters,
            
            #which parameters to give aliases in external contexts and what names to give them
            #{"context_name":[["parameter_name", "external_name", is_compound]]}
            pushed_parameters,
    ):
        self.parameter_classes = parameter_classes
        self.parameter_kwargs = parameter_kwargs
        self.model_classes = model_classes
        self.input_edges = input_edges
        self.model_kwargs = model_kwargs
        self.output_edges = output_edges
        self.fetched_parameters = fetched_parameters
        self.pushed_parameters = pushed_parameters
    
    def apply_to_contexts(
            self,
            #a dictionary of named external contexts
            #{"context_name":context_instance}
            #e.g. {"spectrum": spec, "global":shared_params}
            contexts,
    ):
        import pdb; pdb.set_trace()
        instantiated_params = {}
        for param_name in self.parameter_classes:
            if param_name in self.fetched_parameters:
                context_name, cont_pname = self.fetched_parameters[param_name]
                fetched_p = contexts[context_name][cont_pname]
                instantiated_params[param_name] = fetched_p
            else:
                pkwargs = self.parameter_kwargs.get(param_name, {})
                for pk in pkwargs:
                    raw_arg = pkwargs[pk]
                    if hasattr(raw_arg, "__call__"):
                        pkwargs[pk] = raw_arg(contexts)
                
                pcls = self.parameter_classes[param_name]
                instantiated_params[param_name] = pcls(**pkwargs)
        
        for model_name in self.input_edges:
            model_kwargs = {}
            inp_edges = self.input_edges[model_name]
            for param_name, input_name, is_compound in inp_edges:
                inp_param = instantiated_params[param_name]
                if is_compound:
                    plist = model_kwargs.get(input_name, [])
                    if isinstance(inp_param, Parameter):
                        plist.append(inp_param)
                    else:
                        plist.extend(inp_param)
                    model_kwargs[input_name] = plist
                else:
                    model_kwargs[input_name] = inp_param
            extra_kwargs = self.model_kwargs.get(model_name, {})
            for kw in extra_kwargs:
                raw_kwarg = extra_kwargs[kw]
                if hasattr(raw_kwarg, "__call__"):
                    extra_kwargs[kw] = raw_kwarg(contexts)
            model_kwargs.update(extra_kwargs)
            mod_cls = self.model_classes[model_name]
            outp_name = self.output_edges[model_name]
            c_output_p = instantiated_params[outp_name]
            mod_instance = mod_cls(output_p=c_output_p, **model_kwargs)
        
        for context_name in self.pushed_parameters:
            target_context = contexts[context_name]
            alias_list = self.pushed_parameters[context_name]
            for param_name, alias, is_compound in alias_list:
                cparam = instantiated_params[param_name]
                target_context.add_parameter(
                    alias,
                    parameter=instantiated_params[param_name],
                    is_compound=is_compound)

class ModelingPattern(object):
    
    def __init__(
            self,
            #a list of model components and their unique context specification
            #[[context_identity_tuple, template_instance,]]
            model_templates,
            context_extractors,
    ):
        self.model_templates = model_templates
        self.context_extractors = context_extractors
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

