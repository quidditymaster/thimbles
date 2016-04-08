#Convert the hdf file into a contextually correct thimbles database.
from copy import copy
import matplotlib as mpl
mpl.use("qt4Agg")
import thimbles as tmb
import thimbles.contexts
from thimbles.modeling import Parameter, FloatParameter, PickleParameter
import pandas as pd
import tmb_model_specification as mspec

fpath = "iso_spectra.hdf"
db_path = "iso_spec.db"

gamma_coeff_dict = {
    "offset":-2.5,
    "ep":[5.085e-2, 6.5e-3],
    "logg":[1.3722e-1, 2.256e-2],
    "teff":[-1.21e-4],
}


#create the Chip objects
db = tmb.ThimblesDB(db_path)

#create the chip objects and populate with relevant contexts and info
chips = []
chip_names = "ccd0 ccd1 ccd2".split()

wvs = [pd.read_hdf(fpath, "wvs{}".format(chip_idx)).values for chip_idx in range(3)]
fluxes = [pd.read_hdf(fpath, "fluxes{}".format(chip_idx)).values for chip_idx in range(3)]
var = [pd.read_hdf(fpath, "var{}".format(chip_idx)).values for chip_idx in range(3)]


for chip_idx in range(3):
    chip = tmb.spectrographs.Chip(chip_names[chip_idx])
    wv_param = Parameter()
    tmb.coordinatization.LogLinearCoordinatizationModel(
        output_p=wv_param,
        min=FloatParameter(wvs[chip_idx][0]),
        max=FloatParameter(wvs[chip_idx][-1]),
        npts=len(wvs[chip_idx]),
    )
    chip.add_parameter("wvs", wv_param)
    chips.append(chip)


n_spec = len(fluxes[0])
source_objs = [tmb.star.Star("iso{}".format(i)) for i in range(n_spec)]

spectra = []
for chip_idx in range(3):
    for obj_idx in range(n_spec):
        spec = tmb.Spectrum(
            wvs[chip_idx],
            fluxes[chip_idx][obj_idx],
            var=var[chip_idx][obj_idx],
            chip=chips[chip_idx],
            source=source_objs[obj_idx],
        )
        spectra.append(spec)


db.add_all(spectra)

#create and populate the global parameters
global_params = tmb.analysis.SharedParameterSpace("global")
tmb.contexts.model_spines.add_global("global", global_params)
db.add(global_params)

#generate the model wavelength sampling solution
mspec.model_wvs_mct.apply(None, tmb.contexts.model_spines["global"])

#generate the sampling models per chip
mspec.chip_sampling_mct.search_and_apply(tmb.contexts.model_spines["chips"], db=db)

#generate the per spectrum normalization models
mspec.normalization_mct.search_and_apply(tmb.contexts.model_spines["spectra"], db=db)

#load the entire linelist and add to the database
transitions = tmb.io.linelist_io.read_linelist("/home/tim/linelists/apogee/moog.201306191000.vac.ln")
#transitions = tmb.io.linelist_io.read_linelist("/home/tim/linelists/apogee/transition_skeleton.ln")
n_trans_orig = len(transitions)
print("read in {} transitions".format(n_trans_orig))

min_wv = 15162
max_wv = 16932

transitions = [trans for trans in transitions if (trans.wv < max_wv) and (trans.wv > min_wv)]
print("discarded {} transitions outside of wavelength bounds".format(n_trans_orig-len(transitions)))

print("adding transitions to database")
db.add_all(transitions)
db.commit()

print("generating linelist for feature modeling")
feature_modeled_znums = set([3, 11, 12, 13, 14, 16, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29])

feature_modeled_transitions = [trans for trans in transitions if trans.ion.z in feature_modeled_znums]
#drop any transitions which are very weak
feature_modeled_transitions = [trans for trans in feature_modeled_transitions if trans.pseudo_strength() > -2.5]
feature_modeled_transitions = sorted(feature_modeled_transitions, key=lambda x: x.wv)
print("kept {} transitions total".format(len(feature_modeled_transitions)))

print("building a default value dictionary for per ion pseudostrength shifts")
ions = set()
for trans in feature_modeled_transitions:
    ions.add(trans.ion)
print("{} total species modeled".format(len(ions)))
default_ion_shifts = {ion:-1 for ion in ions}


print("making the global full transition indexer parameter")
transition_indexer_p = tmb.transitions.TransitionIndexerParameter(transitions=feature_modeled_transitions)
global_params.add_parameter("transition_indexer", transition_indexer_p)


print("generating a grouping standard ")
grouping_standard = tmb.transitions.segmented_grouping_standard(
    "s1", db,
    transitions=feature_modeled_transitions,
    wv_split=200.0,
    ep_split=1.0,
    loggf_split=2.5,
    x_split=1.0,
)

exemplar_map = {group.exemplar:group.transitions for group in grouping_standard.groups}

exemplars = sorted(exemplar_map.keys(), key=lambda x: (x.ion.z, x.ion.charge, x.wv))
print("writing out exemplar linelist for inspection")
tmb.io.write_moog_linelist("exemplar_transitions.ln", exemplars)

exemplar_indexer_p = tmb.transitions.TransitionIndexerParameter(transitions=exemplars)
global_params.add_parameter("exemplar_indexer", exemplar_indexer_p)

exemplar_map_p = tmb.transitions.ExemplarGroupingParameter(groups=exemplar_map)
global_params.add_parameter("exemplar_map", exemplar_map_p)

print("generating global transition models")

transition_wvs_p = Parameter()
tmb.features.TransitionWavelengthVectorModel(output_p=transition_wvs_p, indexer=transition_indexer_p)
global_params.add_parameter("transition_wvs", transition_wvs_p)

transition_mol_weights_p = Parameter()
tmb.features.IonWeightVectorModel(
    output_p=transition_mol_weights_p,
    indexer=transition_indexer_p
)
global_params.add_parameter("transition_molecular_weights", transition_mol_weights_p)

transition_ep_p = Parameter()
tmb.features.TransitionEPVectorModel(
    output_p=transition_ep_p,
    indexer=transition_indexer_p
)
global_params.add_parameter("transition_ep", transition_ep_p)

gamma_coeff_p = tmb.modeling.PickleParameter(gamma_coeff_dict)
global_params.add_parameter("gamma_coeff_dict", gamma_coeff_p)

print("generating per star models")
star_context = tmb.contexts.model_spines["stars"]
mspec.star_mct.search_and_apply(star_context, db=db)
print("post populating quick guess parameters")
stars = star_context.find(db=db)
for star in stars:
    star["ion_shifts"].set(copy(default_ion_shifts))

print("linking star models to data models")
#link the model feature parameter to the data
mspec.source_spec_linker_mct.search_and_apply(tmb.contexts.model_spines["source spectrum pairs"], db=db)

db.commit()
