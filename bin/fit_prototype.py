import thimbles as tmb
import time
import h5py
from thimbles.modeling import modeling
from thimbles.hydrogen import get_H_mask
import numpy as np
import pandas as pd
import scipy.sparse
import matplotlib.pyplot as plt
from thimbles.velocity import template_rv_estimate
import argparse
import cPickle
import multiprocessing
from copy import copy
import json

parser = argparse.ArgumentParser("fit prototype")
#parser.add_argument("fname")
parser.add_argument("--ll", default="/home/tim/linelists/vald/5250_3.0_0.01.vald")
parser.add_argument("--output-h5", default="ew_out.h5")
parser.add_argument("--input-h5")
parser.add_argument("--delta-wv", type=float, default=50.0)
parser.add_argument("--delta-x", type=float, default=0.3)
parser.add_argument("--lltype", default="vald")
parser.add_argument("--snr-threshold", type=float, default=5.0)
parser.add_argument("--snr-target", type=float, default=100.0)
parser.add_argument("--x-offset", type=float, default=7.0)
parser.add_argument("--rv-file", default="vrad.txt")
parser.add_argument("--teff", type=float, default=5000.0)
parser.add_argument("--vmicro", type=float, default=2.0)
parser.add_argument("--H_mask_radius", type=float, default=5.0)
parser.add_argument("--model-resolution", type=float, default=1e5)
parser.add_argument("--lsf-pkl")

def get_max_resolution(spectra):
    reses = [np.median(scipy.gradient(spec.wv)/spec.wv) for spec in spectra]
    return np.max(reses)
    
def get_data_transforms(spectra, model_wv):
    print "generating data transforms"
    transforms = []
    for spec in spectra:
        t = spec.lsf_sampling_matrix(model_wv)
        transforms.append(t)
    return transforms

def fit_blackbody_phirls(spectrum, gamma_frac=3.0):
    flux = spectrum.flux
    variance = spectrum.var
    wvs = spectrum.wv
    def get_resids(pvec):
        multiplier, teff = pvec
        model_flux = multiplier*blackbody_spectrum(wvs, teff, normalize=True)
        resid = flux - model_flux
        return tmb.utils.misc.pseudo_residual(resid, variance, gamma_frac=0.5)

class ConstantMultiplierModel(object):
    
    def __init__(self, mult):
        self.mult = mult
        self._lin_op = scipy.sparse.dia_matrix((self.mult, 0), shape=(len(self.mult), len(self.mult)))    
    
    def __call__(self, input, **kwargs):
        return self.mult*input
    
    def as_linear_op(self, input, **kwargs):
        return self._lin_op

def freeze_params(fstate, teff=True, gamma_ratio_5000=True, vmicro=True):
    if teff:
        fstate.model_network.feature_mod.theta_p._free = False
    if gamma_ratio_5000:
        fstate.model_network.feature_mod.gamma_ratio_5000_p._free = False
    if vmicro:
        fstate.model_network.feature_mod.vmicro_p._free = False

def thaw_params(fstate, teff=True, gamma_ratio_5000=True, vmicro=True):
    if teff:
        fstate.model_network.feature_mod.theta_p._free = True
    if gamma_ratio_5000:
        fstate.model_network.feature_mod.gamma_ratio_5000_p._free = True
    if vmicro:
        fstate.model_network.feature_mod.vmicro_p._free = True

def iter_cleanup(fstate):
    print "in fstate {}".format(fstate)
    mnet = fstate.model_network
    spec_idx = mnet.spectrum_id
    teff = mnet.feature_mod.teff
    vmicro = mnet.feature_mod.vmicro
    gamma_rat = mnet.feature_mod.gamma_ratio_5000
    print "teff {:5.3f}  vmicro {:5.3f} gamma_ratio {:5.4f} ".format(teff, vmicro, gamma_rat)
    
    spec = mnet.target_spectra[0]
    if True:
        fig, axes = plt.subplots(2)
        lwv = np.log10(spec.wv)
        axes[0].cla()
        model_values = mnet.trees[0]()
        axes[0].plot(lwv, spec.flux, lwv, model_values)
        axes[1].cla()
        resid = model_values-spec.flux
        axes[1].plot(lwv, np.sign(resid)*np.sqrt(resid**2*spec.inv_var))
        axes[1].set_ylim(-12.0, 12.0)
        fig.savefig("figures/spec_{}_cfit.png".format(spec_idx))

def write_results(fstate):
    mnet = fstate.model_network
    spec_idx = mnet.spectrum_id
    mnet.feature_mod.fdat.to_hdf("ew_files/spec_{}_ew.h5".format(spec_idx), "fdat")
    #cPickle.dump(smod, open("model_files/spec_{}_mod.pkl", "w"))
    f = open("model_files/spec_{}_params.txt".format(spec_idx), "w")
    fmod = mnet.feature_mod
    out_data_dict = dict(teff=fmod.teff, mean_gamma_ratio=fmod.gamma_ratio_5000, vmicro=fmod.vmicro, log_likelihood=fstate.log_likelihood)
    json.dump(out_data_dict, f)
    f.flush()
    f.close()

class GridSearch(modeling.FitState):
    
    def __init__(self):
        pass
    
    def iterate(self):
        return True

class SpectralModeler(modeling.DataModelNetwork):
    
    def __init__(self, target_spectra, lsf_models, ldat, species_grouper, max_iter, spectrum_id, start_teff):
        self.spectrum_id = spectrum_id
        self.hmasks = [get_H_mask(tspec.wv, args.H_mask_radius) for tspec in target_spectra]
        self.target_spectra = target_spectra
        for tspec_idx, tspec in enumerate(self.target_spectra):
            tspec.inv_var = np.where(self.hmasks[tspec_idx], tspec.inv_var, 0.0001)
        
        min_wv = np.min([np.min(spec.wv) for spec in target_spectra])
        max_wv = np.max([np.max(spec.wv) for spec in target_spectra])
        
        self.feature_mod = tmb.features.SaturatedVoigtFeatureModel(
            ldat, 
            min_wv, 
            max_wv,
            max_delta_wv=args.delta_wv, 
            max_delta_x=args.delta_x, 
            snr_threshold=args.snr_threshold,
            snr_target=args.snr_target,
            initial_x_offset=args.x_offset, 
            vmicro=args.vmicro,
            gamma_ratio_5000=0.3,
            teff=start_teff,#args.teff,
            model_resolution=args.model_resolution,
            species_grouper=species_grouper
        )
        
        
        print "generating model components"
        model_wv = self.feature_mod.model_wv
        self.exp_mod = tmb.features.OpacityToTransmission()
        #print "ctm model"
        #self.ctm_mod = tmb.continuum.BlackBodyContinuumModel(model_wv, args.teff)
        #print "hmod"
        #self.hmod = tmb.hydrogen.HydrogenForegroundModel(model_wv, np.array([2.5, 0.1]), 5300, 5e12)
        self.lsf_models = lsf_models
        print "blaze mods"
        #blaze_models = [ConstantMultiplierModel(spec.norm) for spec in spectra]
        self.blaze_models = [tmb.spectrographs.PiecewisePolynomialSpectrographEfficiencyModel(spec.wv, n_max_part=4.5, degree=2) for spec in target_spectra]
        
        print "stitching models together"
        data_model_trees = []
        for spec_idx in range(len(target_spectra)):
            mods = [self.feature_mod, self.exp_mod]#, self.ctm_mod, self.hmod]
            mods.append(self.lsf_models[spec_idx])
            mods.append(self.blaze_models[spec_idx])
            dr_tree = modeling.DataRootedModelTree(mods, target_spectra[spec_idx].flux, target_spectra[spec_idx].inv_var)
            data_model_trees.append(dr_tree)
            tree_res = dr_tree()
            blaze_mod = self.blaze_models[spec_idx]
            blaze_input_res = dr_tree._result_chain[dr_tree.model_index(blaze_mod)]
            norm = target_spectra[spec_idx].normalize()
            blaze_mod.retrain(norm/np.where(blaze_input_res > 0.01, blaze_input_res, 0.5), np.ones(norm.shape))
        
        #import pdb; pdb.set_trace()
        #set up the fit models
        mstack0 = copy(self.blaze_models)
        blaze_alone = modeling.FitState(models=mstack0, clamping_factor=10.1, max_iter=1)
        features_alone = modeling.FitState(models=[self.feature_mod], clamping_factor=10.0, max_iter=1, cleanup_func=thaw_params, setup_func=freeze_params)
        
        teff_only = modeling.FitState(models=[self.feature_mod], 
                                      clamping_factor=10.0, 
                                      max_iter=3, 
                                      cleanup_func=lambda x: thaw_params(x, teff=True), 
                                      setup_func=lambda x: freeze_params(x, teff=False))
        
        vmicro_only = modeling.FitState(models=[self.feature_mod], 
                                      clamping_factor=10.0, 
                                      max_iter=3, 
                                      cleanup_func=lambda x: thaw_params(x, vmicro=True), 
                                      setup_func=lambda x: freeze_params(x, vmicro=False))
        
        gmr_only = modeling.FitState(models=[self.feature_mod], 
                                      clamping_factor=10.0, 
                                      max_iter=3, 
                                      cleanup_func=lambda x: thaw_params(x, gamma_ratio_5000=True), 
                                      setup_func=lambda x: freeze_params(x, gamma_ratio_5000=False))
        
        #together_models = []
        #together_models.extend(self.blaze_models)
        #together_models.append(self.feature_mod)
        #all_together_now = modeling.FitState(models =together_models, clamping_factor=20.0, max_iter=max_iter, max_reweighting_iter=10)
        
        fit_states = []
        for slosh_idx in range(5):
            fit_states.append(features_alone)
            fit_states.append(blaze_alone)
        for slosh_idx in range(5):
            fit_states.append(teff_only)
            fit_states.append(gmr_only)
            #fit_states.append(vmicro_only)
            fit_states.append(blaze_alone)
            fit_states.append(features_alone)
        for slosh_idx in range(20):
            fit_states.append(teff_only)
            fit_states.append(gmr_only)
            fit_states.append(vmicro_only)
            fit_states.append(blaze_alone)
            fit_states.append(features_alone)
        #fit_states.append(all_together_now)
        super(SpectralModeler, self).__init__(data_model_trees)
        fit_policy = modeling.FitPolicy(self, fit_states=fit_states, finish_callback=write_results, iteration_callback=iter_cleanup)
        self.set_fit_policy(fit_policy)

def fit_lowres_spectrum(spec_idx, plot=True, max_iter=20):
    hf = h5py.File("globulars_all.h5", "r")
    spectrum_wvs = np.power(10.0, np.array(hf["log_wvs"]))
    spectrum_lsf = np.array(hf["resolution"])
    min_wv = spectrum_wvs[0]
    max_wv = spectrum_wvs[-1] 
    
    #load the sspp results
    try:
        sspp_dict = json.load(open("sspp_model_files/spec_{}.json".format(spec_idx)))
    except IOError:
        sspp_dict = {"teff":5000.0}
    
    pre_grouped = False
    if not args.input_h5 is None:
        ldat = pd.read_hdf(args.input_h5, "fdat")
        print "WARNING: input-h5 flag supersedes ll flag"
    else:
        ldat = tmb.io.read_linelist(args.ll, file_type=args.lltype)
    
    #clip out wavelengths not present
    ldat = ldat[(ldat.wv > min_wv)*(ldat.wv < max_wv)]    
    
    sp_indicators = [tmb.features.indicator_factory(val, 0.07) for val in [3.0, 22.0, 22.1, 24.05, 25.05, 26.0, 26.1, 27.0, 28.0, 38.05, 58.1]]
    grouper = tmb.features.SpeciesGrouper(sp_indicators, ungrouped_val="last")
    
    npts = int(np.log(max_wv/min_wv)*args.model_resolution)
    model_wv = np.exp(np.linspace(np.log(min_wv), np.log(max_wv), npts))
    print "generating lsf model"
    lsf_models =  [tmb.resolution.LineSpreadFunctionModel(model_wv, spectrum_wvs, spectrum_lsf)]
    
    spec = tmb.Spectrum(spectrum_wvs, np.array(hf["flux"][spec_idx]), np.array(hf["invvar"][spec_idx]))
    smod = SpectralModeler(target_spectra=[spec], lsf_models=lsf_models, ldat=ldat, species_grouper=grouper, max_iter=max_iter, spectrum_id=spec_idx, start_teff=sspp_dict["teff"])
    
    first_stime = time.time()
    #damping_schedule = {0:(1e5, 1e5), 3:(1e4, 1e4), 5:(1e3, 1e3), 7:(3e2, 3e2), 12:(1e2, 1e2), 18:(5.0, 5.0)}
    
    print "fit iterating"
    #import pdb; pdb.set_trace()
    smod.converge()


if __name__ == "__main__":
    args = parser.parse_args()
    
    #pool = multiprocessing.Pool(multiprocessing.cpu_count())
    #pool.map(fit_lowres_spectrum, range(1024))
    
    #import pdb; pdb.set_trace()
    fit_lowres_spectrum(15)
    
