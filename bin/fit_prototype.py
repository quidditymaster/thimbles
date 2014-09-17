import thimbles as tmb
import time
import h5py
from thimbles.modeling import modeling
import numpy as np
import pandas as pd
import scipy.sparse
import matplotlib.pyplot as plt
from thimbles.velocity import template_rv_estimate
import argparse
import cPickle
import multiprocessing
from copy import copy

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



class SpectralModeler(object):
    
    def __init__(self, target_spectra, lsf_models, ldat, species_grouper, max_iter):
        self.target_spectra = target_spectra
        
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
            teff=args.teff,
            model_resolution=args.model_resolution,
            species_grouper=species_grouper
        )
        
        print "generating model components"
        model_wv = self.feature_mod.model_wv
        #print "ctm model"
        #self.ctm_mod = tmb.continuum.BlackBodyContinuumModel(model_wv, args.teff)
        #print "hmod"
        #self.hmod = tmb.hydrogen.HydrogenForegroundModel(model_wv, np.array([2.5, 0.1]), 5300, 5e12)
        self.lsf_models = lsf_models
        print "blaze mods"
        #blaze_models = [ConstantMultiplierModel(spec.norm) for spec in spectra]
        self.blaze_models = [tmb.spectrographs.PiecewisePolynomialSpectrographEfficiencyModel(spec.wv, n_max_part=1.5, degree=2) for spec in target_spectra]
        
        print "building modeler"
        data_model_trees = []
        for spec_idx in range(len(target_spectra)):
            mods = [self.feature_mod]#, self.ctm_mod, self.hmod]
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
        model_net = modeling.DataModelNetwork(data_model_trees)
        #set up the fit models
        mstack0 = copy(self.blaze_models)
        blaze_alone = modeling.FitState(models=mstack0, clamping_factor=0.1, max_iter=1)
        mstack1 = []
        #mstack1.extend(self.blaze_models)
        mstack1.append(self.feature_mod)
        features_alone = modeling.FitState(models=mstack1, clamping_factor=1.0, max_iter=1)
        together_models = []
        together_models.extend(self.blaze_models)
        together_models.append(self.feature_mod)
        together = modeling.FitState(models =together_models, clamping_factor=10.0, max_iter=max_iter, max_reweighting_iter=10)
        fit_states = []
        for slosh_idx in range(5):
            fit_states.append(features_alone)
            fit_states.append(blaze_alone)
        fit_states.append(together)
        fit_policy = modeling.FitPolicy(model_net, fit_states=fit_states)
        model_net.set_fit_policy(fit_policy)
        self.model_net = model_net
    
    def iterate(self):
        return self.model_net.iterate()
        #if models is None:
        #    models = [self.feature_mod]#, self.hmod]
        #    models.extend(self.blaze_models)
        #self.modeler.iterate(models)
    
    def converge(self, callback):
        self.model_net.converge(callback)

def fit_lowres_spectrum(spec_idx, plot=True, max_iter=50):
    hf = h5py.File("globulars_all.h5", "r")
    spectrum_wvs = np.power(10.0, np.array(hf["log_wvs"]))
    spectrum_lsf = np.array(hf["resolution"])
    min_wv = spectrum_wvs[0]
    max_wv = spectrum_wvs[-1] 
    
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
    
    if plot:
        fig, axes = plt.subplots(2)
    spec = tmb.Spectrum(spectrum_wvs, np.array(hf["flux"][spec_idx]), np.array(hf["invvar"][spec_idx]))
    smod = SpectralModeler(target_spectra=[spec], lsf_models=lsf_models, ldat=ldat, species_grouper=grouper, max_iter=max_iter)
    
    first_stime = time.time()
    #damping_schedule = {0:(1e5, 1e5), 3:(1e4, 1e4), 5:(1e3, 1e3), 7:(3e2, 3e2), 12:(1e2, 1e2), 18:(5.0, 5.0)}
    
    def iter_cleanup(fstate):
        teff = smod.feature_mod.teff
        vmicro = smod.feature_mod.vmicro
        gamma_rat = smod.feature_mod.mean_gamma_ratio
        print "teff {:5.3f}  vmicro {:5.3f} gamma_ratio {:5.4f} ".format(teff, vmicro, gamma_rat)
        
        if plot:
            lwv = np.log10(spec.wv)
            axes[0].cla()
            model_values = smod.model_net.trees[0]()
            axes[0].plot(lwv, spec.flux, lwv, model_values)
            axes[1].cla()
            resid = model_values-spec.flux
            axes[1].plot(lwv, np.sign(resid)*np.sqrt(resid**2*spec.inv_var))
            axes[1].set_ylim(-15.0, 15.0)
            fig.savefig("figures/spec_{}_cfit.png".format(spec_idx))
        
        smod.feature_mod.fit_offsets()
        smod.feature_mod.calc_grouping_matrix()
        smod.feature_mod.collapse_feature_matrix()
        
        ctime = time.time()
        total_duration = ctime-first_stime
        print "iter {} finished,  {} seconds average".format(fstate.iter_num+1, total_duration/float(fstate.iter_num+1))
    
    print "fit iterating"
    #import pdb; pdb.set_trace()
    smod.converge(iter_cleanup)
    
    smod.feature_mod.fdat.to_hdf("ew_files/spec_{}_ew.h5".format(spec_idx), "fdat")
    #cPickle.dump(smod, open("model_files/spec_{}_mod.pkl", "w"))
    f = open("model_files/spec_{}_params.txt".format(spec_idx), "w")
    f.write("{}  {}  {}\n".format(smod.feature_mod.teff, smod.feature_mod.vmicro, smod.feature_mod.mean_gamma_ratio))
    f.flush()
    f.close()


if __name__ == "__main__":
    args = parser.parse_args()
    
    #pool = multiprocessing.Pool(multiprocessing.cpu_count())
    #pool.map(fit_lowres_spectrum, range(1024))
    
    #import pdb; pdb.set_trace()
    fit_lowres_spectrum(14)
    