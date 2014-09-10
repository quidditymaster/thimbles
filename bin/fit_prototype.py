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
    
    def __init__(self, target_spectra, lsf_models, ldat, species_grouper='last'):
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
        print "ctm model"
        self.ctm_mod = tmb.continuum.BlackBodyContinuumModel(model_wv, args.teff)
        print "hmod"
        self.hmod = tmb.hydrogen.HydrogenForegroundModel(model_wv, np.array([2.5, 0.1]), 5300, 5e12)
        self.lsf_models = lsf_models
        print "blaze mods"
        #blaze_models = [ConstantMultiplierModel(spec.norm) for spec in spectra]
        self.blaze_models = [tmb.spectrographs.PiecewisePolynomialSpectrographEfficiencyModel(spec.wv, n_max_part=1.5, degree=2) for spec in target_spectra]
        
        print "building modeler"
        modeler = modeling.Modeler()
        for spec_idx in range(len(target_spectra)):
            mods = [self.feature_mod, self.ctm_mod, self.hmod]
            mods.append(self.lsf_models[spec_idx])
            mods.append(self.blaze_models[spec_idx])
            chain = modeling.ModelChain(mods, target_spectra[spec_idx].flux, target_spectra[spec_idx].inv_var)
            modeler.add_chain(chain)
            chain_res = chain()
            blaze_input_res = chain._result_chain[-2]
            norm = target_spectra[spec_idx].normalize()
            blaze_mod = self.blaze_models[spec_idx]
            blaze_mod.retrain(norm/np.where(blaze_input_res > 0.01, blaze_input_res, 0.5), np.ones(norm.shape))
        self.modeler = modeler
    
    def iterate(self, models=None):
        if models is None:
            models = [self.feature_mod, self.hmod]
            models.extend(self.blaze_models)
        self.modeler.iterate(models)

def fit_lowres_spectrum(spec_idx, plot=True):
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
    
    sp_indicators = [tmb.features.indicator_factory(val, 0.05) for val in [3.0, 22.0, 22.1, 24.0, 24.1, 26.0, 26.1, 27.0, 28.0, 58.1]]
    grouper = tmb.features.SpeciesGrouper(sp_indicators, ungrouped_val="last")
    
    npts = int(np.log(max_wv/min_wv)*args.model_resolution)
    model_wv = np.exp(np.linspace(np.log(min_wv), np.log(max_wv), npts))
    print "generating lsf model"
    lsf_models =  [tmb.resolution.LineSpreadFunctionModel(model_wv, spectrum_wvs, spectrum_lsf)]
    
    if plot:
        fig, axes = plt.subplots(2)
    spec = tmb.Spectrum(spectrum_wvs, np.array(hf["flux"][spec_idx]), np.array(hf["invvar"][spec_idx]))
    smod = SpectralModeler(target_spectra=[spec], lsf_models=lsf_models, ldat=ldat, species_grouper=grouper)
    
    max_iter = 15
    first_stime = time.time()
    
    #damping_schedule = {0:(1e5, 1e5), 3:(1e4, 1e4), 5:(1e3, 1e3), 7:(3e2, 3e2), 12:(1e2, 1e2), 18:(5.0, 5.0)}
    
    for iter_idx in range(max_iter):
        iter_stime = time.time()
        
        #set the damping
        #new_damps = damping_schedule.get(iter_idx)
        #if not new_damps is None:
        #    tdamp, vdamp = new_damps
        #    smod.feature_mod.fit_damping_factors["theta"] = tdamp
        #    smod.feature_mod.fit_damping_factors["vmicro"] = vdamp
        
        print "fit iterating"
        smod.iterate()
        
        fpvec = smod.feature_mod.get_pvec() 
        #import pdb; pdb.set_trace()
        if iter_idx < max_iter - 5:
            print "applying fudge factors"
            if iter_idx < 2:
                fpvec[2:] = fpvec[2:] + (5.0-2*iter_idx)
                smod.feature_mod.set_pvec(fpvec)
            cteff = smod.feature_mod.teff
            smod.ctm_mod.teff = cteff
            smod.iterate(smod.blaze_models)
        
        print "teff {:5.4f}  vmicro {:5.3f}".format(5040.0/fpvec[0], fpvec[1])
        print "rest of strengths {}".format(fpvec[2:])
        
        if plot:
            lwv = np.log10(spec.wv)
            axes[0].cla()
            axes[0].plot(lwv, spec.flux, lwv, smod.modeler.chains[0]())
            axes[1].cla()
            axes[1].plot(lwv, np.sqrt((smod.modeler.chains[0]()-spec.flux)**2*spec.inv_var))
            axes[1].set_ylim(0.0, 15.0)
            fig.savefig("figures/spec_{}_iter_{}".format(spec_idx, iter_idx))
                #plt.show()
            
        smod.feature_mod.fit_offsets()
        #svfm.calc_feature_matrix()
        smod.feature_mod.calc_grouping_matrix()
        smod.feature_mod.collapse_feature_matrix()
        
        ctime = time.time()
        iter_duration = ctime - iter_stime
        total_duration = ctime-first_stime
        print "iter {} finished in {},  {} seconds average".format(iter_idx+1, iter_duration, total_duration/float(iter_idx+1))
        
        smod.feature_mod.fdat.to_hdf("ew_files/spec_{}_ew.h5".format(spec_idx), "fdat")
        #cPickle.dump(smod, open("model_files/spec_{}_mod.pkl", "w"))
        f = open("model_files/spec_{}_params.txt".format(spec_idx), "w")
        f.write("{}  {}\n".format(smod.feature_mod.teff, smod.feature_mod.vmicro))
        f.flush()
        f.close()


if __name__ == "__main__":
    args = parser.parse_args()
    
    #pool = multiprocessing.Pool(multiprocessing.cpu_count())
    #pool.map(fit_lowres_spectrum, range(1024))
    
    #import pdb; pdb.set_trace()
    fit_lowres_spectrum(13)
    