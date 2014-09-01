import thimbles as tmb
from thimbles.modeling import modeling
import numpy as np
import pandas as pd
import scipy.sparse
import matplotlib.pyplot as plt
from thimbles.velocity import template_rv_estimate
import argparse
import cPickle

parser = argparse.ArgumentParser("fit prototype")
parser.add_argument("fname")
parser.add_argument("--ll", default="/home/tim/linelists/vald/5000_3.0_0.05.vald")
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
parser.add_argument("--model-resolution", type=float, default=5e5)
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

class ConstantMultiplierModel(object):
    
    def __init__(self, mult):
        self.mult = mult
        self._lin_op = scipy.sparse.dia_matrix((self.mult, 0), shape=(len(self.mult), len(self.mult)))    
    
    def __call__(self, input, **kwargs):
        return self.mult*input
    
    def as_linear_op(self, input, **kwargs):
        return self._lin_op

if __name__ == "__main__":
    args = parser.parse_args()
    
    spectra = tmb.io.spec_io.read_h5(args.fname)
    pre_grouped = False
    if not args.input_h5 is None:
        ldat = pd.read_hdf(args.input_h5, "fdat")
        pre_grouped = True
        print "WARNING: input-h5 flag supersedes ll flag"
    else:
        ldat = tmb.io.read_linelist(args.ll, file_type=args.lltype)
    
    rv_estimate = float(open(args.rv_file).readlines()[0])
    for spec in spectra:
        spec.set_rv(rv_estimate)
    
    min_wv = np.min([np.min(spec.wv) for spec in spectra])
    max_wv = np.max([np.max(spec.wv) for spec in spectra])
    
    svfm = tmb.features.SaturatedVoigtFeatureModel(
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
    model_resolution=args.model_resolution
    )
    
    print "generating model components"
    model_wv = svfm.model_wv
    print "ctm model"
    ctm_mod = tmb.continuum.BlackBodyContinuumModel(model_wv, args.teff)
    print "hmod"
    hmod = tmb.hydrogen.HydrogenForegroundModel(model_wv, 2.0, 5300, 1e12)
    print "lsf mods"
    if not args.lsf_pkl is None:
        print "loading lsf models"
        lsf_models = cPickle.load(open(args.lsf_pkl))
    else:
        print "generating lsf models"
        lsf_models =  [tmb.resolution.LineSpreadFunctionModel(model_wv, spec.wv, spec.wv_soln.lsf) for spec in spectra]
    print "blaze mods"
    #blaze_models = [ConstantMultiplierModel(spec.norm) for spec in spectra]
    blaze_models = [tmb.spectrographs.PiecewisePolynomialSpectrographEfficiencyModel(spec.wv, n_max_part=7, degree=3) for spec in spectra]
    
    print "building modeler"
    modeler = modeling.Modeler()
    for spec_idx in range(len(spectra)):
        mods = [svfm, ctm_mod, hmod]
        mods.append(lsf_models[spec_idx])
        mods.append(blaze_models[spec_idx])
        chain = modeling.ModelChain(mods, spectra[spec_idx].flux, spectra[spec_idx].inv_var)
        modeler.add_chain(chain)
        chain_res = chain()
        blaze_input_res = chain._result_chain[-2]
        norm = spectra[spec_idx].normalize()
        blaze_mod = blaze_models[spec_idx]
        blaze_mod.retrain(norm/np.where(blaze_input_res > 0.01, blaze_input_res, 0.5), np.ones(norm.shape))
    
    for iter_idx in range(2):
        print "fit iterating"
        modeler.iterate(svfm)
        for i in range(len(blaze_models)):
            modeler.iterate(blaze_models[i])
        svfm.fit_offsets()
        svfm.calc_feature_matrix()
    
    svfm.fdat.to_hdf(args.output_h5, "fdat")

