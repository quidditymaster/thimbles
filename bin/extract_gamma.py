
import numpy as np
import matplotlib as mpl
mpl.use("qt4Agg")
import matplotlib.pyplot as plt

import scipy
import scipy.integrate
import pandas as pd

import thimbles as tmb

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--grid-dir", required=True)
parser.add_argument("--grid-manifest")
parser.add_argument("--output-dir", required=True)
parser.add_argument("--gamma-iter", default=50, type=int)

def extract_ews(fluxes, wvs):
    return scipy.integrate.simps(1.0-fluxes, x=wvs, axis=1)

def get_doppler_widths(teffs, vmicros, transition):
    wv = transition["wv"]
    mol_weight = tmb.ptable.ix[(transition["z"], 0)]["weight"]
    twidth = tmb.utils.misc.thermal_width(teffs, wv, mol_weight)
    vwidth = (vmicros/tmb.speed_of_light)*wv
    return np.sqrt(vwidth**2 + twidth**2)

def params_from_pgrid(fname):
    flux = pd.read_hdf(fname, "flux")
    transition = pd.read_hdf(fname, "transition")
    params = pd.read_hdf(fname, "params")
    wvs = pd.read_hdf(fname, "wvs").values
    
    params["ew"] = extract_ews(flux, wvs)
    params["doppler_width"] = get_doppler_widths(params["teff"], params["vmicro"], transition)
    params["tew"] = params["ew"]/params["doppler_width"]
    params["xonh"] = params["met"] + params["xonfe"]
    
    return flux, transition, params, wvs

def make_depth_model(model_params, ews, wvs):
    sigma, gamma, log_stretch = model_params
    
    cog = tmb.cog.voigt_saturation_curve(np.abs(gamma/sigma))
    log_taus = cog.inverse(np.log10(ews) + log_stretch)
    
    opac_profile = tmb.profiles.voigt(wvs, np.median(wvs), sigma, gamma)
    #prof_center_depth = opac_profile[len(opac_profile)//2]
    #rel_depths = central_depths/(prof_center_depth*ews)
    #rel_depths = np.clip(rel_depths, 0, 0.99)
    #import pdb; pdb.set_trace()
    #log_taus = cog.inverse(np.log10(1.0-rel_depths))
    #log_taus = np.where(rel_depths > 0.85, -.0, -1.0)
    #log_taus = np.repeat(-2.0, len(rel_depths))
    
    op_depths = np.power(10.0, log_taus.reshape((-1, 1)))*opac_profile #-taus.reshape((-1, 1))*opac_prof
    profiles = 1.0 - np.power(10.0, -op_depths)
    normalizations = scipy.integrate.simps(profiles, axis=1, x=wvs)
    profiles /= normalizations.reshape((-1, 1))
    depth_model = profiles*ews.reshape((-1, 1))
    return depth_model

def get_resids(model_params, ews, wvs, depths):
    dmod = make_depth_model(model_params, ews, wvs)
    return (depths-dmod).reshape((-1,))

def fit_gamma(
        flux,
        params,
        transition,
        wvs,
        gamma_model,
        min_log_gamma_ratio=-3,
        max_log_gamma_ratio=1,
        n_steps=101,
):
    tl_gb = params.groupby(["teff", "logg"])
    tl_groups = tl_gb.groups
    
    dwvs = scipy.gradient(wvs)
    #gammas = []
    profile_param_results = []
    avg_param_list = []
    avg_std = []#tl_gb.std()
    tl_tuples = list(tl_groups.keys())
    
    trial_gamma_ratios = np.power(10.0, np.linspace(min_log_gamma_ratio, max_log_gamma_ratio, n_steps))
    cog_set = [tmb.cog.voigt_saturation_curve(tgr) for tgr in trial_gamma_ratios]
    
    for tl_tup in tl_tuples:
        tl_idxs = tl_groups[tl_tup]
        ew_vals = params["ew"].ix[tl_idxs].values
        cteff, clogg = tl_tup
        depths = (1.0 - flux.ix[tl_idxs]).values
        
        sigma = np.mean(params["doppler_width"].ix[tl_idxs])
        opt_res = scipy.optimize.leastsq(get_resids, [0.9*sigma, 0.1*sigma, 0.5], args=(ew_vals, wvs, depths))
        #import pdb; pdb.set_trace()
        
        best_params = opt_res[0]
        best_params[:2] = np.abs(best_params[:2])
        print("teff, logg", tl_tup)
        print("best_params", opt_res[0])
        profile_param_results.append(best_params)
        
        avg_param_list.append(np.mean(params.ix[tl_idxs]))
        avg_std.append(np.std(params.ix[tl_idxs]))
        
        #import pdb; pdb.set_trace()
        if False:#iter_idx % 50 == 0:
            fig, axes = plt.subplots(2)
            depth_model = make_depth_model(best_params, ew_vals, wvs)
            
            for i in range(len(depths)):
                axes[0].plot(depths[i], color="k")
                axes[0].plot(depth_model[i], color="r", alpha=0.8)
                axes[1].plot(depths[i]-depth_model[i])
            plt.show()
    
    avg_params = pd.DataFrame(avg_param_list)
    avg_std = pd.DataFrame(avg_std)
    
    #build the aggregated dataframe
    sigmas = [bp[0] for bp in profile_param_results]
    gammas = [bp[1] for bp in profile_param_results]
    stretches = [bp[2] for bp in profile_param_results]
    data_dict =dict(
        teff=[tl_tup[0] for tl_tup in tl_tuples],
        logg=[tl_tup[1] for tl_tup in tl_tuples],
        sigma=sigmas,
        gamma=gammas,
        stretch=stretches,
        )
    nrows = len(gammas)
    for cname in avg_params.columns:
        data_dict[cname] = avg_params[cname].values
        data_dict[cname + "_std"] = avg_std[cname].values
    agg_df = pd.DataFrame(data_dict)
    return agg_df


if __name__ == "__main__":
    import os
    args = parser.parse_args()
    grid_dir = args.grid_dir
    output_dir = args.output_dir
    if args.grid_manifest is None:
        pgrid_files = [fname for fname in os.listdir(grid_dir) if fname[-6:] == ".pgrid"]
    else:
        pgrid_files = [fname.strip() for fname in open(args.grid_manifest, "r").readlines()]
    
    gamma_model = lambda x: 0.2
    for pg_file in pgrid_files:
        fpath = os.path.join(grid_dir, pg_file)
        output_path = os.path.join(output_dir, pg_file + ".ews.hdf") 
        print("processing", pg_file)
        print("extracting ews")
        flux, transition, params, wvs = params_from_pgrid(fpath)
        print("beginning gamma iterations")
        gamma_res = fit_gamma(flux, params, transition, wvs, gamma_model=gamma_model)
        #import pdb; pdb.set_trace()
        params.to_hdf(output_path, "ews")
        gamma_res.to_hdf(output_path, "gammas")
        transition.to_hdf(output_path, "transition")


