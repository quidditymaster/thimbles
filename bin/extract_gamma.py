
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
parser.add_argument("--sigma-clip", type=float, default=2.5)


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


def fit_gamma(
        flux,
        params,
        transition,
        wvs,
        gamma_model,
        n_iter=20,
        n_mask_iter=3,
        sigma_clip=2.5,
        log_rate=100,
):  
    tl_gb = params.groupby(["teff", "logg"])
    tl_groups = tl_gb.groups
    
    gammas = []
    avg_params = tl_gb.mean()
    avg_param_list = []
    avg_std = []#tl_gb.std()
    tl_tuples = list(avg_params.index)
    for tl_tup in tl_tuples:
        tl_idxs = tl_groups[tl_tup]
        ew_vals_full = params["ew"].ix[tl_idxs].values
        cteff, clogg = tl_tup
        depths = 1.0 - flux.ix[tl_idxs]
        
        sigma = avg_params.ix[tl_tup]["doppler_width"]
        gamma = gamma_model(clogg)
        
        mask = np.ones(len(depths), dtype=bool)
        for mask_iter in range(n_mask_iter):
            ew_vals = ew_vals_full[mask]
            depth_vals = depths[mask]
            for iter_idx in range(n_iter):
                prof = tmb.profiles.voigt(wvs, transition["wv"], sigma, gamma)
                gamma_eps = 0.1*gamma
                p_minus = tmb.profiles.voigt(wvs, transition["wv"], sigma, gamma-gamma_eps)
                p_plus = tmb.profiles.voigt(wvs, transition["wv"], sigma, gamma+gamma_eps)
                gamma_deriv = (p_plus - p_minus)/gamma_eps
                
                depth_model = prof.reshape((1, -1))*ew_vals.reshape((-1, 1))
                resids = depth_vals - depth_model
                
                mean_resid = np.mean(resids, axis=0)
                gamma_adj = np.sum(mean_resid*gamma_deriv)/np.sum(gamma_deriv**2)
                gamma = gamma + gamma_adj
                gamma = np.abs(gamma)
                if gamma == 0:
                    gamma = 1e-1*np.random.random()
                #put in a weak prior for zero gamma
                data_weight = len(depth_vals)
                gamma *= data_weight/(data_weight + 0.01)
            full_mod = prof.reshape((1, -1))*ew_vals_full
            full_resids = depths - full_mod
            sq_resids_sums = np.sum(full_resids**2, axis=1)
            abs_resid_sums = np.abs(sq_resid_sums)
            med_sqr = np.median(abs_resid_sums)
            if mask_iter < n_mask_iter-1:
                mask = abs_resid_sums <= sigma_clip*med_sqr
        avg_param_list.append(np.mean(params.ix[tl_idxs][mask]))
        avg_std.append(np.std(params.ix[tl_idxs][mask]))
            
            if False:#iter_idx % 50 == 0:
                plt.plot(mean_resid)
                plt.ylim(-0.005, 0.005)
                plt.show()
        
        gammas.append(gamma)
    avg_params = pd.DataFrame(avg_param_list)
    avg_std = pd.DataFrame(avg_std)
        
    #build the aggregated dataframe
    data_dict =dict(
        teff=[tl_tup[0] for tl_tup in tl_tuples],
        logg=[tl_tup[1] for tl_tup in tl_tuples],
        gamma=gammas,
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
        gamma_res = fit_gamma(flux, params, transition, wvs, gamma_model=gamma_model, n_iter=args.gamma_iter)
        params.to_hdf(output_path, "ews")
        gamma_res.to_hdf(output_path, "gammas")
        transition.to_hdf(output_path, "transition")


