
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
parser.add_argument("--gamma-iter", default=20, type=int)


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
    n_iter=25,
    log_rate=100,
):  
    tl_gb = params.groupby(["teff", "logg"])
    tl_groups = tl_gb.groups
    
    gammas = []
    avg_params = tl_gb.mean()
    avg_std = tl_gb.std()
    tl_tuples = list(avg_params.index)
    for tl_tup in tl_tuples:
        tl_idxs = tl_groups[tl_tup]
        reshaped_ews = params["ew"].ix[tl_idxs].values.reshape((-1, 1))
        cteff, clogg = tl_tup
        depths = 1.0 - flux.ix[tl_idxs]
        
        sigma = avg_params.ix[tl_tup]["doppler_width"]
        gamma = gamma_model(clogg)
        for iter_idx in range(n_iter):
            prof = tmb.profiles.voigt(wvs, transition["wv"], sigma, gamma)
            gamma_eps = 0.1*gamma
            p_minus = tmb.profiles.voigt(wvs, transition["wv"], sigma, gamma-gamma_eps)
            p_plus = tmb.profiles.voigt(wvs, transition["wv"], sigma, gamma+gamma_eps)
            gamma_deriv = (p_plus - p_minus)/gamma_eps
            
            depth_model = prof.reshape((1, -1))*reshaped_ews
            resids = depths - depth_model
            mean_resid = np.mean(resids, axis=0)
            if False:#iter_idx % 50 == 0:
                plt.plot(mean_resid)
                plt.ylim(-0.005, 0.005)
                plt.show()
            
            gamma_adj = np.sum(mean_resid*gamma_deriv)/np.sum(gamma_deriv**2)
            gamma = gamma + gamma_adj
            gamma = np.abs(gamma)
            #put in a weak prior for zero gamma
            data_weight = len(depths)
            gamma *= data_weight/(data_weight + 0.01)
        gammas.append(gamma)
    
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
    for attr_name in transition.index:
        data_dict[attr_name] = np.repeat(transition[attr_name], nrows)
    agg_df = pd.DataFrame(data_dict)
    return agg_df

def fit_cog(params, transition, gamma_model, diagnostic_plot=False):
    tew_adj = np.linspace(-2.0, 2.0, 50)
    gamma_ratio = gamma/np.mean(params["doppler_width"])
    cog = tmb.cog.voigt_saturation_curve(gamma_ratio)
    
    cog_delta_sums = []
    bshifts = []
    ltew = np.log10(params["tew"])
    xonh = params["xonh"]
    for adj_idx in range(len(tew_adj)):
        shift = np.median(params["xonh"] - cog.inverse(ltew-tew_adj[adj_idx]))
        bshifts.append(shift)
        cog_pred = cog(xonh - shift) + tew_adj[adj_idx]
        deltas = ltew - cog_pred
        cog_delta_sums.append(np.sum(deltas**2))
    
    best_idx = np.argmin(cog_delta_sums)
    best_adj = tew_adj[best_idx]
    best_shift = bshifts[best_idx]
    if diagnostic_plot:
        plt.scatter(xonh, ltew, c=params["met"], s=80, alpha=0.7)
        cbar = plt.colorbar()
        cbar.set_label("photospheric [Fe/H]", fontsize=16)
        x = np.linspace(np.min(xonh)-0.5, np.max(xonh)+0.5, 101)
        plt.plot(x, cog(x-best_shift)+best_adj)
        plt.xlabel("[X/H]", fontsize=16)
        plt.ylabel("$log(EW)-log(W_{doppler})$", fontsize=16)
        plt.show()
    
    return best_shift, best_adj


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
        params.to_hdf(output_path, "ews")
        gamma_res.to_hdf(output_path, "gammas")
    
    #gammas.append(cgam)
    #bshift, badj = fit_cog(params.ix[mask], transition, gamma=cgam)
    #shifts.append(bshift)
    #adjs.append(badj)

if False:
    fit_matrix = np.vander(loggs, 2)
    lgams = np.log(gammas)
    nan_mask = np.isnan(lgams)
    weights = np.logical_not(nan_mask).astype(float)
    lgams = np.where(nan_mask, 0, lgams)
    weights *= np.exp(lgams)
    l_coeffs = np.linalg.lstsq(fit_matrix*weights.reshape((-1, 1)), lgams*weights)[0]
    
    lin_x = np.linspace(-1, 4, 100)
    lin_fm = np.vander(lin_x, 2)
    mod_y = np.dot(lin_fm, l_coeffs)
    plt.scatter(loggs, np.log(gammas))
    plt.plot(lin_x, mod_y)
    plt.xlabel("Log(g)", fontsize=16)
    plt.ylabel("log($\gamma$)", fontsize=16)
    plt.show()
