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
parser.add_argument("--output", required=True)


def fit_cog(
        params,
        transition,
        gamma_model,
        diagnostic_plot=False
):
    tew_adj = np.linspace(-1.5, 0.5, 25)
    log_gamma = gamma_model(params.mean(), transition)
    n_delta_gam = 11
    delta_lgams = np.linspace(-0.5, 0.5, n_delta_gam)
    
    opt_esum = np.inf
    opt_gamma_adj_idx = 0
    opt_ew_adj_idx = 0
    opt_x_shift = 0
    
    mean_dop_width = np.mean(params["doppler_width"])
    
    for gam_idx in range(n_delta_gam):
        gamma = np.power(10.0, log_gamma + delta_lgams[gam_idx])
        gamma_ratio = gamma/mean_dop_width
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
        if cog_delta_sums[best_idx] < opt_esum:
            opt_gamma_adj_idx = gam_idx
            opt_ew_adj_idx = best_idx
            opt_esum = cog_delta_sums[best_idx]
            opt_x_shift = bshifts[best_idx]
    
    best_ew_adj = tew_adj[opt_ew_adj_idx]
    best_gam_adj = delta_lgams[opt_gamma_adj_idx]
    best_gam = np.power(10.0, log_gamma + best_gam_adj)
    
    if diagnostic_plot:
        gamma_ratio = best_gam/mean_dop_width
        cog = tmb.cog.voigt_saturation_curve(gamma_ratio)
        plt.scatter(xonh, ltew, c=params["met"], s=80, alpha=0.7)
        cbar = plt.colorbar()
        cbar.set_label("photospheric [Fe/H]", fontsize=16)
        x = np.linspace(np.min(xonh)-0.5, np.max(xonh)+0.5, 101)
        plt.plot(x, cog(x-opt_x_shift)+best_ew_adj)
        plt.xlabel("[X/H]", fontsize=16)
        plt.ylabel("$log(EW)-log(W_{doppler})$", fontsize=16)
        plt.show()
    
    return best_ew_adj, best_gam,  opt_x_shift


if __name__ == "__main__":
    import os
    args = parser.parse_args()
    grid_dir = args.grid_dir
    output = args.output
    if args.grid_manifest is None:
        pgrid_files = [fname for fname in os.listdir(grid_dir) if fname[-8:] == ".ews.hdf"]
    else:
        pgrid_files = [fname.strip() for fname in open(args.grid_manifest, "r").readlines()]
    
    #fit gammas
    gamma_df_list = []
    for fname in pgrid_files:
        fpath = os.path.join(grid_dir, fname)
        gamma_df_list.append(pd.read_hdf(fpath, "gammas"))
    
    gamma_df = pd.concat(gamma_df_list)
    #free up the unconcatenated dfs
    del gamma_df_list
    
    log_tew_avgs = np.log10(gamma_df["tew"])
    #mask out very saturated and very weak averge features
    qmask =  (log_tew_avgs > -0.3)
    qmask &= (log_tew_avgs < 0.4) 
    qmask &= gamma_df["tew_std"] < 1.3
    qmask &= gamma_df["logg"] <= 5.0
    
    mdf = gamma_df[qmask]
    #fit_matrix generation
    lgams = np.log10(mdf["gamma"])
    logg = mdf["logg"]
    
    def make_fit_matrix(
            params,
            degrees,
            offsets=None,
    ):
        if offsets is None:
            offsets = {}
        npts = len(params)
        n_coeffs = sum(degrees.values())+1
        fit_mat = np.zeros((npts, n_coeffs))
        col_idx = 1
        fit_mat[:, 0] = 1.0
        
        coeff_interpretations = [("constant", 0)]
        for col_name in sorted(degrees.keys()):
            max_power = degrees[col_name]
            if col_name[0] == "@":
                eval_ns = {"params":params}
                col_vals = eval(col_name[1:], eval_ns)
            else:
                col_vals = params[col_name]
            offset = offsets.get(col_name, 0)
            deltas = col_vals-offset
            for power in range(1, max_power+1):
                fit_mat[:, col_idx] = deltas**power
                col_idx += 1
                coeff_interpretations.append((col_name, power))
        return fit_mat, coeff_interpretations
    
    sp_tups = list(zip(mdf["z"].values, mdf["charge"].values))
    sp_tup_set = sorted(list(set(sp_tups)))
    #unique_species = np.unique(mdf["z"].values)
    species_indexer = {}
    for tup_idx, tup in enumerate(sp_tup_set):
        species_indexer[tup] = tup_idx
    
    resid_mask = np.ones(len(logg), dtype=bool)
    fit_mat, coeff_interpretations = make_fit_matrix(
        mdf,
        degrees={
            "logg":1,
            "ep":1,
            #'@5040.0*params["ep"]/params["teff"]':1,
            #'@5040.0/params["teff"]':2
        },
    )# species=sp_tups, species_indexer=species_indexer, logg_order=3)
    for i in range(5):
        fit_params = np.linalg.lstsq(fit_mat[resid_mask], lgams[resid_mask])[0]
        
        mod_gam = np.dot(fit_mat, fit_params)
        resids = (lgams-mod_gam).values
        
        med_resid = np.median(np.abs(resids))
        #crop out the points with high residuals
        resid_mask = np.abs(resids) < 3.0*med_resid
    
    fig, axes = plt.subplots(1, 4)
    msub=mdf[resid_mask]
    mresids = resids[resid_mask]
    gsize = (10, 25)
    axes[0].hexbin(msub.logg, mresids, gridsize=gsize)
    axes[0].set_ylabel("Residual log(Gamma)", fontsize=16)
    axes[0].set_xlabel("log(g)", fontsize=16)
    axes[1].hexbin(msub.ep, mresids, gridsize=gsize)
    axes[1].set_xlabel("E.P", fontsize=16)
    axes[2].hexbin(msub.met, mresids, gridsize=gsize)
    axes[2].set_xlabel("Photospheric [Fe/H]")
    axes[3].hexbin(np.log10(msub["gamma"]), mresids, gridsize=gsize)
    axes[3].set_xlabel("log(Gamma)", fontsize=16)
    plt.show()
    
    plt.hist(resids, 100)
    plt.show()
    
    def gamma_model(params, transition):
        cur_gam = fit_params[0]
        coeff_idx = 1
        for cname, power in coeff_interpretations[1:]:
            try:
                cvals = params[cname]
            except KeyError:
                cvals = transition[cname]
            cur_gam += fit_params[coeff_idx]*cvals**power
            coeff_idx += 1
        modgam = cur_gam
        return modgam
    
    for fname in pgrid_files:
        fpath = os.path.join(grid_dir, fname)
        ew_params = pd.read_hdf(fpath, "ews")
        
        #transition = pd.read_hdf(fpath, "transition")
        cep = float(fname.split("_")[3])
        transition = pd.Series({"ep":cep})
        
        tlgb = ew_params.groupby(["teff", "logg"])
        groups = tlgb.groups
        
        fc_accum = []
        for tl_tup in groups:
            fcres = fit_cog(ew_params.ix[groups[tl_tup]], transition, gamma_model=gamma_model, diagnostic_plot=True)
            print(fcres)
            fc_accum.append(fcres)

        fc_accum = np.array(fc_accum)
        
