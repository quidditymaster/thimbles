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
        saturation_model,
        tl_tup,
        diagnostic_plot=False
):
    model_sat = saturation_model(params.mean(), transition)
    tew_adj = np.linspace(model_sat, model_sat, 1)
    log_gamma = gamma_model(params.mean(), transition)
    n_delta_gam = 1
    delta_lgams = np.linspace(0.0, 0.0, n_delta_gam)
    
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
        plt.annotate("{}".format(transition), (0.0, -0.9), fontsize=12)
        plt.annotate("{}".format(tl_tup), (-1.0, -1.2), fontsize=16)
        plt.annotate("sat {:2.3f} log($\gamma$){:2.3f} x_shift {:2.3f}".format(best_ew_adj, np.log10(best_gam), opt_x_shift), (-1.0, -1.5), fontsize=16)
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
        cgam_df = pd.read_hdf(fpath, "gammas")
        transition = pd.read_hdf(fpath, "transition")
        for col_name in transition.index:
            cgam_df[col_name] = np.repeat(transition[col_name], len(cgam_df))
        gamma_df_list.append(cgam_df)
    
    gamma_df = pd.concat(gamma_df_list)
    #free up the unconcatenated dfs
    del gamma_df_list
    
    log_tew_avgs = np.log10(gamma_df["tew"])
    #mask out very saturated and very weak averge features
    qmask =  (log_tew_avgs > -0.3)
    qmask &= (log_tew_avgs < 0.4) 
    qmask &= gamma_df["tew_std"] < 1.3
    qmask &= gamma_df["logg"] <= 5.0
    qmask &= gamma_df["teff"] <= 7000.0
    
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
    fit_mat, gamma_coeff_interpretations = make_fit_matrix(
        mdf,
        degrees={
            "logg":2,
            "ep":2,
            "teff":1,
            #'@5040.0*params["ep"]/params["teff"]':1,
            #'@5040.0/params["teff"]':2
        },
    )# species=sp_tups, species_indexer=species_indexer, logg_order=3)
    for i in range(10):
        fit_params = np.linalg.lstsq(fit_mat[resid_mask], lgams[resid_mask])[0]
        
        mod_gam = np.dot(fit_mat, fit_params)
        resids = (lgams-mod_gam).values
        
        med_resid = np.median(np.abs(resids))
        #crop out the points with high residuals
        resid_mask = np.abs(resids) < 2.5*med_resid
    
    mdf["model_gamma"] = mod_gam
    fig, axes = plt.subplots(2, 4)
    msub=mdf[resid_mask]
    mresids = resids[resid_mask]
    gsize = (10, 20)
    axes[0,0].hexbin(msub.logg, mresids, gridsize=gsize)
    axes[0,0].set_ylabel("Residual log(Gamma)", fontsize=16)
    axes[0,0].set_xlabel("log(g)", fontsize=16)
    axes[0,1].hexbin(msub.ep, mresids, gridsize=gsize)
    axes[0,1].set_xlabel("E.P", fontsize=16)
    axes[0,2].hexbin(msub.wv, mresids, gridsize=gsize)
    axes[0,2].set_xlabel("Teff")
    axes[0,3].hexbin(np.log10(msub["gamma"]), mresids, gridsize=gsize)
    axes[0,3].set_xlabel("log(Gamma)", fontsize=16)
    
    msub_lgam = np.log10(msub.gamma)
    alpha=0.4
    axes[1, 0].scatter(msub.logg, msub_lgam, alpha=alpha, c=msub.logg)
    axes[1, 1].scatter(msub.ep, msub_lgam, alpha=alpha, c=msub.logg)
    axes[1, 2].scatter(msub.wv, msub_lgam, alpha=alpha, c=msub.logg)
    axes[1, 3].hist(msub_lgam.values, 35)
    
    plt.show()
    
    plt.hist(resids, 100)
    plt.show()
    
    tlgb = mdf.groupby(["teff", "logg"])
    def ep_plot(df, diagnostic_prob=0.05):
        diagnostic_plot = np.random.random() < diagnostic_prob
        eps = df["ep"].values
        lgam = np.log10(df["gamma"]).values
        mask = np.ones(len(lgam), dtype=bool)
        for i in range(5):
            fit_mat = np.vander(eps, 2)
            coeffs = np.linalg.lstsq(fit_mat[mask], lgam[mask])[0]
            modgam = np.dot(fit_mat, coeffs)
            resids = lgam - modgam
            abs_resids = np.abs(resids)
            med_resid = np.median(abs_resids)
            mask = abs_resids < 2.5*med_resid

            if diagnostic_plot:
                xv = np.linspace(np.min(eps), np.max(eps), 100)
                plt.plot(xv, np.dot(np.vander(xv, 2), coeffs))
                plt.scatter(eps, lgam, c=np.log10(df["tew"]), s=80, alpha=0.8)#, s=200.0*(np.log10(df["teff"])-3.45).values, alpha=0.6)
        if diagnostic_plot:
            cbar = plt.colorbar()
            cbar.set_label("log(Thermalized Width)")
            plt.xlabel("E.P.", fontsize=16)
            plt.ylabel("log(Gamma)", fontsize=16)
            plt.show()
        return coeffs
    
    gamcf = tlgb.apply(ep_plot)
    
    def gamma_model(params, transition):
        cur_gam = fit_params[0]
        coeff_idx = 1
        for cname, power in gamma_coeff_interpretations[1:]:
            try:
                cvals = params[cname]
            except KeyError:
                cvals = transition[cname]
            cur_gam += fit_params[coeff_idx]*cvals**power
            coeff_idx += 1
        modgam = cur_gam
        return modgam
    
    def saturation_model(params, transition):
        return params.teff*-8.117e-5 - 0.0864
    
    gcog_fname = args.output + ".gcog.hdf5"
    
    import pdb; pdb.set_trace()
    
    transition_index = 0
    for fname in pgrid_files:
        fpath = os.path.join(grid_dir, fname)
        ew_params = pd.read_hdf(fpath, "ews")
        
        transition = pd.read_hdf(fpath, "transition")
        
        tlgb = ew_params.groupby(["teff", "logg"])
        groups = tlgb.groups
        
        fc_accum = []
        fc_teff = []
        fc_logg = []
        for tl_tup in groups:
            show_diagnostic = np.random.random() < 0.00
            fc_teff.append(tl_tup[0])
            fc_logg.append(tl_tup[1])
            fcres = fit_cog(
                ew_params.ix[groups[tl_tup]],
                transition,
                gamma_model=gamma_model,
                saturation_model=saturation_model,
                diagnostic_plot=show_diagnostic,
                tl_tup=tl_tup
            )
            print(fcres)
            fc_accum.append(fcres)
        
        fc_accum = np.array(fc_accum)
        x_shifts = [fc[2] for fc in fc_accum]
