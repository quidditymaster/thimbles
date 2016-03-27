
import numpy as np
import matplotlib as mpl
mpl.use("qt4Agg")
import matplotlib.pyplot as plt

import scipy
import scipy.integrate
import pandas as pd

import thimbles as tmb

fname = "trans_54.pgrid"

flux = pd.read_hdf(fname, "flux")
transition = pd.read_hdf(fname, "transition")
params = pd.read_hdf(fname, "params")
wvs = pd.read_hdf(fname, "wvs").values

def extract_ews(fluxes, wvs):
    return scipy.integrate.simps(1.0-fluxes, x=wvs, axis=1)

def get_doppler_widths(teffs, vmicros, transition):
    wv = transition["wv"]
    mol_weight = tmb.ptable.ix[(transition["z"], 0)]["weight"]
    twidth = tmb.utils.misc.thermal_width(teffs, wv, mol_weight)
    vwidth = (vmicros/tmb.speed_of_light)*wv
    return np.sqrt(vwidth**2 + twidth**2)


params["ew"] = extract_ews(flux, wvs)
params["doppler_width"] = get_doppler_widths(params["teff"], params["vmicro"], transition)
params["tew"] = params["ew"]/params["doppler_width"]
params["xonh"] = params["met"] + params["xonfe"]

def fit_gamma(
        fluxes,
        params,
        wvs,
        transition,
        frac_gamma_init=0.33,
        n_iter=50,
):
    depths = 1.0-fluxes
    
    sigma = np.mean(params["doppler_width"].values)
    gamma = frac_gamma_init * sigma
    for iter_idx in range(n_iter):
        prof = tmb.profiles.voigt(wvs, transition["wv"], sigma, gamma)
        gamma_eps = 0.1*gamma
        p_minus = tmb.profiles.voigt(wvs, transition["wv"], sigma, gamma-gamma_eps)
        p_plus = tmb.profiles.voigt(wvs, transition["wv"], sigma, gamma+gamma_eps)
        gamma_deriv = (p_plus - p_minus)/gamma_eps
        
        depth_model = prof.reshape((1, -1))*params["ew"].values.reshape((-1, 1))
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
        data_weight = len(fluxes)
        gamma *= data_weight/(data_weight + 0.01)
        gamma_ratio = gamma/sigma
        print("gamma_ratio", gamma_ratio)
    return gamma


def fit_cog(params, transition, gamma, diagnostic_plot=False):
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


#mask = (params.teff == 5500) & (params.logg == 2.0)

groups = params.groupby(["teff", "logg"]).groups
gammas = []
teffs = []
loggs = []
shifts = []
adjs = []

n_break = 10000
cur_p = 0
for group_idx in groups:
    cur_p += 1
    if cur_p > n_break:
        break
    teffs.append(group_idx[0])
    loggs.append(group_idx[1])
    mask = groups[group_idx]
    cgam = fit_gamma(
        flux.ix[mask],
        params.ix[mask],
        wvs,
        transition,
    )
    gammas.append(cgam)
    bshift, badj = fit_cog(params.ix[mask], transition, gamma=cgam)
    shifts.append(bshift)
    adjs.append(badj)

gammas = np.array(gammas)
teffs = np.array(teffs)
loggs = np.array(loggs)
shifts = np.array(shifts)
adjs = np.array(adjs)

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
