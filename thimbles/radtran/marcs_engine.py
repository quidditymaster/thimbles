import os
import numpy as np
from scipy.interpolate import LinearNDInterpolator as NDInterp
import sys
import h5py
import matplotlib.pyplot as plt

from thimbles import resource_dir
marcs_dir = os.path.join(resource_dir, "marcs") 

from thimbles.radtran.engines import PhotosphereEngine


def write_moog_marcs_file(layer_data, out_fname, vmicro, abundances, comment_string = None):
    outf = open(out_fname, "wb")
    outf.write("WEBMARCS\n")
    if comment_string == None:
        outf.write("MARCS: \n")
    else:
        outf.write(comment_string)
    outf.write(10*" "+str(len(layer_data))+ "\n")
    outf.write("5000\n")
    for i in range(len(layer_data)):
        laystr = "%3.0f" % i
        cline = (8*" %10G") % tuple(layer_data[i])
        outf.write(laystr+cline + "\n")
    outf.write(" "*5 + "%6E" % fvturb + "\n")
    outf.write("Natoms       0    %4.2f" % infeh + "\n")
    outf.write("Nmol         0 \n")
    outf.flush()
    outf.close()

marcs_grids = None

def _load_marcs():
    global marcs_grids
    if marcs_grids is None:
        grids = {sphericity_type:{} for sphericity_type in ["pp","sp"]}
        #plane parallel models
        grids["pp"]["alpha+"] = h5py.File(os.path.join(marcs_dir, "marcs_pp_alpha_enhanced.h5"))
        grids["pp"]["alpha-"] = h5py.File(os.path.join(marcs_dir, "marcs_pp_alpha_negative.h5"))
        grids["pp"]["alpha0"] = h5py.File(os.path.join(marcs_dir, "marcs_pp_alpha_solar.h5"))
        #sperical models
        grids["sp"]["alpha+"] = h5py.File(os.path.join(marcs_dir, "marcs_sp_alpha_enhanced.h5"))
        grids["sp"]["alpha-"] = h5py.File(os.path.join(marcs_dir, "marcs_sp_alpha_negative.h5"))
        grids["sp"]["alpha0"] = h5py.File(os.path.join(marcs_dir, "marcs_sp_alpha_solar.h5"))
        marcs_grids = grids
    return marcs_grids





class MarcsInterpolator(PhotosphereEngine):
    """interpolates the grid of MARCS model atmospheres"""
    
    def __init__(self, alpha="alpha0"):
        self.alpha_grid = alpha
        #self.grids = _load_marcs()
    
    def _not_implemented(self):
        raise NotImplementedError("Not implemented for this engine type")
    
    def get_layer_data(self, stellar_params, alpha=None):
        teff = stellar_params.teff
        logg = stellar_params.logg
        metalicity = stellar_params.metalicity
        if logg >= 3.0:
            sphericity = "pp"
        else:
            sphericity = "sp"
        if alpha is None:
            alpha = self.alpha_grid
        cgrid = self.grids[sphericity][alpha]
        if teff < 4000:
            teff_tol = 101
        else:
            teff_tol = 300
        if metalicity > -1:
            feh_tol = 0.3
        elif metalicity > -2:
            feh_tol = 0.6
        else:
            feh_tol = 1.1
    
    def make_photosphere(self, fname, stellar_params):
        pass
    
