import os
import numpy as np
from scipy.interpolate import LinearNDInterpolator as NDInterp
import sys
import h5py
import matplotlib.pyplot as plt

from thimbles import resource_dir
marcs_model_dir = os.path.join(resource_dir)  #TODO: allow a user option for setting the location of these files

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


class MarcsInterpolator(PhotosphereEngine):
    """interpolates the grid of MARCS model atmospheres"""
    
    def __init__(self):
        pass
    
    def _not_implemented(self):
        raise NotImplementedError("Not implemented for this engine type")
    
    def make_photosphere(self, fname, stellar_params):
        pass
    
