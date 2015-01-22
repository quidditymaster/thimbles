
import numpy as np
import scipy

def max_norm(spectrum):
    return np.max(spectrum.flux)

def fractional_norm(spectrum, frac=0.98):
    npts = len(spectrum)
    return np.sort(spectrum)[min(int(npts*frac), npts-1)]

def iterative_sorting_norm(spectrum, init_min=0.1, init_max=0.98, degree=4):
    x = spectrum.wv/spectrum.wv[-1]
    y = spectrum.flux
    pfit = np.polyfit(x, y, degree=degree)
    
