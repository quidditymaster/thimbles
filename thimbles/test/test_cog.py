import h5py
import numpy as np
import thimbles as tmb
from thimbles.utils  import piecewise_polynomial
import matplotlib.pyplot as plt

tpath = tmb.__path__[0]

def test_cogh5file ():
    cog_ppol_hf = h5py.File("%s/resources/cog_ppol.h5" % tpath)
    
    coeffs, knots, centers, scales = np.array(cog_ppol_hf["coefficients"]), np.array(cog_ppol_hf["knots"]), np.array(cog_ppol_hf["centers"]), np.array(cog_ppol_hf["scales"])
    iqp = piecewise_polynomial.InvertiblePiecewiseQuadratic(coeffs, knots, centers=centers, scales=scales)
    
    x = np.linspace(-10, -2, 500)
    y = iqp(x)-7.5
    #iqp.branch_sign = np.ones(len(centers), dtype=float)
    yinv = iqp.inverse(y)
    
    plt.plot(x, y)
    plt.plot(x, yinv)
    plt.show()
    
if __name__ == "__main__":
    test_cogh5file()
