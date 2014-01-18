import h5py
import numpy as np
import thimbles as tmb
import matplotlib.pyplot as plt

tpath = tmb.__path__[0]
hf = h5py.File("%s/resources/cog_ppol.h5" % tpath)

coeffs, knots, centers, scales = np.array(hf["coefficients"]), np.array(hf["knots"]), np.array(hf["centers"]), np.array(hf["scales"])
iqp = tmb.utils.piecewise_polynomial.InvertiblePiecewiseQuadratic(coeffs, knots, centers=centers, scales=scales)

x = np.linspace(-10, -2, 500)
y = iqp(x)
#iqp.branch_sign = np.ones(len(centers), dtype=float)
yinv = iqp.inverse(y)

plt.plot(x, y)
plt.plot(x, yinv)
plt.show()
