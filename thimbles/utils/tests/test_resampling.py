import unittest
from thimbles.utils import resampling
import numpy as np
import scipy.stats
import time

class TestApproxCDF(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def test_cdf(self):
        zs = np.linspace(-8, 8, 1001)
        stime = time.time()
        approx_vals = resampling.approximate_normal_cdf(zs)
        etime = time.time()
        ap_time1 = etime-stime
        stime = time.time()
        approx_vals = resampling.approximate_normal_cdf(zs)
        etime = time.time()
        ap_time2 = etime-stime
        stime = time.time()
        scipy_vals= scipy.stats.norm.cdf(zs)
        etime = time.time()
        sp_time1 = etime-stime
        print("""
        approximate_cdf time (first execution) : {}
        approximate_cdf time (second execution): {} 
        scipy.stats.norm.cdf time                {}
        """.format(ap_time1, ap_time2, sp_time1))
        np.testing.assert_allclose(approx_vals, scipy_vals, atol=1e-5)

if __name__ == "__main__":
    unittest.main()
