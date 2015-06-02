import unittest
from thimbles.io.linelist_io import read_linelist
import os

class LineListReadingTester(unittest.TestCase):
    ll_path = os.path.join(os.path.dirname(__file__), "data")    
    
    def setUp(self):
        pass
    
    def test_ir_w_comment(self):
        fpath = os.path.join(self.ll_path, "ir_w_comment.ln") 
        result = read_linelist(fpath)
        self.assertTrue(len(result) == 11)
        self.assertAlmostEqual(result[0].ion.z, 26.0)
        self.assertAlmostEqual(result[0].ion.charge, 0)
        self.assertAlmostEqual(result[-1].ion.z, 60.0)
        self.assertAlmostEqual(result[-1].ion.charge, 1)
    
    def test_single_line(self):
        fpath = os.path.join(self.ll_path, "single_line.ln") 
        result = read_linelist(fpath)
        self.assertTrue(len(result) == 1)
    
    def test_read_vald(self):
        fpath = os.path.join(self.ll_path, "short_example.vald")
        result = read_linelist(fpath)
        zvals = [65, 26, 22, 23, 23, 26, 59, 26, 26, 26, 26, 28, 106, 106, 68]
        ll_z = [l.ion.z for l in result]
        assert len(ll_z) == len(zvals)
        same_z = [l1==l2 for l1, l2 in zip(zvals, ll_z)]
        assert all(same_z)

if __name__ == "__main__":
    unittest.main()
