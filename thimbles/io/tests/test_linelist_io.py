import unittest
from thimbles.linelists import LineList
from thimbles.io.linelist_io import read_linelist
import os

class LineListReadingTester(unittest.TestCase):
    
    def setUp(self):
        self.ll_path = os.path.join(os.path.dirname(__file__), "dummy_line_lists")
    def test_ir_w_comment(self):
        fpath = os.path.join(self.ll_path, "ir_w_comment.ln") 
        result = read_linelist(fpath)
        self.assertTrue(len(result) == 11)
        self.assertTrue(len(result["wv"]) == 11)
        self.assertAlmostEqual(result["species"].iloc[0], 26.0)
        self.assertAlmostEqual(result["species"].iloc[-1], 60.1)
    
    def test_single_line(self):
        fpath = os.path.join(self.ll_path, "single_line.ln") 
        result = read_linelist(fpath)
        self.assertTrue(len(result) == 1)
        self.assertTrue(isinstance(result, LineList))

if __name__ == "__main__":
    unittest.main()
