import unittest
import thimbles as tmb
from thimbles.options import Option, opts

class testoption (unittest.TestCase):
    
    def setUp(self):
        unittest.TestCase.setUp(self)
    
    def test_create_option(self):
        Option("whatsinaname", default=3)
        self.assertEqual(3, opts["whatsinaname"])
    
    def test_parent_dict(self):
        Option("parent", option_style="parent_dict")
        Option("child", parent="parent", default=21)
        self.assertTrue(opts["parent.child"] == 21)
        self.assertTrue(opts["parent"]["child"] == 21)


if __name__ == "__main__":
    unittest.main()
