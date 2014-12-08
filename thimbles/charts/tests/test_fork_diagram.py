import unittest
import matplotlib.pyplot as plt
import numpy as np
import thimbles.charts as charts

class TestForkDiagram(unittest.TestCase):
    
    def setUp(self):
        pass

    def test_late_set(self):
        fd = charts.ForkDiagram(spread_height=0.1)
        fd.set_xvals(np.random.random(10))
        fd.set_depths(np.random.random(10))
        fd.update()
        fd.ax.set_xlim(0, 1)
        fd.ax.set_ylim(-0.1, 1.3)
        plt.show()

if __name__ == "__main__":
    unittest.main()
