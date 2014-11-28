import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.rcParams["image.cmap"] = "winter"

class EWMeasurementComparisonChart(object):

    def __init__(self, meas1, meas2, axes=None):
        if axes is None:
            fig, axes = plt.subplots(2, 2)
        self.axes = axes
    
    def _init_plots(self):
        self.axes[0, 0]
    
    def update(self):
        pass
    
