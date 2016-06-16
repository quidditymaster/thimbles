
import numpy as np
import scipy
import matplotlib as mpl
from matplotlib.collections import LineCollection

from thimblesgui import QtCore, QtGui, Qt
from thimblesgui.mplwidget import MatplotlibWidget
from thimblesgui.prevnext import PrevNext
from thimblesgui.selection_charts import TransitionMarkerChart
from thimblesgui.active_collections import ItemMappedColumn

import thimbles as tmb

ItemMappedColumn

class NormalizationEditor(QtGui.QDialog):
    
    def __init__(
            self,
            spectrum,
            parent,
    ):
        super().__init__(parent=parent)
        layout = QtGui.QVBoxLayout()
        
        self.spectrum = spectrum
        self.setWindowTitle("Norm Editor")
        
        self.control_widget = QtGui.QWidget(parent=self)
        controls = QtGui.QGridLayout()
        revert_btn = QtGui.QPushButton("Revert")
        revert_btn.clicked.connect(self.on_revert)
        controls.addWidget(revert_btn, 1, 0, 1, 1)
        cancel_btn = QtGui.QPushButton("Cancel")
        cancel_btn.clicked.connect(self.on_cancel)
        controls.addWidget(cancel_btn, 1, 2, 1, 1)
        fit_btn = QtGui.QPushButton("Fit")
        fit_btn.clicked.connect(self.on_fit)
        controls.addWidget(fit_btn, 1, 1, 1, 1)
        ok_btn = QtGui.QPushButton("OK")
        ok_btn.clicked.connect(self.on_ok)
        controls.addWidget(ok_btn, 1, 3, 1, 1)
        
        self.control_widget.setLayout(controls)
        layout.addWidget(self.control_widget)
        
        self.flux_plot = MatplotlibWidget(
            nrows=1,
            parent=self,
            mpl_toolbar=True,
        )
        self.flux_plot.buttonPressed.connect(self.on_plot_click)
        layout.addWidget(self.flux_plot)
        ax = self.flux_plot.ax
        
        self.data_line = ax.plot(
            spectrum.wvs,
            spectrum.flux,
            color="b",
            lw=1.5,
            alpha=0.8,
        )[0]
        norm_p = self.spectrum["norm"]
        self.norm_line = ax.plot(
            spectrum.wvs,
            norm_p.value,
            color="k",
            lw=3.0,
            alpha=0.8
        )[0]
        self.setLayout(layout)
    
    def on_revert(self):
        pass
    
    def on_ok(self):
        print("ok!")
    
    def on_cancel(self):
        pass
    
    def on_fit(self):
        print("fit yo!")
    
    def on_plot_click(self, event_list):
        event ,= event_list
        xp, yp = event.xdata, event.ydata
        print(xp, yp)
        modifiers = QtGui.QApplication.keyboardModifiers()
        if modifiers == QtCore.Qt.ShiftModifier:
            print("shift click!")
        
