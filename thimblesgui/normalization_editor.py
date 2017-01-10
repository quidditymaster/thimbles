
import numpy as np
import scipy
import matplotlib as mpl
from matplotlib.collections import LineCollection

from thimblesgui import QtCore, QtGui, QtWidgets, Qt
from thimblesgui.mplwidget import MatplotlibWidget
from thimblesgui.prevnext import PrevNext
from thimblesgui.selection_charts import TransitionMarkerChart
from thimblesgui.active_collections import ItemMappedColumn

import thimbles as tmb

class NormHint(object):
    
    def __init__(
            self,
            x,
            y,
            sig_x,
            sig_y
    ):
        self.x = x
        self.y = y
        self.sig_x = sig_x
        self.sig_y = sig_y


class NormHintsEditor(QtWidgets.QWidget):
    
    def __init__(
            self,
            hint_array,
            hint_selection_channel,
            parent
    ):
        super().__init__(parent=parent)
        n_hints = len(hint_array)
        hints = []
        for i in range(n_hints):
            hints.append(NormHint(*hint_array[i]))
        
        self.hint_collection = ActiveCollection("hints", )
        self.hint_channel = hint_selection_channel


class NormalizationEditor(QtWidgets.QDialog):
    
    def __init__(
            self,
            spectrum,
            parent,
    ):
        super().__init__(parent=parent)
        layout = QtWidgets.QGridLayout()
        
        self.spectrum = spectrum
        self.setWindowTitle("Norm Editor")
        
        self.control_widget = QtWidgets.QWidget(parent=self)
        self.finalization_widget = QtWidgets.QWidget(parent=self)
        controls = QtWidgets.QGridLayout()
        finalization = QtWidgets.QHBoxLayout()
        self.control_widget.setLayout(controls)
        self.finalization_widget.setLayout(finalization)
        
        revert_btn = QtWidgets.QPushButton("Revert")
        revert_btn.clicked.connect(self.on_revert)
        finalization.addWidget(revert_btn)
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(self.on_cancel)
        finalization.addWidget(cancel_btn)
        ok_btn = QtWidgets.QPushButton("OK")
        ok_btn.clicked.connect(self.on_ok)
        finalization.addWidget(ok_btn)
        
        fit_btn = QtWidgets.QPushButton("Fit")
        fit_btn.clicked.connect(self.on_fit)
        controls.addWidget(fit_btn, 1, 1, 1, 1)
        
        self.flux_plot = MatplotlibWidget(
            nrows=1,
            parent=self,
            mpl_toolbar=True,
        )
        self.flux_plot.buttonPressed.connect(self.on_plot_click)
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
        
        layout.addWidget(self.control_widget, 0, 1)
        layout.addWidget(self.finalization_widget, 1, 0, 1, 2)
        layout.addWidget(self.flux_plot, 0, 0, 1, 1)
        self.setLayout(layout)
    
    def on_revert(self):
        pass
    
    def on_ok(self):
        self.accept()
    
    def on_cancel(self):
        self.reject()
    
    def on_fit(self):
        print("fit yo!")
    
    def on_plot_click(self, event_list):
        event ,= event_list
        xp, yp = event.xdata, event.ydata
        print(xp, yp)
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        if modifiers == QtCore.Qt.ShiftModifier:
            print("shift click!")

