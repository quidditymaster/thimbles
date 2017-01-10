
from thimblesgui import QtGui, QtWidgets, QtCore, Qt

import numpy as np

import thimblesgui as tmbg
import thimbles as tmb
import thimbles.charts as charts
from thimbles.charts import SpectrumChart
from thimblesgui import MatplotlibWidget


class FluxDisplay(QtWidgets.QWidget):
    _span_connected = False
    
    def __init__(self, wv_span=None, parent=None):
        super(FluxDisplay, self).__init__(parent)
        self.plot_widget = MatplotlibWidget(
            parent=self,
            nrows=1,
        )
        self.ax = self.plot_widget.ax
        self.ax._tmb_redraw = True
        self.ax.set_xlabel("Wavelength")
        self.ax.set_ylabel("Flux")
        self.ax.get_xaxis().get_major_formatter().set_useOffset(False)
        self.charts = []
        
        if not wv_span is None:
            self.set_bounds(wv_span.bounds)
            self.connect_span(wv_span)
        
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)
    
    def add_chart(self, chart):
        self.charts.append(chart)
    
    def connect_span(self, wv_span):
        if self._span_connected:
            raise Exception("one span already connected disconnect from other span first")
        wv_span.boundsChanged.connect(self.set_bounds)
        self._span_connected = True
    
    @QtCore.Slot(list)
    def set_bounds(self, bounds):
        print("calling set_bounds in flux display")
        print("bounds", bounds)
        self.ax.set_xlim(*bounds)
        for schart in self.charts:
            schart.set_bounds(bounds)
        self.ax._tmb_redraw=True
    
    
