
import numpy as np
import thimblesgui as tmbg

from thimblesgui import QtGui, QtCore, Qt
Signal = QtCore.Signal
Slot = QtCore.Slot
QModelIndex = QtCore.QModelIndex

class WavelengthSpan(QtCore.QObject):
    boundsChanged = Signal(list)
    
    def __init__(self, bounds):
        super(WavelengthSpan, self).__init__()
        self.bounds = bounds
    
    def set_bounds(self, bounds):
        self.bounds = bounds
        self.boundsChanged.emit(self.bounds)
    
    def center_on(self, center):
        log_delta = 0.5*np.log(self.bounds[1]/self.bounds[0])
        new_lb = center*np.exp(-log_delta)
        new_ub = center*np.exp(log_delta)
        self.set_bounds([new_lb, new_ub])


class SpanWidgetBase(object):
    
    def on_step_edit(self):
        self.step_frac = float(self.step_le.text())
    
    def refresh_bounds_text(self):
        clb, cub = self.wv_span.bounds
        self.min_wv_le.setText(self.wv_fmt.format(clb))
        self.max_wv_le.setText(self.wv_fmt.format(cub))
    
    def on_min_edit(self):
        clb, cub = self.wv_span.bounds
        new_bounds = [float(self.min_wv_le.text()), cub]
        self.wv_span.set_bounds(new_bounds)
    
    def on_max_edit(self):
        clb, cub = self.wv_span.bounds
        new_bounds = [clb, float(self.max_wv_le.text())]
        self.wv_span.set_bounds(new_bounds)
    
    def step(self, forward=True):
        clb, cub = self.wv_span.bounds
        if forward:
            delta_frac=np.power(cub/clb, self.step_frac)
        else:
            delta_frac= np.power(clb/cub, self.step_frac)
        new_min = clb*delta_frac
        new_max = cub*delta_frac
        self.wv_span.set_bounds([new_min, new_max])
    
    def step_forward(self):
        self.step(forward=True)
    
    def step_back(self):
        self.step(forward=False)


class FlatWavelengthSpanWidget(SpanWidgetBase, QtGui.QWidget):
    
    def __init__(
            self, 
            wv_span, 
            step_frac=0.5, 
            wv_fmt="{:7.2f}", 
            with_steppers=True, 
            parent=None
    ):
        super(FlatWavelengthSpanWidget, self).__init__(parent)
        self.wv_span = wv_span
        self.step_frac=step_frac #delta_log_wv = log(max_wv/min_wv)*step_frac
        self.wv_fmt=wv_fmt
        
        layout = QtGui.QHBoxLayout()
        self.setLayout(layout)
        if with_steppers:
            self.backward_btn = QtGui.QPushButton("<<")
            layout.addWidget(self.backward_btn)
        layout.addWidget(QtGui.QLabel("min wv"))
        self.min_wv_le = QtGui.QLineEdit()
        #import pdb; pdb.set_trace()
        self.min_wv_le.setFixedWidth(90)
        min_valid = QtGui.QDoubleValidator(0.0, 1e5, 5, self.min_wv_le)
        self.min_wv_le.setValidator(min_valid)
        layout.addWidget(self.min_wv_le)
        if with_steppers:
            layout.addWidget(QtGui.QWidget())
            self.step_le = QtGui.QLineEdit()
            step_valid = QtGui.QDoubleValidator(0.0, 1.0, 3, self.step_le)
            self.step_le.setValidator(step_valid)
            self.step_le.setText("{:03.2f}".format(self.step_frac))
            self.step_le.setFixedWidth(35)
            layout.addWidget(self.step_le)
        layout.addWidget(QtGui.QLabel("max wv"))
        self.max_wv_le = QtGui.QLineEdit()
        self.max_wv_le.setFixedWidth(90)
        max_valid = QtGui.QDoubleValidator(0.0, 1e5, 5, self.max_wv_le)
        self.max_wv_le.setValidator(max_valid)
        layout.addWidget(self.max_wv_le)
        if with_steppers:
            self.forward_btn = QtGui.QPushButton(">>")
            layout.addWidget(self.forward_btn)
        
        self.refresh_bounds_text() 
        
        #connect
        self.min_wv_le.editingFinished.connect(self.on_min_edit)
        self.max_wv_le.editingFinished.connect(self.on_max_edit)
        self.wv_span.boundsChanged.connect(self.refresh_bounds_text)
        if with_steppers:
            self.step_le.editingFinished.connect(self.on_step_edit)
            self.backward_btn.clicked.connect(self.step_back)
            self.forward_btn.clicked.connect(self.step_forward)
    
    def keyPressEvent(self, event):
        ekey = event.key()
        print(ekey)
        if (ekey == Qt.Key_Enter) or (ekey == Qt.Key_Return):
            #self.on_set()
            return
        super(FlatWavelengthSpanWidget, self).keyPressEvent(event)

class WavelengthSpanWidget(SpanWidgetBase, QtGui.QWidget):
    
    def __init__(self, wv_span, step_frac=0.5, wv_fmt="{:7.2f}", parent=None):
        super(WavelengthSpanWidget, self).__init__(parent)
        self.wv_span = wv_span
        self.step_frac=step_frac #delta_log_wv = log(max_wv/min_wv)*step_frac
        self.wv_fmt=wv_fmt
        
        layout = QtGui.QGridLayout()
        self.setLayout(layout)
        #label row
        layout.addWidget(QtGui.QLabel("  Min Wv"), 0, 0, 1, 1)
        layout.addWidget(QtGui.QLabel("   Step   "), 0, 1, 1, 1)
        layout.addWidget(QtGui.QLabel("  Max Wv"), 0, 2, 1, 1)
        #btn row
        self.backward_btn = QtGui.QPushButton("<<")
        layout.addWidget(self.backward_btn, 1, 0, 1, 1)
        self.step_le = QtGui.QLineEdit()
        step_valid = QtGui.QDoubleValidator(0.0, 1.0, 3, self.step_le)
        self.step_le.setValidator(step_valid)
        self.step_le.setText("{:03.2f}".format(self.step_frac))
        layout.addWidget(self.step_le)
        self.forward_btn = QtGui.QPushButton(">>")
        layout.addWidget(self.forward_btn, 1, 2, 1, 1)
        #wv bounds row
        self.min_wv_le = QtGui.QLineEdit()
        min_valid = QtGui.QDoubleValidator(0.0, 1e5, 5, self.min_wv_le)
        self.min_wv_le.setValidator(min_valid)
        layout.addWidget(self.min_wv_le, 2, 0, 1, 1)
        layout.addWidget(QtGui.QLabel("< Wavelength <"), 2, 1, 1, 1)
        self.max_wv_le = QtGui.QLineEdit()
        max_valid = QtGui.QDoubleValidator(0.0, 1e5, 5, self.max_wv_le)
        self.max_wv_le.setValidator(max_valid)
        layout.addWidget(self.max_wv_le, 2, 2, 1, 1)
        self.refresh_bounds_text()        
        
        #connect
        self.min_wv_le.editingFinished.connect(self.on_min_edit)
        self.max_wv_le.editingFinished.connect(self.on_max_edit)
        self.wv_span.boundsChanged.connect(self.refresh_bounds_text)
        self.step_le.editingFinished.connect(self.on_step_edit)
        self.backward_btn.clicked.connect(self.step_back)
        self.forward_btn.clicked.connect(self.step_forward)
    
    def keyPressEvent(self, event):
        ekey = event.key()
        print(ekey)
        if (ekey == Qt.Key_Enter) or (ekey == Qt.Key_Return):
            #self.on_set()
            return
        super(WavelengthSpanWidget, self).keyPressEvent(event)
