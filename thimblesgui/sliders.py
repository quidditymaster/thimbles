import threading
import numpy as np
import matplotlib

from PySide import QtCore
from PySide import QtGui
from PySide.QtCore import Signal, Slot
Qt = QtCore.Qt

class FloatSlider(QtGui.QWidget):
    valueChanged = Signal(float)
    
    def __init__(
            self, 
            name, 
            hard_min, 
            hard_max, 
            n_steps=127, 
            orientation=Qt.Vertical, 
            format_str="{:5.3f}", 
            parent=None
    ):
        super(FloatSlider, self).__init__(parent)
        label = QtGui.QLabel(name, parent=self)
        if orientation == Qt.Horizontal:
            lay = QtGui.QHBoxLayout()
        elif orientation == Qt.Vertical:
            lay = QtGui.QVBoxLayout()
        else:
            raise NotImplementedError
        self.hard_min = hard_min
        self.hard_max = hard_max
        self.c_min = hard_min
        self.c_max = hard_max
        self.n_steps = n_steps
        self.calculate_delta()
        self.slider = QtGui.QSlider(orientation, self)
        self.format_str = format_str
        self.value_indicator = QtGui.QLabel()
        self.refresh_indicator()
        self.min_lineedit = QtGui.QLineEdit()
        min_valid = QtGui.QDoubleValidator(self.hard_min, self.hard_max, 4, self.min_lineedit)
        self.min_lineedit.setValidator(min_valid)
        self.min_lineedit.setText("{:10.3g}".format(self.c_min))
        self.max_lineedit = QtGui.QLineEdit()
        max_valid = QtGui.QDoubleValidator(self.hard_min, self.hard_max, 4, self.max_lineedit)
        self.max_lineedit.setValidator(max_valid)
        self.max_lineedit.setText("{:10.3g}".format(self.c_max))
        self.slider.setRange(0, n_steps)
        lay.addWidget(label)
        lay.addWidget(self.value_indicator)
        lay.addWidget(self.max_lineedit)
        lay.addWidget(self.slider)
        lay.addWidget(self.min_lineedit)
        self.min_lineedit.setFixedWidth(50)
        self.max_lineedit.setFixedWidth(50)
        self.setLayout(lay)
        
        #connect up the line edit events
        self.min_lineedit.textChanged.connect(self.on_min_input)
        self.max_lineedit.textChanged.connect(self.on_max_input)
        self.slider.valueChanged.connect(self._value_changed)
        #self.slider.valueChanged.connect(self.refresh_indicator)
    
    def _value_changed(self, value):
        self.refresh_indicator()
        self.valueChanged.emit(self.value())
    
    def refresh_indicator(self):
        self.value_indicator.setText(self.format_str.format(self.value()))
    
    def on_min_input(self):
        self.set_min(self.min_lineedit.text())
    
    def on_max_input(self):
        self.set_max(self.max_lineedit.text())
    
    def calculate_delta(self):
        self.delta = float(self.c_max-self.c_min)
    
    def set_min(self, min):
        cur_value = self.value() 
        self.c_min = float(min)
        self.calculate_delta()
        self.set_value(cur_value)
    
    def set_max(self, max):
        cur_value = self.value()
        self.c_max = float(max)
        self.calculate_delta()
        self.set_value(cur_value)
    
    def value(self):
        return self.c_min + self.slider.value()*(self.delta/self.n_steps)
    
    def set_value(self, val):
        try:
            if val > self.c_max:
                print("value above slider max, truncating")
                val = self.c_max
            elif val < self.c_min:
                print("value below slider min, truncating")
                val = self.c_min
            elif val == np.nan:
                print("value is nan, using minimum")
                val = self.c_min
            vfrac = (val-self.c_min)/self.delta
            opt_idx = int(np.around(vfrac*self.n_steps))
            self.slider.setValue(opt_idx)
            self.refresh_indicator()
        except Exception as e:
            print("failed slider value setting resulted in error %s" % e)


class SliderCascade(QtGui.QWidget):
    slidersChanged = Signal(int)
    
    def __init__(self, sliders, parent=None):
        super(SliderCascade, self).__init__(parent)
        layout = QtGui.QHBoxLayout()
        self.sliders = sliders
        for slider in sliders:
            layout.addWidget(slider)
        self.setLayout(layout)
    
    def _connect_sliders(self):
        for slider in self.sliders:
            slider.valueChanged.connect(self.slidersChanged.emit)


if __name__ == "__main__":
    qap = QtGui.QApplication([])
    slider = FloatSlider("test", 0, 10)
    slider.show()
    qap.exec_()
