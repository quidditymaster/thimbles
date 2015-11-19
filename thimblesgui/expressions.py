
from thimblesgui import QtCore, QtGui, Qt
import thimbles as tmb

_parse_error_style = """
    background-color: rgb(255, 175, 175);
"""

_parse_success_style = """
    background-color: rgb(240, 240, 240);
"""

class PythonExpressionLineEdit(QtGui.QWidget):
    _global_namespace = tmb.wds.__dict__
    _is_valid = False
    _value = None
    
    def __init__(self, field_label, expression, parent=None):
        super().__init__(parent=parent)
        layout = QtGui.QHBoxLayout()
        self.field_label = QtGui.QLabel(field_label)
        layout.addWidget(self.field_label)
        self.ledit = QtGui.QLineEdit(expression, parent=self)
        layout.addWidget(self.ledit)
        self.parse_btn = QtGui.QPushButton("parse", parent=self)
        layout.addWidget(self.parse_btn)
        self.parse_btn.clicked.connect(self.on_parse)
        
        self.error_label = QtGui.QLabel("")
        layout.addWidget(self.error_label)
        
        self.parse_btn.clicked.connect(self.on_parse)
        self.ledit.editingFinished.connect(self.on_parse)
        self.setLayout(layout)
    
    def keyPressEvent(self, event):
        ekey = event.key()
        #print("key event {}".format(ekey))
        if (ekey == Qt.Key_Enter) or (ekey == Qt.Key_Return):
            #self.on_set()
            return
        super().keyPressEvent(event)
    
    def set_valid(self):
        self._is_valid = True
        self.error_label.setText("")
        self.ledit.setStyleSheet(_parse_success_style)
    
    def set_invalid(self, error):
        self._is_valid = False
        self.error_label.setText(str(error))
        self.ledit.setStyleSheet(_parse_error_style)
    
    @property
    def value(self):
        if not self._is_valid:
            self.on_parse()
        return self._value
    
    def on_parse(self):
        try:
            self._value = eval(self.ledit.text(), self._global_namespace)
            self.set_valid()
        except Exception as err:
            self.set_invalid(err)





