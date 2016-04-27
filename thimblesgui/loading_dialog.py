
from thimblesgui import QtGui, Qt
from thimblesgui.expressions import PythonExpressionLineEdit

import thimbles as tmb

class OpenFileWidget(QtGui.QWidget):
    
    def __init__(
            self,
            file_label="file",
            default_file="",
            parent=None,
    ):
        super().__init__(parent=parent)
        layout = QtGui.QHBoxLayout()
        self.setLayout(layout)
        self.file_label = QtGui.QLabel(file_label)
        layout.addWidget(self.file_label)
        self.file_le = QtGui.QLineEdit(default_file)
        layout.addWidget(self.file_le)
        browse_btn = QtGui.QPushButton("Browse")
        browse_btn.clicked.connect(self.on_browse)
        layout.addWidget(browse_btn)
    
    def on_browse(self):
        file_path = QtGui.QFileDialog.getOpenFileName(parent=self, caption="this is a caption")
        self.file_le.setText(file_path)
    
    @property
    def path(self):
        return self.file_le.text()


class LoadDialog(QtGui.QDialog):
    result = None
    
    def __init__(
            self,
            rfunc_expr="tmb.io.read_spec",
            parent=None
    ):
        super().__init__(parent=None)
        layout = QtGui.QVBoxLayout()
        self.setLayout(layout)
        self.read_func_expression = PythonExpressionLineEdit(
            field_label="read in function",
            expression=rfunc_expr,
            parent=self,
        )
        layout.addWidget(self.read_func_expression)
        
        self.file_path_widget = OpenFileWidget()
        layout.addWidget(self.file_path_widget)
        
        self.status_label = QtGui.QLabel("")
        layout.addWidget(self.status_label)
        
        control_group = QtGui.QWidget(parent=self)
        cglay = QtGui.QHBoxLayout()
        control_group.setLayout(cglay)
        
        spacer = QtGui.QWidget()
        cglay.addWidget(spacer)
        load_btn = QtGui.QPushButton("Load")
        load_btn.clicked.connect(self.on_load)
        cglay.addWidget(load_btn)
        layout.addWidget(control_group)
    
    def on_load(self):
        read_func = self.read_func_expression.value
        if self.read_func_expression._is_valid:
            try:
                fname = self.file_path_widget.path
                self.result = read_func(fname)
                self.accept()
            except Exception as e:
                self.status_label.setText(str(e))
