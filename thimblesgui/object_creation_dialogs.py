from thimblesgui import QtGui, Qt
import thimbles as tmb
#import inspect

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

class NewObjectDialog(QtGui.QDialog):
    obj = None
    
    def __init__(
            self,
            fields,
            factory,
            parent=None
    ):
        super().__init__(parent)
        self.setWindowTitle("Create Object")
        self.factory = factory
        self.fields = fields
        layout = QtGui.QVBoxLayout()
        
        #self.kwarg_dict = {}
        self.expr_dict = {}
        for field_idx in range(len(fields)):
            field_name, default_expr = fields[field_idx]
            pele = PythonExpressionLineEdit(field_name, expression=default_expr)
            self.expr_dict[field_name] = pele
            layout.addWidget(pele)
        
        control_group = QtGui.QWidget()
        cg_lay = QtGui.QHBoxLayout()
        cg_lay.addWidget(QtGui.QWidget())
        control_group.setLayout(cg_lay)
        self.create_btn = QtGui.QPushButton("create")
        cg_lay.addWidget(self.create_btn)
        self.create_btn.clicked.connect(self.on_create)
        self.cancel_btn = QtGui.QPushButton("cancel")
        cg_lay.addWidget(self.cancel_btn)
        self.cancel_btn.clicked.connect(self.on_cancel)
        layout.addWidget(control_group)
        
        self.setLayout(layout)
    
    @classmethod
    def get_new(cls, parent):
        new_dialog = cls(parent=parent)
        new_dialog.exec_()
        return new_dialog.obj
    
    def on_create(self):
        kwargs = {}
        kwargs_complete = True
        for key in self.expr_dict:
            expr_wid = self.expr_dict[key]
            expr_val = expr_wid.value
            if expr_wid._is_valid:
                kwargs[key] = expr_val
            else:
                kwargs_complete = False
                break
        if kwargs_complete:
            self.obj = self.factory(**kwargs)
            self.accept()
    
    def on_cancel(self):
        self.reject()

class NewStarDialog(NewObjectDialog):
    
    def __init__(self, parent):
        super().__init__(
            fields=[
                ("name", "'star_name'"),
                ("ra", "None"),
                ("dec", "None"),
                ("teff", "5500"),
                ("logg", "3.0"),
                ("metalicity", "-0.5"),
                ("vmicro", "2.0"),
                ("vmacro", "1.0"),
                ("vsini", "5.0"),
                ("ldark", "0.6"),
                ("mass", "1.0"),
                ("age", "5.0"),
                ("info", "{}"),
            ],
            factory = tmb.star.Star,
            parent=parent
        )
