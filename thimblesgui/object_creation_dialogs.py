from thimblesgui import QtGui, Qt
from thimblesgui.expressions import PythonExpressionLineEdit
import thimbles as tmb
#import inspect


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
                ("name", '""'),
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

class NewApertureDialog(NewObjectDialog):
    
    def __init__(self, parent):
        super().__init__(
            fields=[
                ("name", '""'),
                ("info", "{}"),
            ],
            factory = tmb.spectrographs.Aperture,
            parent=parent
        )

class NewOrderDialog(NewObjectDialog):
    
    def __init__(self, parent):
        super().__init__(
            fields=[
                ("number", "0"),
            ],
            factory = tmb.spectrographs.Order,
            parent=parent
        )

class NewChipDialog(NewObjectDialog):
    
    def __init__(self, parent):
        super().__init__(
            fields=[
                ("name", '""'),
                ("info", "{}"),
            ],
            factory = tmb.spectrographs.Chip,
            parent=parent
        )

class NewExposureDialog(NewObjectDialog):
    
    def __init__(self, parent):
        super().__init__(
            fields=[
                ("name", '""'),
                ("time", "0.0"),
                ("duration", "0.0"),
                ("type", '"science"'),
                ("info", "{}"),
            ],
            factory = tmb.observations.Exposure,
            parent=parent
        )
