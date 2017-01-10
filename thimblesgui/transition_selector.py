
from thimblesgui import QtGui, QtWidgets, QtCore, Qt
Signal = QtCore.Signal
Slot = QtCore.Slot
QModelIndex = QtCore.QModelIndex

from thimbles.abundances import Ion
from thimbles import ptable
from thimbles.periodictable import symbol_to_z, z_to_symbol
from thimblesgui.expressions import PythonExpressionLineEdit
from thimblesgui.wavelength_span import FlatWavelengthSpanWidget
from thimblesgui.column_sets import full_transition_columns, base_transition_columns
from thimblesgui.active_collections import ActiveCollectionView
from thimbles.transitions import Transition
from thimbles.abundances import Ion

def _to_z(val):
    try:
        z = int(val)
    except ValueError:
        z=symbol_to_z(val)
    return z

_parse_error_style = """
    background-color: rgb(255, 175, 175);
"""

_parse_success_style = """
    background-color: rgb(255, 255, 255);
"""

class SpeciesSelectorWidget(QtWidgets.QWidget):
    speciesChanged = Signal(list)
    
    def __init__(self, zvals=None, parent=None):
        super(SpeciesSelectorWidget, self).__init__(parent)
        if zvals is None:
            zvals = []
        self._z_set = set(zvals)
        
        layout = QtWidgets.QHBoxLayout()
        self.setLayout(layout)
        self.label = QtWidgets.QLabel("Species")
        layout.addWidget(self.label)
        self.species_le = QtWidgets.QLineEdit(parent=self)
        layout.addWidget(self.species_le)
        #self.species_le.setFixedWidth(150)
        self.parse_btn = QtWidgets.QPushButton("parse")
        layout.addWidget(self.parse_btn)
        
        self.parse_btn.clicked.connect(self.parse_text)
        self.species_le.editingFinished.connect(self.parse_text)
    
    def keyPressEvent(self, event):
        ekey = event.key()
        print("key event {}".format(ekey))
        if (ekey == Qt.Key_Enter) or (ekey == Qt.Key_Return):
            #self.on_set()
            return
        super(SpeciesSelectorWidget, self).keyPressEvent(event)
    
    def parse_text(self):
        sp_strs = str(self.species_le.text()).split()
        z_vals = []
        for sp in sp_strs:
            spl = [_to_z(s) for s in sp.split("-")]
            if any([s is None for s in spl]):
                self.species_le.setStyleSheet(_parse_error_style)
                return
            if len(spl) == 1:
                z_vals.extend(spl)
            elif len(spl) == 2:
                z1, z2 = sorted(spl)
                z_vals.extend(list(range(z1, z2)))
        self.zvals = z_vals
        print("zvals", self.zvals)
        self.species_le.setStyleSheet(_parse_success_style)
    
    @property
    def zvals(self):
        return list(self._z_set)
    
    @zvals.setter
    def zvals(self, value):
        self._z_set = set(value)
        self.speciesChanged.emit(self.zvals)


class TransitionConstraintsWidget(QtWidgets.QWidget):
    constraintsChanged = Signal(list)
    
    def __init__(self, wv_span, parent=None):
        super().__init__(parent)
        
        layout = QtWidgets.QGridLayout()
        self.setLayout(layout)
        self.wv_span = wv_span
        self.wv_span_cb = QtWidgets.QCheckBox()
        self.wv_span_cb.setCheckState(Qt.CheckState(2))
        layout.addWidget(self.wv_span_cb, 0, 0, 1, 1)
        self.wv_span_widget = FlatWavelengthSpanWidget(wv_span, with_steppers=False, parent=self)
        layout.addWidget(self.wv_span_widget, 0, 1, 1, 1)
        self.species_filter_cb = QtWidgets.QCheckBox()
        self.species_filter_cb.setCheckState(Qt.CheckState(2))
        self.species_selector = SpeciesSelectorWidget(parent=self)
        layout.addWidget(self.species_filter_cb, 1, 0, 1, 1)
        layout.addWidget(self.species_selector, 1, 1, 1, 1)
        self.expr_filter_cb = QtWidgets.QCheckBox()
        self.expr_filter_cb.setCheckState(Qt.CheckState(2))
        self.expr_wid = PythonExpressionLineEdit(parent=self, field_label="filter", expression="None")
        layout.addWidget(self.expr_filter_cb, 2, 0, 1, 1)
        layout.addWidget(self.expr_wid, 2, 1, 1, 1)
        
        self.wv_span.boundsChanged.connect(self.emit_constraints)
        self.wv_span_cb.toggled.connect(self.emit_constraints)
        self.species_filter_cb.toggled.connect(self.emit_constraints)
        self.expr_filter_cb.toggled.connect(self.emit_constraints)
        #self.expr_wid.expressionChanged.connect(self.on_expression_changed)
        self.species_selector.speciesChanged.connect(self.on_species_changed)
    
    def emit_constraints(self, bounds=None):
        print("emitting constraints")
        tc = self.transition_constraints()
        self.constraintsChanged.emit(tc)
    
    def on_species_changed(self, new_species):
        if self.species_filter_cb.checkState():
            self.emit_constraints()
    
    def on_expression_changed(self):
        if self.expr_filter_cb.checkState():
            self.emit_constraints()
    
    def transition_constraints(self):
        constraints = []
        if self.wv_span_cb.checkState():
            min_wv, max_wv = self.wv_span.bounds
            constraints.append(Transition.wv >= min_wv)
            constraints.append(Transition.wv <= max_wv)
        if self.species_filter_cb.checkState():
            zvals = self.species_selector.zvals
            if len(zvals) == 1:
                constraints.append(Ion.z == zvals[0])
            elif len(zvals) > 1:
                constraints.append(Ion.z.in_(self.species_selector.zvals))
        if self.expr_filter_cb.checkState():
            filter_val = self.expr_wid.value
            if not filter_val is None:
                constraints.append(filter_val)
        return constraints


class TransitionSelectorWidget(QtWidgets.QWidget):
    
    def __init__(
            self,
            db,
            wv_span,
            collection,
            selection,
            parent
    ):
        super().__init__(parent=parent)
        self.db = db
        layout = QtWidgets.QVBoxLayout()
        self.constraints_widget = TransitionConstraintsWidget(
            wv_span=wv_span,
            parent=self,
        )
        layout.addWidget(self.constraints_widget)
        self.collection = collection
        self.constraints_widget.constraintsChanged.connect(self.on_constraints_changed)
        self.transition_list_view = ActiveCollectionView(
            active_collection=collection,
            columns = base_transition_columns,
            selection=selection,
            selection_channel = "transition",
        )
        layout.addWidget(self.transition_list_view)
        self.setLayout(layout)
    
    @Slot(list)
    def on_constraints_changed(self, constraint_list):
        query = self.db.query(Transition).join(Ion)
        for constraint in constraint_list:
            if not constraint is None:
                query = query.filter(constraint)
        transitions = query.all()
        self.collection.set(transitions)
    
        
            
