from copy import copy
import thimblesgui as tmbg

from PySide import QtGui, QtCore
Qt = QtCore.Qt
from PySide.QtCore import Signal, Slot
from PySide.QtCore import QModelIndex

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import thimbles as tmb
from thimbles import workingdataspace as wds
from thimbles.options import Option, opts
from thimbles.thimblesdb import Base
from thimbles.transitions import Transition, as_transition_group
from thimbles.transitions import TransitionGroupingStandard, TransitionGroup
from thimbles.abundances import Ion
from thimbles import as_wavelength_sample
from thimbles import ptable
from thimbles.periodictable import symbol_to_z, z_to_symbol
import thimbles.charts as charts
#from thimbles.charts import MatplotlibCanvas

from thimblesgui import MatplotlibWidget
from thimblesgui import FluxDisplay

class SelectionTier(QtCore.QObject):
    changed = Signal(list)
    focusChanged = Signal(list)
    
    def __init__(self, values=None):
        super(SelectionTier, self).__init__()
        if values is None:
            values = []
        self.set_values(values)
    
    def __len__(self):
        return len(self.values)
    
    def index(self, index):
        return self.indexes.get(index)
    
    def extend(self, obj_list):
        added = []
        for obj in obj_list:
            existing_idx = self.index(obj)
            if existing_idx is None:
                added.append(obj)
                self.indexes[obj] = len(self.indexes)
        if len(added) > 0:
            self.values.extend(added)
            self.changed.emit(self.values)
        return added
    
    def append(self, obj):
        self.extend([obj])
    
    def set_focus(self, focus):
        if isinstance(focus[0], int):
            self.focus = focus
        else:
            focus = [self.index(fc) for fc in focus]
            focus = sorted([fc for fc in focus if not (fc is None)])
        if not (len(focus) == len(self.focus)):
            self.focus = focus
            self.focusChanged.emit(self.focus)
        elif any([f1!=f2 for f1, f2 in zip(focus, self.focus)]):
            self.focus = focus
            self.focusChanged.emit(self.focus)
    
    def set_values(self, values):
        self.indexes = {}
        self.focus = []
        for idx in range(len(values)):
            self.indexes[values[idx]] = idx
        self.values = values
        self.changed.emit(self.values)
    
    def clear(self):
        self.indexes = {}
        self.values = []
        self.changed.emit(self.values)


class TieredSelection(object):
    """A a wrapper for managing multiple simultaneous levels of selection.
    """
    
    def __init__(self):
        self.background = SelectionTier()
        self.foreground = SelectionTier()
        self.active = SelectionTier()


class GroupingEditorSelection(object):
    
    def __init__(self):
        self.transitions = TieredSelection()
        self.groups = TieredSelection()


class WavelengthSpan(QtCore.QObject):
    boundsChanged = Signal(list)
    
    def __init__(self, min_wv, max_wv):
        super(WavelengthSpan, self).__init__()
        self._min_wv = min_wv
        self._max_wv = max_wv
    
    def emit_bounds_changed(self):
        print "bounds changed! {} < {}".format(self.min_wv, self.max_wv)
        self.boundsChanged.emit([self.min_wv, self.max_wv])
    
    @property
    def min_wv(self):
        return self._min_wv
    
    @min_wv.setter
    def min_wv(self, value):
        self._min_wv = value
        self.emit_bounds_changed()
    
    @property
    def max_wv(self):
        return self._max_wv
    
    @max_wv.setter
    def max_wv(self, value):
        self._max_wv = value
        self.emit_bounds_changed()
    
    @property
    def bounds(self):
        return [self._min_wv, self._max_wv]
    
    @bounds.setter
    def bounds(self, value):
        self.set_bounds(*value)
    
    def set_bounds(self, min_wv, max_wv):
        self._min_wv = min_wv
        self._max_wv = max_wv
        self.emit_bounds_changed()        

class SpanWidgetBase(object):
    def keyPressEvent(self, event):
        ekey = event.key()
        print ekey
        if (ekey == Qt.Key_Enter) or (ekey == Qt.Key_Return):
            #self.on_set()
            return
        super(WavelengthSpanWidget, self).keyPressEvent(event)
    
    def on_step_edit(self):
        self.step_frac = float(self.step_le.text())
    
    def refresh_bounds_text(self):
        self.min_wv_le.setText(self.wv_fmt.format(self.wv_span.min_wv))
        self.max_wv_le.setText(self.wv_fmt.format(self.wv_span.max_wv))
    
    def on_min_edit(self):
        self.wv_span.min_wv = float(self.min_wv_le.text())
    
    def on_max_edit(self):
        self.wv_span.max_wv = float(self.max_wv_le.text())
    
    def step_forward(self):
        cmin = self.wv_span.min_wv
        cmax = self.wv_span.max_wv
        delta_frac=np.power(cmax/cmin, self.step_frac)
        new_min = cmin*delta_frac
        new_max = cmax*delta_frac
        self.wv_span.set_bounds(new_min, new_max)
    
    def step_back(self):
        cmin = self.wv_span.min_wv
        cmax = self.wv_span.max_wv
        delta_frac=np.power(cmin/cmax, self.step_frac)
        new_min = cmin*delta_frac
        new_max = cmax*delta_frac
        self.wv_span.set_bounds(new_min, new_max)

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
    background-color: rgb(240, 240, 240);
"""

class SpeciesSelectorWidget(QtGui.QWidget):
    speciesChanged = Signal(list)
    
    def __init__(self, zvals=None, parent=None):
        super(SpeciesSelectorWidget, self).__init__(parent)
        if zvals is None:
            zvals = []
        self._z_set = set(zvals)
        
        layout = QtGui.QHBoxLayout()
        self.setLayout(layout)
        self.label = QtGui.QLabel("species")
        layout.addWidget(self.label)
        self.species_le = QtGui.QLineEdit(parent=self)
        layout.addWidget(self.species_le)
        #self.species_le.setFixedWidth(150)
        self.parse_btn = QtGui.QPushButton("parse")
        layout.addWidget(self.parse_btn)
        
        self.parse_btn.clicked.connect(self.parse_text)
        self.species_le.editingFinished.connect(self.parse_text)
    
    def keyPressEvent(self, event):
        ekey = event.key()
        print "key event {}".format(ekey)
        if (ekey == Qt.Key_Enter) or (ekey == Qt.Key_Return):
            #self.on_set()
            return
        super(SpeciesSelectorWidget, self).keyPressEvent(event)
    
    def parse_text(self):
        sp_strs = self.species_le.text().split()
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
                z_vals.extend(range(z1, z2))
        self.zvals = z_vals
        print "zvals", self.zvals
        self.species_le.setStyleSheet(_parse_success_style)
    
    @property
    def zvals(self):
        return list(self._z_set)
    
    @zvals.setter
    def zvals(self, value):
        self._z_set = set(value)
        self.speciesChanged.emit(self.zvals)


class BaseExpressionWidget(QtGui.QWidget):
    expression = None
    expressionChanged = Signal()
    
    def __init__(self, sqla_base=None, label="base expression", parent=None):
        super(BaseExpressionWidget, self).__init__(parent)
        if sqla_base is None:
            sqla_base = Base
        self.sqla_base = sqla_base
        
        layout = QtGui.QHBoxLayout()
        self.setLayout(layout)
        #self.text_box = QtGui.QPlainTextEdit()
        self.label = QtGui.QLabel(label)
        layout.addWidget(self.label)
        self.expression_le = QtGui.QLineEdit(parent=self)
        layout.addWidget(self.expression_le)
        self.parse_btn = QtGui.QPushButton("parse")
        layout.addWidget(self.parse_btn)
        
        self.expression_le.editingFinished.connect(self.parse_text)
        self.parse_btn.clicked.connect(self.parse_text)
    
    def parse_text(self):
        text = self.expression_le.text().strip()
        if text == "":
            self.expression = None
            self.expression_le.setStyleSheet(_parse_success_style)
            self.expressionChanged.emit()
        else:
            try:
                text.replace("\n", " ")
                expr = eval(text, {}, self.sqla_base._decl_class_registry)
                self.expression = expr
                self.expression_le.setStyleSheet(_parse_success_style)
                self.expressionChanged.emit()
            except Exception as e:
                print e
                self.expression_le.setStyleSheet(_parse_error_style)
    
    def keyPressEvent(self, event):
        ekey = event.key()
        print "key event in arbfilt {}".format(ekey)
        if (ekey == Qt.Key_Enter) or (ekey == Qt.Key_Return):
            return
        super(BaseExpressionWidget, self).keyPressEvent(event)

class TransitionConstraints(QtGui.QWidget):
    constraintsChanged = Signal(list)
    
    def __init__(self, wv_span, parent=None):
        super(TransitionConstraints, self).__init__(parent)
        
        layout = QtGui.QGridLayout()
        self.setLayout(layout)
        self.wv_span = wv_span
        self.wv_span_cb = QtGui.QCheckBox()
        self.wv_span_cb.setCheckState(Qt.CheckState.Checked)
        layout.addWidget(self.wv_span_cb, 0, 0, 1, 1)
        self.wv_span_widget = FlatWavelengthSpanWidget(wv_span, with_steppers=False, parent=self)
        layout.addWidget(self.wv_span_widget, 0, 1, 1, 1)
        self.species_filter_cb = QtGui.QCheckBox()
        self.species_filter_cb.setCheckState(Qt.CheckState.Checked)
        self.species_selector = SpeciesSelectorWidget(parent=self)
        layout.addWidget(self.species_filter_cb, 1, 0, 1, 1)
        layout.addWidget(self.species_selector, 1, 1, 1, 1)
        self.expr_filter_cb = QtGui.QCheckBox()
        self.expr_filter_cb.setCheckState(Qt.CheckState.Checked)
        self.expr_wid = BaseExpressionWidget(parent=self, label="filter")
        layout.addWidget(self.expr_filter_cb, 2, 0, 1, 1)
        layout.addWidget(self.expr_wid, 2, 1, 1, 1)
        
        self.wv_span.boundsChanged.connect(self.emit_constraints)
        self.species_filter_cb.toggled.connect(self.emit_constraints)
        self.expr_filter_cb.toggled.connect(self.emit_constraints)
        self.species_selector.speciesChanged.connect(self.on_species_changed)
        self.expr_wid.expressionChanged.connect(self.on_expression_changed)
    
    def emit_constraints(self, bounds=None):
        print "emitting constraints"
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
        constraints.append(Transition.wv >= self.wv_span.min_wv)
        constraints.append(Transition.wv <= self.wv_span.max_wv)
        if self.species_filter_cb.checkState():
            zvals = self.species_selector.zvals
            if len(zvals) == 1:
                constraints.append(Ion.z == zvals[0])
            elif len(zvals) > 1:
                constraints.append(Ion.z.in_(self.species_selector.zvals))
        if self.expr_filter_cb.checkState():
            expr = self.expr_wid.expression
            if not expr is None:
                constraints.append(expr)
        return constraints


class TransitionExpressionWidget(QtGui.QWidget):
    xExpressionChanged = Signal(str)
    yExpressionChanged = Signal(str)
    
    def __init__(
            self, 
            x_expression,
            y_expression,
            success_style=None,
            error_style=None,
            parent=None,
    ):
        super(TransitionExpressionWidget, self).__init__(parent)
        self.x_expression = x_expression
        self.y_expression = y_expression
        if success_style is None:
            success_style = _parse_success_style
        self.success_style = success_style
        if error_style is None:
            error_style   = _parse_error_style
        self.error_style = error_style
        self.mplwid = MatplotlibWidget(parent=self)
        self.ax = self.mplwid.ax
        self.ax._tmb_redraw = True
        layout = QtGui.QGridLayout()
        self.setLayout(layout)
        layout.addWidget(self.mplwid, 0, 0, 2, 2)
        #x and y expression line edits
        layout.addWidget(QtGui.QLabel("X"), 1, 0, 1, 1)
        self.x_le = QtGui.QLineEdit()
        self.x_le.setText(self.x_expression)
        self.ax.set_xlabel(self.x_expression)
        layout.addWidget(self.x_le, 1, 1, 1, 1)
        layout.addWidget(QtGui.QLabel("Y"), 2, 0, 1, 1)
        self.y_le = QtGui.QLineEdit()
        self.y_le.setText(self.y_expression)
        self.ax.set_ylabel(self.y_expression)
        layout.addWidget(self.y_le, 2, 1, 1, 1)
        
        self.dummy_trans = Transition(500.0, (26, 1), 1.0, -1.0)
        
        self.x_le.editingFinished.connect(self.on_x_changed)
        self.y_le.editingFinished.connect(self.on_y_changed)
    
    def minimumSizeHint(self):
        return QtCore.QSize(200, 250)
    
    def on_x_changed(self):
        x_text = self.x_le.text()
        good_expr = self.check_expression(x_text)
        if good_expr:
            self.x_le.setStyleSheet(self.success_style)
            self.ax.set_xlabel(self.x_expression)
            self.xExpressionChanged.emit(x_text)
        else:
            self.x_le.setStyleSheet(self.error_style)
    
    def on_y_changed(self):
        y_text = self.y_le.text()
        good_expr = self.check_expression(y_text)
        if good_expr:
            self.y_le.setStyleSheet(self.success_style)
            self.ax.set_ylabel(self.y_expression)
            self.yExpressionChanged.emit(y_text)
        else:
            self.y_le.setStyleSheet(self.error_style)
    
    def check_expression(self, expression):
        good_expr = True
        try:
            res = [eval(expression) for t in [copy(self.dummy_trans)]]
            try:
                float(res[0])
            except TypeError:
                good_expr = False
        except Exception as e:
            good_expr = False
        return good_expr


class TransitionGroupBurstPlot(object):
    _plot_initialized = False
    
    def __init__(
            self,
            x_expression,
            y_expression,
            ax=None,
            **mplkwargs
    ):
        if ax is None:
            fig, ax = plt.subplots()
        self.ax = ax
    
    @Slot(list)
    def set_groups(self, glist):
        self.groups = glist
        self.update()
    
    def update(self):
        if not self._plot_initialized:
            self.bursts = mpl.collections.LineCollection
    
    def _line_burst_data(self, x, y, center_x=None, center_y=None):
        if center_x is None:
            center_x = np.nanmean(x)
        if center_y is None:
            center_y = np.nanmean(y)
        burst_data = np.zeros((len(x), 2, 2))
        #start each line at the center
        burst_data[:, 0, 0] = center_x
        burst_data[:, 0, 1] = center_y
        #then move to the position of the transition
        burst_data[:, 1, 0] = x
        burst_data[:, 1, 1] = y
        return burst_data
    
    def update_group(self):
        group = self.group
        gtrans = group.transitions
        grp_color = "#99ff33"
        grp_x, grp_y = self.resolve_xy(gtrans)
        grp_x_center, grp_y_center = np.nanmean(grp_x), np.nanmean(grp_y)
        burst_data = self._line_burst_data(grp_x, grp_y, grp_x_center, grp_y_center)
        if not self._group_initialized:
            self.group_burst = mpl.collections.LineCollection(burst_data, color=grp_color, linewidth=3.0, zorder=1) #not pickable
            self.ax.add_artist(self.group_burst)
            self.group_dot ,= self.ax.plot([grp_x_center], [grp_y_center], picker=6, color=grp_color, markersize=10, zorder=1)
            self._group_initialized = True
        else:
            self.group_burst.set_segments(burst_data)
            self.group_dot.set_data([grp_x_center], [grp_y_center])
        self.group_dot._md = dict(name="group", group=group)


class TransitionScatterPlot(object):
    _plot_initialized = False
    
    def __init__(
            self,
            transitions=None,
            x_expression="t.pseudo_strength()", 
            y_expression="t.ep",
            picker=None,
            auto_zoom=False,
            ax=None,
            **mplkwargs
    ):
        self.x_expression=x_expression
        self.y_expression=y_expression
        self.picker = picker
        self.auto_zoom = auto_zoom
        mplkwargs.setdefault("marker", "o")
        self.mplkwargs = mplkwargs
        if ax is None:
            fig, ax = plt.subplots()
        self.ax = ax
        self.set_transitions(transitions)
    
    @Slot(list)
    def set_transitions(self, tlist):
        if tlist is None:
            tlist = []
        self.transitions = tlist
        self.update()
    
    @Slot(str)
    def set_x_expression(self, expr_text):
        self.x_expression = expr_text
        self.ax.set_xlabel(self.x_expression)
        self.update()
    
    @Slot(str)
    def set_y_expression(self, expr_text):
        self.y_expression = expr_text
        self.ax.set_ylabel(self.y_expression)
        self.update()
    
    def resolve_expression(self, transition_list, expression):
        try:
            return [eval(expression) for t in transition_list]
        except Exception as e:
            return None
    
    def resolve_xy(self, transition_list):
        xvals = self.resolve_expression(transition_list, self.x_expression)
        yvals = self.resolve_expression(transition_list, self.y_expression)
        return xvals, yvals
    
    def update(self):
        transitions = self.transitions
        if len(transitions) == 0:
            if self._plot_initialized:
                self.scatter.set_visible(False)
        else:
            xvals, yvals = self.resolve_xy(transitions)
            if not self._plot_initialized:
                self.scatter ,= self.ax.plot(xvals, yvals, picker=self.picker, linestyle="none", **self.mplkwargs)
                self._plot_initialized =True
            else:
                self.scatter.set_data(xvals, yvals)
            self.scatter._md = dict(kind="transitions", transitions=transitions)
            self.scatter.set_visible(True)
            if self.auto_zoom:
                self.zoom_to_data()
        self.ax._tmb_redraw = True
    
    def zoom_to_data(self, x_pad=0.1, y_pad=0.1):
        xpts, ypts = self.scatter.get_data()
        xmin, xmax = np.nanmin(xpts), np.nanmax(xpts)
        ymin, ymax = np.nanmin(ypts), np.nanmax(ypts)
        dx = xmax-xmin
        dy = ymax-ymin
        self.ax.set_xlim(xmin-x_pad*dx, xmax+x_pad*dx)
        self.ax.set_ylim(ymin-y_pad*dy, ymax+y_pad*dy)

class ListMappedColumn(object):
    
    def __init__(
            self,
            header,
            getter,
            setter=None,
            value_converter=None,
            string_converter=None,
            qt_flag=None,
    ):
        self.header = header
        self.getter = getter
        self.setter = setter
        if value_converter is None:
            value_converter = lambda x: "{:10.3f}".format(x)
        self.value_converter = value_converter
        if string_converter is None:
            string_converter = float
        self.string_converter = string_converter
        if qt_flag is None:
            qt_flag = Qt.ItemIsSelectable | Qt.ItemIsEnabled
            if not setter is None:
                qt_flag |= Qt.ItemIsEditable
        self.qt_flag = qt_flag
    
    def get(self, data_obj, role):
        if role == Qt.DisplayRole:
            return self.value_converter(self.getter(data_obj))
    
    def set(self, data_obj, value, role):
        if role == Qt.EditRole:
            self.setter(data_obj, self.string_converter(value))
    

class MappedListModel(QtCore.QAbstractTableModel):
    
    def __init__(self, mapped_list, columns):
        super(MappedListModel, self).__init__()
        self._data = mapped_list
        self.column_map = columns
    
    @Slot(list)
    def set_mapped_list(self, value):
        self.beginResetModel()
        self._data = value
        self.endResetModel()
    
    def rowCount(self, parent=QModelIndex()):
        return len(self._data)
    
    def columnCount(self, parent=QModelIndex()):
        return len(self.column_map)
    
    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                self.column_map[section].header
    
    def flags(self, index):
        col = index.column()
        return self.column_map[col].qt_flag
    
    def data(self, index, role=Qt.DisplayRole):
        row, col = index.row(), index.column()
        data_obj = self._data[row]
        col_obj = self.column_map[col]
        return col_obj.get(data_obj, role)
    
    def setData(self, index, value, role=Qt.EditRole):
        row, col = index.row(), index.column()
        data_obj = self._data[row]
        col_obj = self.column_map[col]
        try:
            col_obj.set(data_obj, value, role)
            return True
        except Exception as e:
            print e
            return False

class TransitionListModel(MappedListModel):
    
    def __init__(self, transitions):
        wv = ListMappedColumn("Wavelength", lambda x: x.wv)
        #z = ListMappedColumn("Z", lambda x: x.ion.z)
        #charge = ListMappedColumn("Ion Charge", lambda x: x.ion.charge)
        symbol = ListMappedColumn("Symbol", lambda x: "{} {}".format(x.ion.symbol, "I"*(x.ion.charge+1)), value_converter=lambda x: x)
        ep = ListMappedColumn("Excitation\nPotential", lambda x: x.ep)
        loggf = ListMappedColumn("log(gf)", lambda x: x.loggf)
        cols = [wv, symbol, ep, loggf]
        super(TransitionListModel, self).__init__(transitions, cols)


class TransitionGroupListModel(MappedListModel):
    
    def __init__(self, groups):
        symbol = ListMappedColumn("Symbol", lambda x: "{} {}".format(x.transitions[0].ion.symbol, "I"*(x.transitions[0].ion.charge+1)), value_converter=lambda x: x)
        minwv = ListMappedColumn("Min Wavelength", lambda x: x.aggregate("wv", np.min))
        #wvsig = ListMappedColumn("Wavelength\nSpread", lambda x: x.aggregate("wv", np.std))
        maxwv = ListMappedColumn("Max Wavelength", lambda x: x.aggregate("wv", np.max))
        n= ListMappedColumn("N", lambda x: len(x), value_converter=lambda x: "{}".format(x))
        epmean = ListMappedColumn("mean E.P.", lambda x: x.aggregate("ep", np.mean))        
        epsig = ListMappedColumn("sigma E.P.", lambda x: x.aggregate("ep", np.std)) 
        xmean = ListMappedColumn("mean pseudo-strength", lambda x: x.aggregate("x", np.mean))        
        xsig = ListMappedColumn("sigma pseudo-strength", lambda x: x.aggregate("wv", np.mean))               
        cols = [symbol, n, minwv, maxwv, epmean, epsig, xmean, xsig]
        super(TransitionGroupListModel, self).__init__(groups, cols)


class SelectionListView(QtGui.QTabWidget):
    
    def __init__(
            self, 
            tiered_selection, 
            list_model_class,
            tiers="background foreground active",
            parent=None
    ):
        self.selection = tiered_selection
        super(SelectionListView, self).__init__(parent)
        
        tiers = tiers.split()
        self.tab_dict = {}
        self.tmods = {}
        self.tviews = {}
        for tier in tiers:
            qw = QtGui.QWidget(parent=self)
            layout = QtGui.QHBoxLayout()
            qw.setLayout(layout)
            tbv = QtGui.QTableView(parent=qw)
            layout.addWidget(tbv)
            self.tviews[tier] = tbv
            tier_obj = getattr(tiered_selection, tier)
            cmod = list_model_class(tier_obj.values)
            tier_obj.changed.connect(cmod.set_mapped_list)
            tbv.setModel(cmod)
            self.tmods[tier] = cmod
            self.tab_dict[tier] = qw
            self.addTab(qw, tier)


class BackgroundTransitionListWidget(QtGui.QWidget):
    
    def __init__(self, selection, parent=None):
        super(BackgroundTransitionListWidget, self).__init__(parent)
        layout = QtGui.QGridLayout()
        self.setLayout(layout)
        
        bk_tier = self.selection.transitions.background
        self.table_model = TransitionListModel(bk_tier.values)

        self.wv_span = WavelengthSpan(3700, 3705)
        self.constraints = TransitionConstraints(self.wv_span)
        self.table_view = QtGui.QTableView(parent=self)
        layout.addWidget(self.table_view, 1, 0, 1, 3)

class ActiveTransitionListWidget(QtGui.QWidget):
    
    def __init__(self, selection, parent=None):
        self.selection = selection
        super(ActiveTransitionListWidget, self).__init__(parent)
        layout = QtGui.QGridLayout()
        self.setLayout(layout)
        
        active_trans = self.selection.transitions.active.values
        self.table_model = TransitionListModel(active_trans)
        self.table_view = QtGui.QTableView(parent=self)
        self.table_view.setModel(self.table_model)
        layout.addWidget(self.table_view, 0, 0, 1, 3)
        self.clear_btn = QtGui.QPushButton("Clear")
        self.clear_btn.clicked.connect(self.on_clear)
        layout.addWidget(self.clear_btn, 1, 0, 1, 1)
        self.inject_btn = QtGui.QPushButton("inject")
        self.inject_btn.clicked.connect(on_inject)
        layout.addWidget(self.inject_btn, 1, 2, 1, 1)
    
    def on_clear(self):
        self.selection.transitions.active.clear()
    
    def on_inject(self):
        print "inject1!one!"
    

class TransitionSelectionView(QtGui.QTabWidget):
    
    def __init__(
            self, 
            transition_selection, 
            parent=None
    ):
        self.selection = transition_selection
        super(SelectionListView, self).__init__(parent)
        
        self.bk_tab = QtGui.QWidget(parent=self)
        self.bk_mod = TransitionListModel
        self.bk_view = None

mplframerate = Option("mplframerate", default=10, parent="GUI")

class GroupingStandardEditor(QtGui.QMainWindow):
    _redraw_all = False
    
    def __init__(
            self, 
            standard_name, 
            tdb,
            spectra=None,
            parent=None,
    ):
        super(GroupingStandardEditor, self).__init__(parent)
        self.selection = GroupingEditorSelection()
        
        if spectra is None:
            spectra = []
        self.spectra = spectra
        
        self.tdb = tdb
        gstand = tdb.query(tmb.transitions.TransitionGroupingStandard)\
                    .filter(TransitionGroupingStandard.name==standard_name).one()
        
        self.grouping_standard = gstand
        self.grouping_dict = {}
        for group in self.grouping_standard.groups:
            for trans in group.transitions:
                self.grouping_dict[trans] = group
        
        blue_trans = tdb.query(Transition).order_by(Transition.wv).first()
        self.wv_span = WavelengthSpan(blue_trans.wv, blue_trans.wv*(1.0015))
        background_transitions = tdb.query(Transition)\
                .filter(Transition.wv >= self.wv_span.min_wv-0.1)\
                .filter(Transition.wv <= self.wv_span.max_wv+0.1)\
                .all()
        foreground_transitions = background_transitions #should we apply a filter?
        
        for trans in foreground_transitions:
            cgroup = self.grouping_dict.get(trans)
            if not cgroup is None:
                if len(cgroup) > 1:
                    break
        
        self.selection.transitions.background.set_values(background_transitions)
        self.selection.transitions.foreground.set_values(foreground_transitions)
        self.selection.groups.active.set_values([cgroup])
        
        self.make_actions()
        self.make_menus()
        self.make_toolbars()
        self.make_status_bar()
        self.mpl_displays = []
        
        self.flux_display = FluxDisplay(self.wv_span, parent=self)
        self.mpl_displays.append(self.flux_display)
        self.flux_display.ax.set_ylim(0, 1.4)
        self.flux_display.plot_widget.pickEvent.connect(self.on_pick_event)
        self.setCentralWidget(self.flux_display)
        chart_kwargs = {}
        chart_kwargs["ax"] = self.flux_display.ax
        chart_kwargs["label_axes"] = False
        chart_kwargs["auto_zoom"] = False
        for spec in self.spectra:
            schart = charts.SpectrumChart(spec, **chart_kwargs)
            self.flux_display.add_chart(schart)
        
        lmin = -1.5
        lmax = 5.0
        self.background_tines = tmb.charts.fork_diagram.TransitionsChart(
            self.selection.transitions.background.values,
            lmin=lmin,
            lmax=lmax,
            color="#aaaaaa",
            ax=self.flux_display.ax,
        )
        self.flux_display.add_chart(self.background_tines)
        self.selection.transitions.background.changed.connect(self.background_tines.set_transitions)
        self.foreground_tines = tmb.charts.fork_diagram.TransitionsChart(
            self.selection.transitions.foreground.values,
            #grouping_dict=self.grouping_dict,
            color="#0a0f15",
            tine_picker=6,
            lmin=lmin,
            lmax=lmax,
            linewidth=1.5,
            ax=self.flux_display.ax,
        )
        self.flux_display.add_chart(self.foreground_tines)
        self.selection.transitions.foreground.changed.connect(self.foreground_tines.set_transitions)
        self.active_fork_diagram = tmb.charts.fork_diagram.TransitionsChart(
            self.selection.transitions.active.values,
            grouping_dict=self.grouping_dict,
            color="#ea7355",
            lmin=lmin,
            lmax=lmax,
            linewidth=3.0,
            ax=self.flux_display.ax,
        )
        self.flux_display.add_chart(self.active_fork_diagram)
        self.selection.transitions.active.changed.connect(self.on_active_transitions_changed)
        
        dock = QtGui.QDockWidget("Transiton Constraints", self)
        self.constraints = TransitionConstraints(self.wv_span, parent=dock)
        self.constraints.constraintsChanged.connect(self.on_constraints_changed)
        dock.setWidget(self.constraints)
        self.addDockWidget(Qt.TopDockWidgetArea, dock)
        
        dock = QtGui.QDockWidget("Transition Scatter", self)
        x_exp = "t.pseudo_strength()"
        y_exp = "t.ep"
        self.scatter_display = TransitionExpressionWidget(
            x_expression=x_exp,
            y_expression=y_exp,
            parent=dock)
        self.mpl_displays.append(self.scatter_display)
        self.background_scatter = TransitionScatterPlot(
            self.selection.transitions.background.values, 
            ax=self.scatter_display.ax, 
            markersize=3, 
            color="#555555",
            auto_zoom=True,
            alpha=0.6
        )
        self.selection.transitions.background.changed.connect(
            self.background_scatter.set_transitions)
        self.scatter_display.xExpressionChanged.connect(
            self.background_scatter.set_x_expression)
        self.scatter_display.yExpressionChanged.connect(
            self.background_scatter.set_y_expression)
        self.foreground_scatter = TransitionScatterPlot(
            self.selection.transitions.foreground.values, 
            ax=self.scatter_display.ax, 
            picker=6, markersize=6,
            auto_zoom=False,
            color="#ff2000"
        )
        self.scatter_display.mplwid.pickEvent.connect(self.on_pick_event)
        self.selection.transitions.foreground.changed.connect(
            self.foreground_scatter.set_transitions)
        self.scatter_display.xExpressionChanged.connect(
            self.foreground_scatter.set_x_expression)
        self.scatter_display.yExpressionChanged.connect(
            self.foreground_scatter.set_y_expression)
        dock.setWidget(self.scatter_display)
        self.addDockWidget(Qt.TopDockWidgetArea, dock)
        
        dock = QtGui.QDockWidget("Transitions")
        self.transition_list_view = SelectionListView(
            self.selection.transitions,
            list_model_class=TransitionListModel,
            tiers="background foreground active",
            parent=dock
        )
        self.transition_list_view.setCurrentIndex(2)
        dock.setWidget(self.transition_list_view)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        
        dock = QtGui.QDockWidget("Groups")
        self.group_list_view = SelectionListView(
            self.selection.groups,
            list_model_class = TransitionGroupListModel,
            tiers="background foreground active",
            parent=dock,
        )
        dock.setWidget(self.group_list_view)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        self.draw_timer = QtCore.QTimer(self)
        self.draw_timer.start(1000.0/mplframerate.value)
        self.draw_timer.timeout.connect(self.on_draw_timeout)
    
    def on_draw_timeout(self):
        for display in self.mpl_displays:
            if self._redraw_all or display.ax._tmb_redraw:
                display.ax._tmb_redraw = False
                display.ax.figure.canvas.draw()
        self._redraw_all = False
    
    def make_actions(self):
        self.save_act = QtGui.QAction("Save changes to database", self)
        save_seq = QtGui.QKeySequence(Qt.Key_Control + Qt.Key_X, Qt.Key_Control + Qt.Key_S)
        self.save_act.setShortcut(save_seq)
        self.save_act.triggered.connect(self.on_save)
        
        self.nextwv_act = QtGui.QAction("next wv region", self)
        self.nextwv_act.setStatusTip("move wavelength bounds forward a step")
        self.nextwv_act.setShortcut(QtGui.QKeySequence(Qt.Key_Control+Qt.Key_Right))
        self.nextwv_act.triggered.connect(self.next_wv_region)
        
        self.prevwv_act = QtGui.QAction("prev wv region", self)
        self.prevwv_act.setStatusTip("move wavelength bounds back a step")
        self.prevwv_act.setShortcut(QtGui.QKeySequence(Qt.Key_Control+Qt.Key_Left))
        self.prevwv_act.triggered.connect(self.prev_wv_region)
    
    def make_menus(self):
        self.fileMenu = self.menuBar().addMenu("&File")
        self.fileMenu.addAction(self.save_act)
    
    def make_toolbars(self):
        self.wavelength_toolbar = self.addToolBar("Wavelength")
        self.global_wv_span_widget = FlatWavelengthSpanWidget(self.wv_span)
        self.wavelength_toolbar.addWidget(self.global_wv_span_widget)
        self.wavelength_toolbar.addSeparator()
    
    def make_status_bar(self):
        self.statusBar().showMessage("Ready")
    
    @Slot(list)
    def on_pick_event(self, event_l):
        event ,= event_l
        if hasattr(event.artist, "_md"):
            metdat = event.artist._md
            if metdat["kind"] == "transitions":
                transitions = event.artist._md["transitions"]
                trans = transitions[event.ind[0]]
                group = self.grouping_dict.get(trans)
                if not group is None:
                    trans_to_add = group.transitions
                else:
                    trans_to_add = [trans]
                start_idx = len(self.selection.transitions.active)
                added = self.selection.transitions.active.extend(trans_to_add)
                if len(added) >= 1:
                    end_idx = start_idx+len(added)-1
                    self.transition_list_view.tmods["active"].beginInsertRows(QModelIndex(), start_idx, end_idx)
                    self.transition_list_view.tmods["active"].endInsertRows()
    
    @Slot(list)
    def on_active_transitions_changed(self, active_list):
        self.active_fork_diagram.set_transitions(active_list)
        self._redraw_all = True
    
    @Slot(list)
    def on_constraints_changed(self, constraints):
        self.transition_list_view.tmods["background"].beginResetModel()
        query = self.tdb.query(Transition)
        #the first two constraints will be the wavelength bounds
        for constraint in constraints[:2]:
            query = query.filter(constraint)
        bk_trans = query.all()
        self.selection.transitions.background.set_values(bk_trans)
        self.transition_list_view.tmods["background"].endResetModel()
        self.transition_list_view.tmods["foreground"].beginResetModel()
        if len(constraints) == 2:
            foreground_trans = bk_trans
        else:
            query = query.join(Ion)
            for constraint in constraints[2:]:
                query = query.filter(constraint)
            foreground_trans = query.all()
        self.selection.transitions.foreground.set_values(foreground_trans)
        self.transition_list_view.tmods["foreground"].endResetModel()
        self.flux_display.ax.figure.canvas.draw()
    
    def on_save(self):
        try:
            self.tdb.commit()
            self.statusBar().showMessage(
                "Committed changes to database {}"\
                .format(self.tdb.db_url))
        except Exception as e:
            self.statusBar().showMessage(
                "save failed with error {}"\
                .format(e))
            print "save action failed"
            print e
            #TODO: raise a warning dialog
    
    def next_wv_region(self):
        self.wv_span_widget.step_forward()
    
    def prev_wv_region(self):
        self.wv_span_widget.step_back()
    
    def remove_transition(self):
        """remove current transition from the current group if applicable"""
        pass
    
    def add_transition(self):
        """add current transition to current group"""
        pass
    
    def select_parent_group(self):
        """set current group to group containing current transition"""
        pass
    
    def new_group(self):
        """begin a new group with the current transition"""
        pass



if __name__ == "__main__":
    
    qap = QtGui.QApplication([])
    
    #wvspan = WavelengthSpan(5000.0, 5500.0)
    #wvspanwid = WavelengthSpanWidget(wvspan)
    #wvspanwid.show()
    
    #sfw = SpeciesSetWidget()
    #sfw.show()
    
    #arbfilt = BaseExpressionWidget()
    #arbfilt.show()
    
    #import pdb; pdb.set_trace()
    tdb = tmb.ThimblesDB("/home/tim/sandbox/cyclefind/junk.tdb")
    #transitions = tdb.query(Transition).all()
    
    #tscat = TransitionScatter()
    #tscat.set_pool(transitions)
    #tscat.set_transition(transitions[0])
    #tscat.set_group(transitions[3:8])
    #tscat.show()
    
    spectra = tmb.io.read_spec("/home/tim/data/HD221170/hd.3720rd")
    gse = GroupingStandardEditor("default", tdb, spectra=spectra)
    gse.show()
    
    qap.exec_()
    
    #trans_chart = tmb.charts.fork_diagram.TransitionsChart(transitions)
    #trans_chart.ax.set_xlim(3500, 4000)
    #plt.show()
    
