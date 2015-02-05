from copy import copy
import thimblesgui as tmbg

from PySide import QtGui, QtCore
Qt = QtCore.Qt
from PySide.QtCore import Signal, Slot

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import thimbles as tmb
from thimbles import workingdataspace as wds
from thimbles.thimblesdb import Base
from thimbles.transitions import Transition, as_transition_group
from thimbles.abundances import Ion
from thimbles import as_wavelength_sample
from thimbles import ptable
from thimbles.periodictable import symbol_to_z, z_to_symbol
import thimbles.charts as charts
#from thimbles.charts import MatplotlibCanvas

from thimblesgui import MatplotlibWidget
from thimblesgui import FluxDisplay

class CurrentSelection(QtCore.QObject):
    transitionChanged = Signal(Transition)
    poolChanged = Signal(list)
    groupChanged = Signal(list)
    
    _pool = None
    _transition = None
    _group = None
    
    def __init__(self, transition=None, pool=None, group=None):
        super(GroupingEditorSelection, self).__init__()
        self._transition = transition
        self._pool = pool
        self._group = group
    
    @property
    def transition(self):
        return self._transition
        
    @transition.setter
    def transition(self, value):
        self._transition = value
        self.transitionChanged.emit(value)
    
    @property
    def pool(self):
        return self._pool
    
    @pool.setter
    def pool(self, value):
        self._pool = value
        self.poolChanged.emit(value)
    
    @property
    def group(self):
        return self._group

    @group.setter
    def group(self, value):
        self._group = value
        self.groupChanged.emit(value)

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

class WavelengthSpanWidget(QtGui.QWidget):
    boundsChanged = Signal(list)
    
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
        self.wv_span_widget = WavelengthSpanWidget(wv_span, parent=self)
        layout.addWidget(self.wv_span_widget, 0, 1, 1, 1)
        self.species_filter_cb = QtGui.QCheckBox()
        self.species_selector = SpeciesSelectorWidget(parent=self)
        layout.addWidget(self.species_filter_cb, 1, 0, 1, 1)
        layout.addWidget(self.species_selector, 1, 1, 1, 1)
        self.expr_filter_cb = QtGui.QCheckBox()
        self.expr_wid = BaseExpressionWidget(parent=self, label="filter")
        layout.addWidget(self.expr_filter_cb, 2, 0, 1, 1)
        layout.addWidget(self.expr_wid, 2, 1, 1, 1)
        
        self.wv_span.boundsChanged.connect(self.emit_constraints)
        self.species_filter_cb.toggled.connect(self.emit_constraints)
        self.expr_filter_cb.toggled.connect(self.emit_constraints)
        self.species_selector.speciesChanged.connect(self.on_species_changed)
        self.expr_wid.expressionChanged.connect(self.on_expression_changed)
    
    def emit_constraints(self):
        self.constraintsChanged.emit(self.transition_constraints())
    
    def on_species_changed(self, new_species):
        if self.filter_cb.checkState():
            self.emit_constraints()
    
    def on_expression_changed(self):
        if self.expr_filter_cb.checkState():
            self.emit_constraints()
    
    def transition_constraints(self):
        constraints = []
        constraints.append([Transition.wv >= self.wv_span.min_wv])
        constraints.append([Transition.wv <= self.wv_span.max_wv])
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

class TransitionScatter(QtGui.QWidget):
    _scatter_initialized = False
    _group_initialized = False
    _transition_initialized = False
    
    def __init__(
            self, 
            x_expression="t.loggf", 
            y_expression="t.ep",
            parent=None
    ):
        super(TransitionScatter, self).__init__(parent)
        self.x_expression = x_expression
        self.y_expression = y_expression
        self.mplwid = MatplotlibWidget(parent=self)
        self.ax = self.mplwid.ax
        
        layout = QtGui.QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self.mplwid)
        #x and y expression line edits
        self.x_le = QtGui.QLineEdit()
        self.x_le.setText(self.x_expression)
        self.ax.set_xlabel(self.x_expression)
        layout.addWidget(self.x_le)
        self.y_le = QtGui.QLineEdit()
        self.y_le.setText(self.y_expression)
        self.ax.set_ylabel(self.y_expression)
        layout.addWidget(self.y_le)
        
        self.dummy_trans = Transition(5000.0, (26, 1), 1.0, -1.0)
        self.x_le.editingFinished.connect(self.on_expression_changed)
        self.y_le.editingFinished.connect(self.on_expression_changed)
    
    @Slot(list)
    def set_pool(self, tlist):
        self.pool = tlist
        self.update_scatter()
        self.mplwid.draw()
    
    @Slot(Transition)
    def set_transition(self, transition):
        self.transition = transition
        self.update_transition()
        self.mplwid.draw()
    
    @Slot(list)
    def set_group(self, group):
        self.group = as_transition_group(group)
        self.update_group()
        self.mplwid.draw()
    
    def on_expression_changed(self):
        #import pdb; pdb.set_trace()
        x_text = self.x_le.text()
        dummy_x = self.resolve_expression([copy(self.dummy_trans)], x_text) 
        y_text = self.y_le.text()
        dummy_y = self.resolve_expression([copy(self.dummy_trans)], y_text) 
        bad_x = False
        if dummy_x is None:
            self.x_le.setStyleSheet(_parse_error_style)
            bad_x = True
        try:
            float(dummy_x[0])
        except TypeError:
            bad_x = True
        
        bad_y = False
        if dummy_y is None:
            self.y_le.setStyleSheet(_parse_error_style)
            bad_y = True
        try:
            float(dummy_y[0])
        except TypeError:
            bad_y = True
        if all([not bad_x, not bad_y]):
            self.x_expression = x_text
            self.y_expression = y_text
            self.x_le.setStyleSheet(_parse_success_style)
            self.y_le.setStyleSheet(_parse_success_style)
            self.ax.set_xlabel(self.x_expression)
            self.ax.set_ylabel(self.y_expression)
            self.update_all()
            self.auto_zoom()
            self.mplwid.draw()
    
    def resolve_expression(self, transition_list, expression):
        try:
            return [eval(expression) for t in transition_list]
        except Exception as e:
            return None
    
    def resolve_xy(self, transition_list):
        xvals = self.resolve_expression(transition_list, self.x_expression)
        yvals = self.resolve_expression(transition_list, self.y_expression)
        return xvals, yvals
    
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
    
    def update_scatter(self):
        pool = self.pool
        xvals, yvals = self.resolve_xy(transitions)
        if not self._scatter_initialized:
            self.scatter ,= self.ax.plot(xvals, yvals, picker=5, linestyle="none", marker="o", color="k", linewidth=2)
            self._scatter_initialized =True
        else:
            self.scatter.set_data(xvals, yvals)
        self.scatter._md = dict(name="pool", transitions=pool)
        self.auto_zoom()
    
    def update_transition(self):
        transition = self.transition
        sel_x , sel_y= self.resolve_xy([self.transition])
        #import pdb; pdb.set_trace()
        if not self._transition_initialized:
            selection_color = "#AA5500"
            self.select_hl ,= self.ax.plot([sel_x], [sel_y], markersize=30, linestyle="none", color=selection_color, marker="+") #not selectable
            self._transition_initialized = True
        else:
            self.select_hl.set_data([sel_x], [sel_y])
    
    def update_group(self):
        group = self.group
        gtrans = group.transitions
        grp_color = "#55AAFF"
        grp_x, grp_y = self.resolve_xy(gtrans)
        grp_x_center, grp_y_center = np.nanmean(grp_x), np.nanmean(grp_y)
        burst_data = self._line_burst_data(grp_x, grp_y, grp_x_center, grp_y_center)
        if not self._group_initialized:
            self.group_burst = mpl.collections.LineCollection(burst_data, color=grp_color, linewidth=4.0) #not pickable
            self.ax.add_artist(self.group_burst)
            self.group_dot ,= self.ax.plot([grp_x_center], [grp_y_center], picker=6, color=grp_color, marker="o", markersize=15)
            self._group_initialized = True
        else:
            self.group_burst.set_segments(burst_data)
            self.group_dot.set_data([grp_x_center], [grp_y_center])
        self.group_dot._md = dict(name="group", group=group)
    
    def update_all(self):
        self.update_scatter()
        self.update_transition()
        self.update_group()
    
    def resolve_pick_event(self, event):
        print "a pick event!", event
    
    def _connect_plot_events(self):
        self.mplwid.pickEvent.connect(self.resolve_pick_event)
    
    def auto_zoom(self, x_pad=0.1, y_pad=0.1):
        xpts, ypts = self.scatter.get_data()
        xmin, xmax = np.nanmin(xpts), np.nanmax(xpts)
        ymin, ymax = np.nanmin(ypts), np.nanmax(ypts)
        dx = xmax-xmin
        dy = ymax-ymin
        self.ax.set_xlim(xmin-x_pad*dx, xmax+x_pad*dx)
        self.ax.set_ylim(ymin-y_pad*dy, ymax+y_pad*dy)


class GroupingStandardEditor(QtGui.QMainWindow):
    
    def __init__(
            self, 
            standard_name, 
            tdb,
            spectra=None,
            parent=None,
    ):
        super(GroupingStandardEditor, self).__init__(parent)
        if spectra is None:
            spectra = []
        self.spectra = spectra
        
        self.tdb = tdb
        gstand = tdb.query(tmb.transitions.TransitionGroupingStandard)\
                    .filter(tmb.transitions.TransitionGroupingStandard.name==standard_name).one()
        self.grouping_standard = gstand
        self.grouping_dict = {}
        for group in self.grouping_standard.groups:
            for trans in group.transitions:
                self.grouping_dict[trans] = group
        
        blue_trans = tdb.query(Transition).order_by(Transition.wv).first()
        self.wv_span = WavelengthSpan(blue_trans.wv, blue_trans.wv*(1.002))
        
        self.make_actions()
        self.make_status_bar()
        
        self.flux_display = FluxDisplay(self.wv_span, parent=self)
        self.setCentralWidget(self.flux_display)
        chart_kwargs = {}
        chart_kwargs["ax"] = self.flux_display.ax
        chart_kwargs["label_axes"] = False
        chart_kwargs["auto_zoom"] = False
        for spec in self.spectra:
            schart = charts.SpectrumChart(spec, **chart_kwargs)
            self.flux_display.add_chart(schart)
        
        transitions = self.tdb.query(tmb.transitions.Transition).all()
        self.fork_diagram = tmb.charts.fork_diagram.TransitionsChart(transitions, ax=self.flux_display.ax)
        
        
        dock = QtGui.QDockWidget("Transiton Filter", self)
        self.constraints = TransitionConstraints(self.wv_span, parent=dock)
        dock.setWidget(self.constraints)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
    
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
    
    def make_status_bar(self):
        self.statusBar().showMessage("Ready")
    
    @Slot(list)
    def on_constraints_changed(self, constraints):
        print "constraints changed!"
    
    def on_save(self):
        try:
            self.tdb.commit()
            self.statusBar().showMessage("Committed changes to database {}"\
                                         .format(self.tdb.db_url))
        except Exception as e:
            print "save action failed"
            print e
            #TODO: raise a warning dialog
    
    def next_wv_region(self):
        print "next wv region!"
    
    def prev_wv_region(self):
        print "prev wv region!"
    
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
    
