from copy import copy
import thimblesgui as tmbg

from thimblesgui import QtGui, QtWidgets, QtCore, Qt
Signal = QtCore.Signal
Slot = QtCore.Slot
QModelIndex = QtCore.QModelIndex

import numpy as np
import matplotlib as mpl
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt

import thimbles as tmb
from thimbles import workingdataspace as wds
from thimbles.options import Option, opts
from thimbles.thimblesdb import Base
from thimbles.transitions import Transition, as_transition_group
from thimbles.transitions import TransitionGroupingStandard, TransitionGroup
from thimbles.abundances import Ion
from thimbles import ptable
from thimbles.periodictable import symbol_to_z, z_to_symbol
import thimbles.charts as charts

from thimblesgui import MatplotlibWidget
from thimblesgui import FluxDisplay

#color gamut
fg_color = "#2056DD" 
bk_color = "#708080" 
focus_color = "#F0DB62"


class TransitionExpressionWidget(QtWidgets.QWidget):
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
        
        self.dummy_trans = Transition(
            wv=5000.0, 
            ion=(26, 1), 
            ep=1.0, 
            loggf=-1.0,
            damp=tmb.transitions.Damping(0.0, 0.0, 0.0, 0.0),
        )
        
        self.x_le.editingFinished.connect(self.on_x_changed)
        self.y_le.editingFinished.connect(self.on_y_changed)
    
    def minimumSizeHint(self):
        return QtCore.QSize(200, 200)
    
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


class XYExpressionResolver(object):
    
    def resolve_expression(self, transition_list, expression):
        try:
            return [eval(expression) for t in transition_list]
        except Exception as e:
            return None
    
    def resolve_xy(self, transition_list):
        xvals = self.resolve_expression(transition_list, self.x_expression)
        yvals = self.resolve_expression(transition_list, self.y_expression)
        return xvals, yvals


class TransitionGroupBurstPlot(XYExpressionResolver):
    _bursts_initialized=False
    _centers_initialized=False
    
    def __init__(
            self,
            groups=None,
            x_expression="t.pseudo_strength()", 
            y_expression="t.ep",
            picker=None,
            ax=None,
            transition_tags=None, 
            **mplkwargs
    ):
        if groups is None:
            groups = []
        self.groups = groups
        self.x_expression=x_expression
        self.y_expression=y_expression
        self.picker = picker
        if ax is None:
            fig, ax = plt.subplots()
        self.ax = ax
        #print "mplkwargs", mplkwargs
        self.mplkwargs=mplkwargs
    
    @Slot(list)
    def set_groups(self, glist):
        if glist is None:
            glist = []
        self.groups = glist
        self.update()
    
    @Slot(str)
    def set_x_expression(self, expr_text):
        self.x_expression = expr_text
        self.update()
    
    @Slot(str)
    def set_y_expression(self, expr_text):
        self.y_expression = expr_text
        self.update()
    
    def update(self):
        #import pdb; pdb.set_trace()
        groups = self.groups
        if len(groups) == 0:
            if self._bursts_initialized:
                self.bursts.set_visible(False)
            if self._centers_initialized:
                self.centers.set_visible(False)
        else:
            group_xys = []
            nz_groups = [group for group in self.groups if len(group.transitions) > 0]
            group_xys = [self.resolve_xy(group.transitions) for group in nz_groups]
            nlines_each = [len(gr_xy[0]) for gr_xy in group_xys]
            nlines_tot = sum(nlines_each)
            burst_data = np.zeros((nlines_tot, 2, 2))
            x_centers = [np.mean(gr_xy[0]) for gr_xy in group_xys]
            y_centers = [np.mean(gr_xy[1]) for gr_xy in group_xys]
            lb = 0
            for i in range(len(group_xys)):
                ub = lb + nlines_each[i]
                burst_data[lb:ub, 0, 0] = x_centers[i]
                burst_data[lb:ub, 0, 1] = y_centers[i]
                burst_data[lb:ub, 1, 0] = group_xys[i][0]
                burst_data[lb:ub, 1, 1] = group_xys[i][1]
                lb = ub
            if not self._bursts_initialized:
                self.bursts = LineCollection(burst_data, **self.mplkwargs)
                self.ax.add_artist(self.bursts)
                self._bursts_initialized = True
            else:
                self.bursts.set_segments(burst_data)
            self.bursts.set_visible(True)
            if not self._centers_initialized:
                self.centers ,= self.ax.plot(x_centers, y_centers, picker=self.picker, markersize=10, marker="o", linestyle="none", **self.mplkwargs)
                self._centers_initialized = True
            else:
                self.centers.set_data(x_centers, y_centers)
            self.centers.set_visible(True)
            mdat = dict(kind="groups", groups=nz_groups, xvals=x_centers, yvals=y_centers)
            self.centers._md = mdat
            self.ax.figure._tmb_redraw=True


class TransitionScatterPlot(XYExpressionResolver):
    _points_initialized = False
    
    def __init__(
            self,
            transitions=None,
            x_expression="t.pseudo_strength()", 
            y_expression="t.ep",
            picker=None,
            auto_zoom=False,
            ax=None,
            transition_tags=None, 
            **mplkwargs
    ):
        self.x_expression=x_expression
        self.y_expression=y_expression
        self.picker = picker
        if transition_tags is None:
            transition_tags = {}
        self.transition_tags =transition_tags
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
    
    def update(self):
        transitions = self.transitions
        if len(transitions) == 0:
            if self._points_initialized:
                self.scatter.set_visible(False)
        else:
            xvals, yvals = self.resolve_xy(transitions)
            if not self._points_initialized:
                self.scatter ,= self.ax.plot(xvals, yvals, picker=self.picker, linestyle="none", **self.mplkwargs)
                self._points_initialized =True
            else:
                self.scatter.set_data(xvals, yvals)
            mdat = dict(kind="transitions", transitions=transitions, xvals=xvals, yvals=yvals)
            mdat.update(self.transition_tags)
            self.scatter._md = mdat
            self.scatter.set_visible(True)
            if self.auto_zoom:
                self.zoom_to_data()
        self.ax.figure._tmb_redraw = True
    
    def zoom_to_data(self, x_pad=0.1, y_pad=0.1):
        xpts, ypts = self.scatter.get_data()
        xmin, xmax = np.nanmin(xpts), np.nanmax(xpts)
        ymin, ymax = np.nanmin(ypts), np.nanmax(ypts)
        dx = xmax-xmin
        dy = ymax-ymin
        self.ax.set_xlim(xmin-x_pad*dx, xmax+x_pad*dx)
        self.ax.set_ylim(ymin-y_pad*dy, ymax+y_pad*dy)



class TransitionListModel(MappedListModel):
    
    def __init__(self, transitions=None):
        if transitions is None:
            transitions = []
        wv = ItemMappedColumn("Wavelength", lambda x: x.wv)
        #z = ItemMappedColumn("Z", lambda x: x.ion.z)
        #charge = ItemMappedColumn("Ion Charge", lambda x: x.ion.charge)
        symbol = ItemMappedColumn("Symbol", lambda x: "{} {}".format(x.ion.symbol, "I"*(x.ion.charge+1)), value_converter=lambda x: x)
        ep = ItemMappedColumn("Excitation\nPotential", lambda x: x.ep)
        loggf = ItemMappedColumn("log(gf)", lambda x: x.loggf)
        cols = [wv, symbol, ep, loggf]
        super(TransitionListModel, self).__init__(transitions, cols)


def group_species_symbol(group):
    transitions = group.transitions
    if len(transitions) == 0:
        return "None"
    symbols = list(set([t.ion.symbol for t in transitions]))
    charges = list(set([t.ion.charge for t in transitions]))
    species_symbol = "-".join(symbols)
    if len(charges) == 1:
        charge_symbol = "I"*(charges[0]+1)
    else:
        min_num = min(charges) + 1
        max_num = max(charges) + 1
        charge_symbol = "{}-{}".format(min_num, max_num)
    return "{} {}".format(symbols, charge_symbol)
    

class TransitionGroupListModel(MappedListModel):
    
    def __init__(self, groups):
        symbol = ListMappedColumn("Symbol", group_species_symbol, value_converter=lambda x: x)
        minwv = ListMappedColumn("Min Wavelength", lambda x: x.aggregate("wv", np.min))
        #wvsig = ListMappedColumn("Wavelength\nSpread", lambda x: x.aggregate("wv", np.std))
        maxwv = ListMappedColumn("Max Wavelength", lambda x: x.aggregate("wv", np.max))
        n= ListMappedColumn("N", lambda x: len(x), value_converter=lambda x: "{}".format(x))
        #epmean = ListMappedColumn("mean E.P.", lambda x: x.aggregate("ep", np.mean))        
        epsig = ListMappedColumn("sigma E.P.", lambda x: x.aggregate("ep", np.std)) 
        xmean = ListMappedColumn("mean\npseudo-strength", lambda x: x.aggregate("x", np.mean))        
        xsig = ListMappedColumn("sigma\npseudo-strength", lambda x: x.aggregate("x", np.std))               
        cols = [symbol, n, minwv, maxwv, epsig, xmean, xsig]
        super(TransitionGroupListModel, self).__init__(groups, cols)


class BackgroundTransitionListWidget(QtWidgets.QWidget):
    
    def __init__(self, grouping_standard_editor, parent=None):
        self.selection=grouping_standard_editor.selection
        self.tdb = grouping_standard_editor.tdb
        super(BackgroundTransitionListWidget, self).__init__(parent)
        layout = QtGui.QGridLayout()
        self.setLayout(layout)
        bk_tier = self.selection.transitions.background
        self.table_model = TransitionListModel(bk_tier.values)
        bk_tier.changed.connect(self.table_model.set_mapped_list)
        self.constraints = TransitionConstraints(grouping_standard_editor.wv_span, parent=self)
        self.constraints.constraintsChanged.connect(self.set_transition_constraints)
        layout.addWidget(self.constraints, 0, 0, 1, 1)
        self.table_view = QtGui.QTableView(parent=self)
        self.table_view.setModel(self.table_model)
        #import pdb; pdb.set_trace()
        self.table_view.setSelectionBehavior(1)#1==select rows
        #self.table_view.setSelectionMode(QtGui.QAbstractItemView.SelectionMode.SingleSelection)
        layout.addWidget(self.table_view, 1, 0, 1, 1)
    
    @Slot(list)
    def set_transition_constraints(self, constraint_list):
        self.table_model.beginResetModel()
        query = self.tdb.query(Transition).join(Ion)
        for constraint in constraint_list:
            query = query.filter(constraint)
        trans = query.all()
        self.selection.transitions.background.set_values(trans)
        self.table_model.endResetModel()

class ForegroundTransitionListWidget(QtWidgets.QWidget):
    
    def __init__(self, grouping_standard_editor, parent=None):
        self.parent_editor = grouping_standard_editor
        self.selection = grouping_standard_editor.selection
        self.tdb = grouping_standard_editor.tdb
        super(ForegroundTransitionListWidget, self).__init__(parent)
        layout = QtGui.QGridLayout()
        self.setLayout(layout)
        fg_tier = self.selection.transitions.foreground
        self.table_model = TransitionListModel(fg_tier.values)
        fg_tier.changed.connect(self.table_model.set_mapped_list)
        self.constraints = TransitionConstraints(grouping_standard_editor.wv_span, parent = self)
        self.constraints.constraintsChanged.connect(self.set_transition_constraints)
        layout.addWidget(self.constraints, 0, 0, 1, 3)
        self.table_view = QtGui.QTableView(parent=self)
        self.table_view.setModel(self.table_model)
        #self.table_view.setSelectionBehavior(QtGui.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_view.setSelectionBehavior(1)
        self.selection.transitions.foreground.set_selection_model(self.table_view.selectionModel())
        table_selection = self.table_view.selectionModel()
        layout.addWidget(self.table_view, 1, 0, 1, 3)
        self.inject_btn = QtGui.QPushButton("add to group")
        self.inject_btn.clicked.connect(self.on_inject)
        layout.addWidget(self.inject_btn, 2, 0, 1, 1)
        self.remove_btn = QtGui.QPushButton("remove from group")
        self.remove_btn.clicked.connect(self.on_remove)
        layout.addWidget(self.remove_btn, 2, 1, 1, 1)
        self.select_all_btn = QtGui.QPushButton("select all")
        self.select_all_btn.clicked.connect(self.on_select_all)
        layout.addWidget(self.select_all_btn, 2, 2, 1, 1)    

    @Slot(list)
    def set_transition_constraints(self, constraint_list):
        self.table_model.beginResetModel()
        query = self.tdb.query(Transition).join(Ion)
        for constraint in constraint_list:
            query = query.filter(constraint)
        trans = query.all()
        self.selection.transitions.foreground.set_values(trans)
        self.table_model.endResetModel()
    
    def purge_groups(self, groups):
        #get the current group model
        #old_gmod = self.parent_editor.group_view.groups_model
        #gmod.beginDeleteRows(QModelIndex(), old_group_index, old_group_index)
        #remove from selection
        cur_group_tier = self.selection.groups
        needs_reset = False
        for group in groups:
            try:
                group_index = self.selection.groups.index(group)
                cur_group_tier.values.pop(group_index)
                needs_reset = True
            except ValueError:
                pass
            self.parent_editor.tdb.delete(group)
        if needs_reset:
            cur_group_tier.set_values(cur_group_tier.values)#trigger updates
    
    def on_remove(self):
        focus_indexes = self.selection.groups.focus
        sbar = self.parent_editor.statusBar()
        to_remove = self.selection.transitions.foreground.focused
        if len(to_remove) == 0:
            sbar.showMessage("removal failed no transitions selected")
        else:
            gdict = self.parent_editor.grouping_dict
            groups_to_purge = []
            n_removed = 0
            for t in to_remove:
                group = gdict.get(t)
                if group is None:
                    pass #transition is not grouped
                else:
                    group.transitions.remove(t)
                    n_removed += 1
                    gdict.pop(t)
                    if len(group.transitions) == 0:
                        groups_to_purge.append(group)
            self.purge_groups(groups_to_purge)
            self.selection.groups.set_values(self.selection.groups.values)
            sbar.showMessage(
                "removed {} transitions; purged {} empty groups".format(n_removed, len(groups_to_purge)))
    
    def on_inject(self):
        focus_indexes = self.selection.groups.focus
        sbar = self.parent_editor.statusBar()
        if len(focus_indexes) == 0:
            sbar.showMessage("injection failed, no group selected!")
        elif len(focus_indexes) > 1:
            sbar.showMessage("injection failed, multiple groups selected!")
        else:
            group ,= self.selection.groups.focused
            to_inject = self.selection.transitions.foreground.focused
            old_trans = group.transitions
            new_trans = [t for t in to_inject if not (t in old_trans)]
            gdict = self.parent_editor.grouping_dict
            if len(new_trans) > 0:
                groups_to_purge = []
                for t in new_trans:
                    old_group = gdict.get(t)
                    if not old_group is None:
                        old_group.transitions.remove(t)
                        if len(old_group.transitions) == 0:
                            groups_to_purge.append(old_group)
                    group.transitions.append(t)
                    gdict[t] = group
                self.purge_groups(groups_to_purge)
                self.selection.groups.changed.emit(self.selection.groups.values)
                self.selection.groups.set_focus([group])
                sbar.showMessage(
                    "{} transitions added; {} empty groups purged".format(len(new_trans), len(groups_to_purge)))
            else:
                sbar.showMessage("all selected transitions already in selected group!")
    
    def on_select_all(self):
        n_foreground = len(self.selection.transitions.foreground.values)
        focus_idxs = range(n_foreground)
        self.selection.transitions.foreground.set_focus(focus_idxs)


class GroupSelectionWidget(QtWidgets.QWidget):
    
    def __init__(self, grouping_standard_editor, parent=None):
        self.selection = grouping_standard_editor.selection
        self.parent_editor = grouping_standard_editor
        self.grouping_dict = grouping_standard_editor.grouping_dict
        super(GroupSelectionWidget, self).__init__(parent)
        layout = QtGui.QGridLayout()
        self.setLayout(layout)
        
        cur_groups = self.transition_groups(self.selection.transitions.foreground.values)
        self.groups_model = TransitionGroupListModel(cur_groups)
        self.selection.groups.set_values(cur_groups)
        group_tier = self.selection.groups
        group_tier.changed.connect(self.groups_model.set_mapped_list)
        self.selection.transitions.foreground.changed.connect(self.set_groups_from_transitions)
        
        self.groups_view = QtGui.QTableView(parent=self)
        self.groups_view.setModel(self.groups_model)
        self.groups_view.setSelectionBehavior(1)
        #self.groups_view.setSelectionMode(QtGui.QAbstractItemView.SelectionMode.SingleSelection)
        self.selection.groups.set_selection_model(self.groups_view.selectionModel())
        layout.addWidget(self.groups_view, 0, 0, 1, 3)
        
        self.add_empty_btn = QtGui.QPushButton("add_empty")
        self.add_empty_btn.clicked.connect(self.on_add_empty)
        layout.addWidget(self.add_empty_btn, 1, 0, 1, 1)
    
    def on_add_empty(self):
        cur_groups = self.selection.groups.values
        empty_group = tmb.transitions.TransitionGroup()
        cur_groups.append(empty_group)
        self.selection.groups.set_values(cur_groups)
    
    def transition_groups(self, transitions):
        groups = set()
        for trans in transitions:
            tgroup = self.grouping_dict.get(trans)
            if not tgroup is None:
                groups.add(tgroup)
        return list(groups)
    
    @Slot(list)
    def set_groups_from_transitions(self, transitions):
        if self.parent_editor.group_cascade_cb.checkState():
            transitions = self.selection.transitions.foreground.values
            groups = self.transition_groups(transitions)
            self.selection.groups.set_values(groups)


mplframerate = Option("mplframerate", default=10, parent="GUI")

class NearestDataSpacePicker(object):
    
    def __init__(self, xtol, ytol):
        self.xtol = xtol
        self.ytol = ytol
    
    def __call__(self, artist, mouseevent):
        xp, yp = mouseevent.xdata, mouseevent.ydata
        xdata = artist.get_xdata()
        ydata = artist.get_ydata()
        dsquared = ((xdata-xp)/self.xtol)**2 + ((ydata-yp)/self.ytol)**2
        nearest_idx = np.argmin(dsquared)
        if dsquared[nearest_idx] < 1.0:
            return True, dict(ind=[nearest_idx])
        else:
            return False, None


class GroupingStandardEditor(QtWidgets.QMainWindow):
    _redraw_all = False
    
    def __init__(
            self, 
            grouping_standard,
            tdb,
            spectra=None,
            parent=None,
            burst_pick=None,
            scatter_pick=None,
    ):
        super(GroupingStandardEditor, self).__init__(parent)
        if spectra is None:
            spectra = []
        self.spectra = spectra
        self.tdb = tdb
        if burst_pick is None:
            burst_pick = NearestDataSpacePicker(0.2, 0.2)
        self.burst_pick = burst_pick
        if scatter_pick is None:
            scatter_pick = NearestDataSpacePicker(0.2, 0.2)
        self.scatter_pick = scatter_pick
        self.grouping_standard = grouping_standard
        self.grouping_dict = {}
        for group in self.grouping_standard.groups:
            for trans in group.transitions:
                self.grouping_dict[trans] = group
        
        blue_trans = tdb.query(Transition).order_by(Transition.wv).first()
        self.wv_span = WavelengthSpan(blue_trans.wv, blue_trans.wv*(1.002))
        background_transitions = tdb.query(Transition)\
                .filter(Transition.wv >= self.wv_span.min_wv-0.1)\
                .filter(Transition.wv <= self.wv_span.max_wv+0.1)\
                .all()
        foreground_transitions = background_transitions
        
        self.selection = GroupingEditorSelection()
        self.selection.transitions.background.set_values(background_transitions)
        self.selection.transitions.foreground.set_values(foreground_transitions)
        
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
        
        lmin = -1.0
        lmax = 4.5
        l_nub = 0.02
        self.background_tines = tmb.charts.fork_diagram.TransitionsChart(
            self.selection.transitions.background.values,
            lmin=lmin,
            lmax=lmax,
            l_nub=l_nub,
            linewidth=1.0,
            color=bk_color,
            tine_tags={"tier":"background"},
            ax=self.flux_display.ax,
        )
        self.flux_display.add_chart(self.background_tines)
        
        self.foreground_tines = tmb.charts.fork_diagram.TransitionsChart(
            self.selection.transitions.foreground.values,
            #grouping_dict=self.grouping_dict,
            color=fg_color,
            tine_picker=True,
            tine_tags={"tier":"foreground"},
            lmin=lmin,
            lmax=lmax,
            l_nub=l_nub,
            linewidth=1.0,
            zorder=2,
            ax=self.flux_display.ax,
        )
        self.flux_display.add_chart(self.foreground_tines)
        
        self.active_fork_diagram = tmb.charts.fork_diagram.TransitionsChart(
            self.selection.transitions.foreground.focused,
            grouping_dict=self.grouping_dict,
            color=focus_color,
            lmin=lmin,
            lmax=lmax,
            l_nub=l_nub,
            linewidth=2.5,
            handle_picker=True,
            zorder=3,
            ax=self.flux_display.ax,
        )
        self.flux_display.add_chart(self.active_fork_diagram)
        
        #set up transition scatter dock
        dock = QtGui.QDockWidget("Transition Scatter", self)
        x_exp = "t.pseudo_strength()"
        y_exp = "t.ep"
        self.scatter_display = TransitionExpressionWidget(
            x_expression=x_exp,
            y_expression=y_exp,
            parent=dock)
        self.mpl_displays.append(self.scatter_display)
        #background transitions
        self.background_scatter = TransitionScatterPlot(
            self.selection.transitions.background.values, 
            ax=self.scatter_display.ax, 
            markersize=4, 
            color=bk_color,
            auto_zoom=True,
            alpha=0.75
        )
        
        #foreground transitions
        self.foreground_scatter = TransitionScatterPlot(
            self.selection.transitions.foreground.values, 
            ax=self.scatter_display.ax, 
            picker=self.scatter_pick, 
            transition_tags={"tier":"foreground"},
            markersize=6,
            auto_zoom=False,
            color=fg_color,
        )
        
        #focused transitions
        self.focus_scatter = TransitionScatterPlot(
            self.selection.transitions.foreground.focused, 
            ax=self.scatter_display.ax, 
            picker=None,
            transition_tags={"tier":"focused"},
            markersize=6,
            auto_zoom=False,
            color=focus_color,
        )
        
        #current transition groups
        self.group_bursts = TransitionGroupBurstPlot(
            self.selection.groups.values, 
            ax=self.scatter_display.ax, 
            picker=self.burst_pick,
            color=fg_color,
        )
        #focused group
        self.focused_group_bursts = TransitionGroupBurstPlot(
            self.selection.groups.focused, 
            ax=self.scatter_display.ax, 
            color=focus_color,
            zorder=2,
        )
        
        self.scatter_display.mplwid.pickEvent.connect(self.on_pick_event)
        dock.setWidget(self.scatter_display)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        
        
        dock = QtGui.QDockWidget("Background Transitions")
        self.background_view = BackgroundTransitionListWidget(self, parent=dock)
        dock.setWidget(self.background_view)
        self.addDockWidget(Qt.TopDockWidgetArea, dock)

        dock = QtGui.QDockWidget("Foreground Transitions")
        self.foreground_view = ForegroundTransitionListWidget(self, parent=dock)
        dock.setWidget(self.foreground_view)
        self.addDockWidget(Qt.TopDockWidgetArea, dock)
        
        dock = QtGui.QDockWidget("Groups")
        self.group_view = GroupSelectionWidget(self, parent=dock)
        dock.setWidget(self.group_view)
        self.addDockWidget(Qt.TopDockWidgetArea, dock)
        
        self._connect_plot_events()
        #trigger update cascades
        self.wv_span.emit_bounds_changed()
    
    def _connect_plot_events(self):
        #tine diagrams listening to selection changes.
        #background
        self.selection.transitions.background.changed.connect(self.background_tines.set_transitions)
        #foreground
        self.selection.transitions.foreground.changed.connect(self.foreground_tines.set_transitions)
        #active
        self.selection.transitions.foreground.focusChanged.connect(self.active_fork_diagram.set_transitions)
        
        #scatter connections
        scatter_grams = [
            self.background_scatter,
            self.foreground_scatter,
            self.focus_scatter,
            self.group_bursts,
            self.focused_group_bursts,
        ]
        for scg in scatter_grams:
            self.scatter_display.xExpressionChanged.connect(
                scg.set_x_expression)
            self.scatter_display.yExpressionChanged.connect(
                scg.set_y_expression)
        #scatter plots
        self.selection.transitions.background.changed.connect(
            self.background_scatter.set_transitions)
        self.selection.transitions.foreground.changed.connect(
            self.foreground_scatter.set_transitions)
        self.selection.transitions.foreground.focusChanged.connect(
            self.focus_scatter.set_transitions)
        #burst diagrams
        self.selection.groups.changed.connect(
            self.group_bursts.set_groups)
        self.selection.groups.focusChanged.connect(
            self.focused_group_bursts.set_groups)
        
        #zoom on select
        self.selection.transitions.foreground.focusChanged.connect(self.transition_zoom)
    
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
        self.sel_toolbar = self.addToolBar("Selection")
        #grouping selection cascade toggle
        self.group_cascade_cb = QtGui.QCheckBox("Transition->Group Cascade", parent=self)
        self.group_cascade_cb.setCheckState(Qt.CheckState(2))
        self.sel_toolbar.addWidget(self.group_cascade_cb)
        #zoom to selection toggle
        self.selection_zoom_cb = QtGui.QCheckBox("Zoom to Selection", parent=self)
        self.selection_zoom_cb.setCheckState(Qt.CheckState(2))
        self.sel_toolbar.addWidget(self.selection_zoom_cb)
        #wavelength stepper widget
        self.global_wv_span_widget = FlatWavelengthSpanWidget(self.wv_span)
        self.sel_toolbar.addWidget(self.global_wv_span_widget)
    
    def make_status_bar(self):
        self.statusBar().showMessage("Ready")
    
    @Slot(list)
    def transition_zoom(self, translist):
        if self.selection_zoom_cb.checkState():
            if len(translist) > 0:
                twvs = [t.wv for t in translist]
                min_wv = np.min(twvs)
                max_wv = np.max(twvs)
                delta_wv = 0.001*max_wv
                self.flux_display.ax.set_xlim(min_wv-delta_wv, max_wv+delta_wv)
    
    @Slot(list)
    def on_pick_event(self, event_l):
        #print "pick event"
        event ,= event_l
        #print "kind", event.artist._md["kind"]
        #print event.ind
        if hasattr(event.artist, "_md"):
            metdat = event.artist._md
            if metdat["kind"] == "transitions":
                transitions = event.artist._md["transitions"]
                trans = transitions[event.ind[0]]
                self.selection.transitions.foreground.set_focus([trans])
            elif metdat["kind"] == "groups":
                groups = event.artist._md["groups"]
                group = groups[event.ind[0]]
                self.selection.groups.set_focus([group])
    
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
            print("save action failed")
            print(e)
            #TODO: raise a warning dialog
    
    def next_wv_region(self):
        self.wv_span_widget.step_forward()
    
    def prev_wv_region(self):
        self.wv_span_widget.step_back()


@tmb.tasks.task(
    result_name="grouping_standard",
)
def edit_grouping_standard(standard_name, tdb, spectra):
    gse = GroupingStandardEditor(standard_name, tdb, spectra=spectra)
    gse.show()


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
    #tdb = tmb.ThimblesDB("/home/tim/sandbox/cyclefind/junk.tdb")
    #tdb = tmb.ThimblesDB("/home/tim/globulars_hres_abundances/globulars.tdb")
    #tdb = tmb.ThimblesDB("/home/tim/sandbox/cyclefind/fdwarf_strong.db")
    tdb = tmb.ThimblesDB("/home/tim/data/SEGUE_globulars/glob.db")
    #transobj = tdb.query(Transition).first()
    #import pdb; pdb.set_trace()
    #transitions = tdb.query(Transition).all()
    
    #tscat = TransitionScatter()
    #tscat.set_pool(transitions)
    #tscat.set_transition(transitions[0])
    #tscat.set_group(transitions[3:8])
    #tscat.show()
    
    #spectra = tmb.io.read_spec("/home/tim/data/HD221170/hd.3720rd")
    
    spectra=[]
    
    load_elodie = True
    load_boss = True
    load_segue=True
    
    elodie_loads = [3]#, 3, 11, 21]
    boss_loads = []#[10, 20, 30, 40, 50]
    segue_loads = [3, 5, 7]
    
    if load_elodie:
        import h5py
        hf = h5py.File("/home/tim/data/elodie/elodie.h5")
        #wvs = tmb.utils.misc.vac_to_air_sdss(np.array(hf["h_wvs"]))
        wvs = np.array(hf["h_wvs"])
        for e_idx in elodie_loads:
            flux = np.array(hf["h_flux"][e_idx])
            spectra.append(tmb.Spectrum(wvs, flux))
        hf.close()
    
    if load_boss:
        import h5py
        hf = h5py.File("/home/tim/data/boss_standards/regularized_standards.h5")
        wvs = np.array(hf["wvs"])
        for b_idx in boss_loads:
            flux = np.array(hf["flux"][b_idx])
            spectra.append(tmb.Spectrum(wvs, flux))
    
    if load_segue:
        import h5py
        hf = h5py.File("/home/tim/data/SEGUE_globulars/globulars_all.h5")
        wvs = np.power(10, np.array(hf["log_wvs"]))
        for s_idx in segue_loads:
            flux = np.array(hf["flux"][s_idx])
            spectra.append(tmb.Spectrum(wvs, flux))
    
    TGS = tmb.wds.TransitionGroupingStandard
    gstand = tdb.query(TGS).filter(TGS.name == "default").one()
    gse = GroupingStandardEditor(gstand, tdb, spectra=spectra)
    gse.show()
    
    qap.exec_()
    
    #trans_chart = tmb.charts.fork_diagram.TransitionsChart(transitions)
    #trans_chart.ax.set_xlim(3500, 4000)
    #plt.show()
    
