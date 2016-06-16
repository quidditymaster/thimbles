
# standard library
from itertools import cycle, product
import os
import sys
import time
import pickle

# 3rd party packages
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.optimize

#internal imports
from thimblesgui import QtCore, QtGui, Qt
from thimbles.options import opts
from thimbles.tasks import task_registry
from thimbles import workingdataspace as wds

import thimblesgui as tmbg
from thimblesgui.views import ObjectTreeWidget
from thimblesgui.task_widgets import TaskLauncher 
import thimbles as tmb
from thimblesgui import active_collections
from thimblesgui.active_collections import MappedListModel, ActiveCollection, ActiveCollectionView, QueryDialog
from thimblesgui.object_creation_dialogs import NewStarDialog
from thimblesgui.loading_dialog import LoadDialog
from thimblesgui.column_sets import star_columns, full_transition_columns, base_transition_columns, spectrum_columns
from thimblesgui.column_sets import namec as name_column
from thimblesgui.mct_dialog import ModelComponentTemplateApplicationDialog as MCTApplicationDialog
from thimblesgui.ew_editor import WidthsEditor
from thimblesgui.wavelength_span import WavelengthSpan, FlatWavelengthSpanWidget
from thimblesgui import MatplotlibWidget
from thimblesgui.transition_selector import TransitionSelectorWidget
from thimblesgui.normalization_editor import NormalizationEditor

from thimbles.contexts import model_spines as contextualizers

from thimbles import ThimblesDB
gui_resource_dir = os.path.join(os.path.dirname(__file__),"resources")

# ########################################################################### #

class TargetCollectionDialog(QtGui.QDialog):
    collection = None
    behavior = None
    
    def __init__(
            self,
            collections,
            behavior_options,
            parent,
    ):
        super(TargetCollectionDialog, self).__init__(parent=parent)
        collection_dict = {coll.name:coll for coll in collections}
        self.collections = collection_dict
        
        layout = QtGui.QGridLayout()
        names_label = QtGui.QLabel("collection")
        layout.addWidget(names_label, 0, 0, 1, 1)
        
        names_cbox = QtGui.QComboBox()
        layout.addWidget(names_cbox)
        for collection_name in collection_dict:
            names_cbox.addItem(collection_name)
        
        self.names_cbox = names_cbox
        layout.addWidget(names_cbox, 0, 1, 1, 1)
        
        behavior_cbox = QtGui.QComboBox()
        for behavior_opt in behavior_options:
            behavior_cbox.addItem(behavior_opt)
        self.behavior_cbox = behavior_cbox
        layout.addWidget(behavior_cbox, 1, 1, 1, 1)
        
        ok_btn = QtGui.QPushButton("OK")
        ok_btn.clicked.connect(self.on_ok)
        layout.addWidget(ok_btn, 2, 1, 1, 1)
        
        self.setLayout(layout)
    
    def on_ok(self):
        coll_name = self.names_cbox.currentText()
        self.collection = self.collections[coll_name]
        self.behavior=self.behavior_cbox.currentText()
        self.accept()


class NewCollectionDialog(QtGui.QDialog):
    new_collection = None
    
    def __init__(
            self,
            current_collections,
            #collection_factories,
            parent,
    ):
        super().__init__(parent=parent)
        collection_dict = {coll.name:coll for coll in current_collections}
        self.collections = collection_dict
        layout = QtGui.QGridLayout()
        names_label = QtGui.QLabel("collection name")
        layout.addWidget(names_label, 0, 0, 1, 1)
        
        self.names_le = QtGui.QLineEdit()
        layout.addWidget(self.names_le, 0, 1, 1, 1)
        
        self.status_label = QtGui.QLabel("")
        layout.addWidget(self.status_label, 1, 0, 1, 1)
        
        ok_btn = QtGui.QPushButton("OK")
        ok_btn.clicked.connect(self.on_ok)
        layout.addWidget(ok_btn, 2, 1, 1, 1)
        
        self.setLayout(layout)
    
    def on_ok(self):
        coll_name = self.names_le.text()
        if coll_name == "":
            self.status_label.setText("Name Invalid")
        elif coll_name in self.collections:
            self.status_label.setText("Name Already Taken!")
        else:
            new_collection = ActiveCollection(name=coll_name)
            self.new_collection = new_collection 
            self.accept()


class ParameterEvalCascadeWorker(QtCore.QThread):
    increment = QtCore.Signal(int)
    
    def __init__(
            self,
            parameters,
    ):
        QtCore.QThread.__init__(self)
        self.parameters = parameters
    
    def __del__(self):
        self.wait()
    
    def run(self):
        for param_idx in range(len(self.parameters)):
            param = self.parameters[param_idx]
            val = param.value
            self.increment.emit(param_idx)


class ParameterCalculationProgressBar(QtGui.QDialog):
    
    def __init__(
            self,
            parameters,
            label="calculating parameters",
            parent=None
    ):
        super().__init__(parent=parent)
        #self.parameters = parameters
        layout = QtGui.QVBoxLayout()
        label = QtGui.QLabel(label, parent=self)
        layout.addWidget(label)
        self.progress = QtGui.QProgressBar(minimum=0, maximum=len(parameters)-1, parent=self)
        layout.addWidget(self.progress)
        self.setLayout(layout)
        print("making me")
        self.calculate()

class YBoundsWidget(QtGui.QWidget):
    
    def __init__(
            self,
            ax,
            parent,
            min=0.0,
            max=1.1
    ):
        super().__init__(parent=parent)
        layout = QtGui.QHBoxLayout()
        self.ax = ax
        
        self.min_le = QtGui.QLineEdit(str(min))
        self.min_le.setFixedWidth(70)
        self.max_le = QtGui.QLineEdit(str(max))
        self.max_le.setFixedWidth(70)
        layout.addWidget(self.min_le)
        layout.addWidget(self.max_le)
        
        self.max_le.editingFinished.connect(self.on_bounds_changed)
        self.max_le.editingFinished.connect(self.on_bounds_changed)
        self.setLayout(layout)
        self.on_bounds_changed()
    
    def on_bounds_changed(self):
        print("changing ybounds")
        ymin = float(self.min_le.text())
        ymax = float(self.max_le.text())
        print(ymin, ymax)
        self.ax.set_ylim(ymin, ymax)
        self.ax.figure._tmb_redraw = True


class ActiveFluxDisplay(QtGui.QWidget):
    
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.wv_span = WavelengthSpan([15100, 15200])
        
        layout = QtGui.QGridLayout()
        self.span_widget = FlatWavelengthSpanWidget(
            wv_span=self.wv_span,
            step_frac=0.5,
            parent=self
        )
        layout.addWidget(self.span_widget, 0, 0)
        
        self.plot_widget = MatplotlibWidget(
            nrows = 1,
            parent=self,
            mpl_toolbar=False
        )
        layout.addWidget(self.plot_widget, 1, 0, 1, 2)
        
        self.ax = self.plot_widget.ax
        self.y_bounds_widget = YBoundsWidget(
            ax=self.ax,
            parent=self,
        )
        layout.addWidget(self.y_bounds_widget, 0, 1, 1, 1)
        
        self.setLayout(layout)
        
        self.wv_span.boundsChanged.connect(self.update_xlim)
        self.update_xlim()
        
    def update_xlim(self):
        self.ax.set_xlim(*self.wv_span.bounds)

        
class ThimblesMainWindow(QtGui.QMainWindow):
    
    def __init__(self, app):
        super(ThimblesMainWindow, self).__init__()
        self.db = app.project_db#ThimblesDB(project_db_path)
        app.finish_splash(self)
        
        self.setWindowTitle("Thimbles")
        self.selection = tmbg.selection.GlobalSelection(
            channels=[
                "star",
                "spectrum",
                "transition",
                "exemplar",
                "collection",
            ],
        )
        self.make_collections()
        tmb.wds.gui = self
        tmb.wds.gui_selection = self.selection
        tmb.wds.db = self.db
        
        SharedParameterSpace =tmb.analysis.SharedParameterSpace
        gp_result = self.db.query(SharedParameterSpace).filter(SharedParameterSpace.name == "global").all()
        if len(gp_result) > 1:
            raise Exception("multiple global parameter spaces found")
        elif len(gp_result) == 0:
            gp = tmb.analysis.SharedParameterSpace("global")
            gp_result = [gp]
            self.db.add(gp)
        global_params = gp_result[0]
        self.global_params = global_params
        contextualizers.add_global("global", global_params)
        
        self.main_flux_display = ActiveFluxDisplay(
            parent=self,
        )
        self.setCentralWidget(self.main_flux_display)
        self.make_charts()
        
        self.make_actions()
        self.make_menus()
        self.make_tool_bar()
        self.make_status_bar()
        self.make_dock_widgets()
        
        #initialize star collection
        top_stars = self.db.query(tmb.wds.Star).limit(100).all()
        star_coll = self.get_collection("stars")
        star_coll.set(top_stars)
        
        #initialize exemplar collection
        exemplars = global_params["exemplar_indexer"].value.transitions
        exemp_coll = self.get_collection("exemplars")
        exemp_coll.set(exemplars)
    
    def get_collection(self, collection_name):
        for coll in self.collection_collection.values:
            if coll.name == collection_name:
                return coll
        return None
    
    def make_collections(self):
        self.collection_collection = ActiveCollection("collections")
        #self.collection_view = ActiveCollectionView(
        #    self.collection_collection,
        #    selection=self.selection,
        #    columns=[name_column],
        #    selection_channel="collection",
        #    parent=self
        #)
        #self.setCentralWidget(self.collection_view)
        self.collection_collection.extend([
            ActiveCollection(name="stars"),
            ActiveCollection(name="spectra"),
            ActiveCollection(name="transitions"),
            ActiveCollection(name="exemplars"),
        ])
    
    def make_charts(self):
        spectra_collection = self.get_collection("spectra")
        self.spectra_chart = tmb.charts.spec_charts.SpectraChart(
            spectra = spectra_collection.values,
            bounds=self.main_flux_display.wv_span.bounds,
            ax = self.main_flux_display.ax
        )
        spectra_collection.changed.connect(self.on_spectra_changed)
        spectra_collection.changed.connect(self.print_args)
        self.main_flux_display.wv_span.boundsChanged.connect(self.on_main_wv_bounds_changed)
        #transition_collection = self.get_collection("transitions")
    
    def on_spectra_changed(self):
        print("on_spectra_changed called")
        spec_coll = self.get_collection("spectra")
        self.spectra_chart.set_spectra(spec_coll.values)
    
    def on_main_wv_bounds_changed(self, bounds):
        self.spectra_chart.set_bounds(bounds)
    
    def make_actions(self):
        self.new_collection_act = QtGui.QAction("&New Collection", self)
        self.new_collection_act.setStatusTip("Make a new named collection")
        self.new_collection_act.triggered.connect(self.on_new_collection)
        
        self.load_act = QtGui.QAction("&Load", self)
        self.load_act.setStatusTip("read data from file")
        self.load_act.triggered.connect(self.on_load)
        
        #QtGui.QAction(QtGui.QIcon(":/images/new.png"), "&Attach Database", self)
        
        #DB actions
        self.query_db_act = QtGui.QAction("&Query", self)
        self.query_db_act.setToolTip("Load Results of a Database Query into a named Collection")
        self.query_db_act.triggered.connect(self.on_query_db)
        
        self.commit_act = QtGui.QAction("&Commit", self)
        #self.save_act.setShortcut("Ctrl+s")
        self.commit_act.setStatusTip("commit state to database")
        self.commit_act.triggered.connect(self.on_commit)
        
        self.quit_act = QtGui.QAction("&Quit", self)
        self.quit_act.setShortcut("Ctrl+Q")
        self.quit_act.setStatusTip("Quit Thimbles")
        self.quit_act.triggered.connect(self.close)
        
        self.new_star_act = QtGui.QAction("&New Star", self)
        self.new_star_act.setStatusTip("Create a New Star object")
        self.new_star_act.triggered.connect(self.on_new_star)
        
        self.about_act = QtGui.QAction("&About", self)
        self.about_act.setStatusTip("About Thimbles")
        self.about_act.triggered.connect(self.on_about)
        
        self.view_widths_act = QtGui.QAction("widths", self)
        self.view_widths_act.triggered.connect(self.on_view_widths)
        
        self.view_spectroscopy_act = QtGui.QAction("spectroscopy", self)
        self.view_spectroscopy_act.triggered.connect(self.on_view_spectroscopy)
        
        self.edit_normalization_act = QtGui.QAction("Normalization", self)
        self.edit_normalization_act.triggered.connect(self.on_edit_normalization)
        
        self.broadcast_star_spectroscopy_act = QtGui.QAction("star->spectra", self)
        self.broadcast_star_spectroscopy_act.triggered.connect(self.on_broadcast_star_spectroscopy)
    
    
    def make_menus(self):
        menu_bar = self.menuBar()
        
        self.file_menu = menu_bar.addMenu("&File")
        self.edit_menu = menu_bar.addMenu("&Edit")
        self.db_menu = menu_bar.addMenu("&DB")
        self.context_menu = menu_bar.addMenu("&Context")
        self.source_menu = self.context_menu.addMenu("Sources")
        self.view_menu = menu_bar.addMenu("&View")
        self.help_menu = self.menuBar().addMenu("&Help")
        
        self.file_menu.addAction(self.load_act)
        self.file_menu.addAction(self.new_collection_act)
        self.file_menu.addAction(self.quit_act)
        
        self.db_menu.addAction(self.query_db_act)
        self.db_menu.addAction(self.commit_act)
        
        self.edit_menu.addAction(self.edit_normalization_act)
        
        self.source_menu.addAction(self.new_star_act)
        
        #self.modeling_menu = menu_bar.addMenu("&Modeling")
        #self.modeling_menu.addAction(self.apply_mct_act)
        
        self.view_menu.addAction(self.view_widths_act)
        #self.view_menu.addAction(self.view_spectroscopy_act)
        
        self.broadcast_menu = menu_bar.addMenu("Broadcast")
        self.broadcast_menu.addAction(self.broadcast_star_spectroscopy_act)
        
        self.help_menu.addAction(self.about_act)
    
    
    def make_tool_bar(self):
        self.tool_bar = self.addToolBar("Common Actions")
        self.tool_bar.addAction(self.new_collection_act)
        self.tool_bar.addAction(self.commit_act)
    
    def make_status_bar(self):
        self.statusBar().showMessage("Ready")
    
    def attach_as_dock(self, dock_name, widget, dock_area):
        dock = QtGui.QDockWidget(dock_name, self)
        dock.setAllowedAreas(Qt.AllDockWidgetAreas)
        dock.setWidget(widget)
        self.addDockWidget(dock_area, dock)
    
    def make_dock_widgets(self):
        star_collection = self.get_collection("stars")
        star_list_view = ActiveCollectionView(
            active_collection=star_collection,
            selection=self.selection,
            columns = star_columns,
            selection_channel = "star",
        )
        self.attach_as_dock("stars", star_list_view, Qt.BottomDockWidgetArea)
        
        spec_collection = self.get_collection("spectra")
        spec_list_view = ActiveCollectionView(
            active_collection = spec_collection,
            selection=self.selection,
            columns = spectrum_columns,
            selection_channel="spectrum",
        )
        self.attach_as_dock("spectra", spec_list_view, Qt.BottomDockWidgetArea)
        self.selection.channels["star"].changed.connect(self.on_star_selected)
        
        transition_collection = self.get_collection("transitions")
        transition_selector = TransitionSelectorWidget(
            db = self.db,
            wv_span = self.main_flux_display.wv_span,
            collection=transition_collection,
            selection=self.selection,
            parent=None,
        )
        transition_selector.constraints_widget.emit_constraints()
        self.attach_as_dock("transitions", transition_selector, Qt.RightDockWidgetArea)
        
        exemplar_collection = self.get_collection("exemplars")
        exemplar_list_view = ActiveCollectionView(
            active_collection = exemplar_collection,
            selection=self.selection,
            selection_channel="exemplar",
            columns=base_transition_columns,
        )
        self.attach_as_dock("exemplars", exemplar_list_view, Qt.RightDockWidgetArea)
        
        #dock = QtGui.QDockWidget("working data space", self)
        #dock.setAllowedAreas(Qt.AllDockWidgetAreas)
        #self.wds_widget = ObjectTreeWidget(wds, parent=dock)
        #dock.setWidget(self.wds_widget)
        #self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        #self.setCentralWidget(dock)
        
        #dock = QtGui.QDockWidget("tasks", self)
        #self.task_launcher = TaskLauncher(parent=dock)
        #dock.setWidget(self.task_launcher)
        #self.addDockWidget(Qt.RightDockWidgetArea, dock)
    
    def bad_selection(self, msg=None):
        """indicate when operations cannot be performed because of bad user selections
        """
        if msg == None:
            msg = "invalid selection\n"
        else:
            msg = "invalid selection\n" + msg
        self.wd = tmbg.dialogs.WarningDialog(msg)
        self.wd.warn()
    
    def get_target_collection(self, behaviors, ):
        tcd = TargetCollectionDialog(
            collections=self.collection_collection.values,
            behavior_options=behaviors,
            parent=self,
        )
        tcd.exec_()
        return tcd.collection, tcd.behavior
    
    def on_about(self):
        about_msg =\
"""
THIMBLES:
  Tools for
  Handling 
  Integrated
  Modeling of
  Breathtakingly
  Large
  Ensembles of
  Spectra

Thimbles was developed in the Cosmic Origins group of Inese Ivans at the University of Utah.
"""
        QtGui.QMessageBox.about(self, "about Thimbles GUI", about_msg)
    
    def on_broadcast_star_spectroscopy(self):
        selected_star = self.selection["star"]
        if selected_star is None:
            self.bad_selection("No star selected")
        collection, behavior = self.get_target_collection(
            behaviors=["replace", "extend"]
        )
        if collection is None:
            return
        spectra = selected_star.spectroscopy
        if behavior == "replace":
            collection.set(spectra)
        elif behavior == "extend":
            collection.extend(spectra)
    
    def on_edit_normalization(self):
        selected_spec = self.selection["spectrum"]
        if selected_spec is None:
            self.bad_selection("No spectrum selected")
        self._ne = NormalizationEditor(
            spectrum=selected_spec,
            parent=self,
        )
        self._ne.show()
        #self._ne.exec_()
    
    def on_new_collection(self):
        ncd = NewCollectionDialog(
            current_collections=self.collection_collection.values,
            parent=self
        )
        ncd.exec_()
        if not ncd.new_collection is None:
            new_coll = ncd.new_collection
            self.collection_collection.extend([new_coll])
    
    def on_new_star(self):
        new_star = NewStarDialog.get_new(parent=self)
        if not new_star is None:
            self.db.add(new_star)
    
    def on_query_db(self):
        collection, behavior = self.get_target_collection(
            behaviors=["replace", "extend"]
        )
        if collection is None:
            return
        query_dialog = QueryDialog(
            parent=self,
        )
        query_dialog.exec_()
        
        query = query_dialog.query
        if query is None:
            return
        qresult = query.all()
        if behavior == "replace":
            collection.set(qresult)
        elif behavior == "extend":
            collection.extend(qresult)
    
    def on_load(self):
        collection, behavior = self.get_target_collection(
            behaviors=["replace", "extend"]
        )
        if collection is None:
            return
        
        coll_name = collection.name.lower()
        if "spect" in coll_name:
            rfunc_expr="tmb.io.read_spec"
        elif ("transition" in coll_name) or ("line" in coll_name):
            rfunc_expr = "tmb.io.read_linelist"
        else:
            rfunc_expr = ""
        ld = tmbg.loading_dialog.LoadDialog(
            rfunc_expr=rfunc_expr,
            parent=self,
        )
        ld.exec_()
        if not ld.result is None:
            if behavior == "replace":
                collection.set(ld.result)
            elif behavior == "extend":
                collection.extend(ld.result)
    
    def on_star_selected(self):
        selected_star = self.selection["star"]
        if selected_star is None:
            spectroscopy = []
        else:
            spectroscopy = selected_star.spectroscopy
        spec_collection = self.get_collection("spectra")
        spec_collection.set(spectroscopy)
    
    def on_view_spectroscopy(self):
        selected_star = self.selection["star"]
        if selected_star is None:
            self.bad_selection("No Star Selected")   
    
    def on_view_widths(self):
        selected_star = self.selection["star"]
        if selected_star is None:
            self.bad_selection("No Star Selected")
        
        #calc_params = [spec["obs_flux"] for spec in selected_star.spectroscopy]
        #progress_dialog = QtGui.QProgressDialog(
        #    "executing parameter cascades",
        #    "cancel? maybe?",
        #    0,
        #    len(calc_params)-1,
        #    parent=self
        #)
        #calc_thread = ParameterEvalCascadeWorker(calc_params)
        #calc_thread.increment.connect(progress_dialog.setValue)
        #calc_thread.start()
        #progress_dialog.exec_()
        #pcpb = ParameterCalculationProgressBar(calc_params, parent=self)
        #pcpb.exec_()
        
        transition_indexer = self.global_params["transition_indexer"].value
        exemplar_indexer = self.global_params["exemplar_indexer"].value
        exemplar_map = self.global_params["exemplar_map"].value
        widths_editor = WidthsEditor(
            star=selected_star,
            transition_indexer=transition_indexer,
            exemplar_indexer=exemplar_indexer,
            exemplar_map=exemplar_map,
            selection=self.selection,
            parent=self
        )
        widths_editor.show()
    
    def on_commit(self):
        self.db.commit()
        save_msg = "committed to {}".format(self.db.path)
        self.statusBar().showMessage(save_msg)
    
    def print_args(self, *args, **kwargs):
        """prints the arguments passed in
        useful for connecting to events during gui debugging"""
        print("in print_args")
        print(args, kwargs)

