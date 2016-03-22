
from thimblesgui import QtCore, QtGui, Qt
from thimblesgui.object_creation_dialogs import NewObjectDialog
from thimblesgui.loading_dialog import LoadDialog
import thimbles as tmb
QModelIndex = QtCore.QModelIndex

class QueryDialog(QtGui.QDialog):
    _global_namespace = tmb.wds.__dict__
    _query = None
    _query_valid = False
    
    def __init__(
            self,
            active_collection,
            #default_query="db.query()",
            parent=None
    ):
        super().__init__(parent=parent)
        
        layout = QtGui.QGridLayout()
        self.setLayout(layout)
        
        self.active_collection = active_collection
        query_expr = self.active_collection.default_query
        #if query_expr is None:
        #    query_expr = default_query
        self.query_edit = QtGui.QTextEdit()
        self.query_edit.setPlainText(query_expr)
        self.query_edit.textChanged.connect(self.on_editing)
        layout.addWidget(self.query_edit, 0, 0, 1, 3)
        
        self.status_label = QtGui.QLabel("query unprocessed")
        layout.addWidget(self.status_label, 1, 0, 1, 3)
        
        self.parse_btn = QtGui.QPushButton("parse query")
        self.parse_btn.clicked.connect(self.parse)
        layout.addWidget(self.parse_btn, 2, 2, 1, 1)
        
        self.run_btn = QtGui.QPushButton("run query")
        self.run_btn.clicked.connect(self.on_run)
        layout.addWidget(self.run_btn, 2, 3, 1, 1)
    
    def on_editing(self):
        if self._query_valid:
            self._query_valid = False
            self.status_label.setText("query edited")
    
    def parse(self):
        query_text = self.query_edit.toPlainText()
        self._query_text = query_text
        try:
            self._query = eval(query_text, self._global_namespace, {"db":self.active_collection.db})
            self._query_valid = True
            self.status_label.setText("query parsed successfully")
        except Exception as e:
            self._query_valid = False
            self.status_label.setText(repr(e))
    
    @property
    def query(self):
        if not self._query_valid:
            self.parse()
        return self._query
    
    def on_run(self):
        query = self.query
        if self._query_valid:
            result = query.all()
            self.active_collection.set(result)
            self.active_collection.default_query = self._query_text
            self.accept()

class ItemMappedColumn(object):
    
    def __init__(
            self,
            column_name,
            getter,
            setter=None,
            value_converter=None,
            string_converter=None,
            qt_flag=None,
            role_dict=None,
    ):
        self.column_name = column_name
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
        if role_dict is None:
            role_dict = {}
        self.role_dict = role_dict
    
    def get(self, data_obj, role):
        value = self.getter(data_obj)
        if role == Qt.DisplayRole:
            return self.value_converter(self.getter(data_obj))
        elif role in self.role_dict:
            role_converter = self.role_dict[role]
            return role_converter(value)
    
    def set(self, data_obj, value, role):
        if role == Qt.EditRole:
            self.setter(data_obj, self.string_converter(value))
        return


class NewItemMappedColumnDialog(NewObjectDialog):

    def __init__(self, parent):
        super().__init__(
            fields=[
                ("column_name", ""),
                ("getter", "lambda x: x"),
                ("value_converter", 'lambda x: "{}".format(x)'),
            ],
            factory = ItemMappedColumn,
            parent=parent
        )


class MappedListModel(QtCore.QAbstractTableModel):
    
    def __init__(self, active_collection, columns):
        super(MappedListModel, self).__init__()
        self.active_collection = active_collection
        self.column_map = columns
    
    @property
    def _data(self):
        return self.active_collection.values
    
    @QtCore.pyqtSlot()
    def on_reset(self):
        self.beginResetModel()
        self.endResetModel()
    
    @QtCore.pyqtSlot(list)
    def on_extended(self, extended_idxs):
        self.beginInsertRows(QtCore.QModelIndex(), extended_idxs[0], extended_idxs[-1])
        self.endInsertRows()
    
    def rowCount(self, parent=QModelIndex()):
        return len(self._data)
    
    def columnCount(self, parent=QModelIndex()):
        return len(self.column_map)
    
    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self.column_map[section].column_name
            if orientation == Qt.Vertical:
                return section
    
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
            print("data set failed with exception")
            print(e)
            return False


repr_column = ItemMappedColumn("value", getter=lambda x: x, value_converter=lambda x: repr(x))

class ActiveCollection(QtCore.QObject):
    reset = QtCore.pyqtSignal()
    extended = QtCore.pyqtSignal()
    
    def __init__(
            self,
            name,
            db,
            values=None,
            default_query=None,
            default_columns=None,
            default_read_func=None,
    ):
        super(ActiveCollection, self).__init__()
        self.name = name
        if values is None:
            values = []
        self.values = values
        self.db = db
        if default_query is None:
            default_query = "db.query().offset(0).limit(20)"
        self.default_query=default_query
        if default_columns is None:
            default_columns = [repr_column]
        self.default_columns = default_columns
        self.default_read_func = default_read_func
        
        self.indexer = {}
        self.set(values, )
    
    def __getitem__(self, index):
        return self.values[index]
    
    def set(self, values):
        self.indexer = {}
        for idx, val in enumerate(values):
            self.indexer[val] = idx
        self.values = values
        self.reset.emit()
    
    def extend(self, values):
        extend_vals = []
        for val in values:
            if not val in self.indexer:
                extend_vals.append(val)
                self.indexer[val] = len(values)
                self.values.append(val)
        if len(extend_vals) > 0:
            self.extended.emit(extend_vals)
    
    def add_all(self):
        self.db.add_all(self.values)


class ActiveCollectionView(QtGui.QWidget):
    
    def __init__(
            self,
            active_collection,
            selection,
            columns = None,
            selection_channel=None,
            parent=None
    ):
        super(ActiveCollectionView, self).__init__(parent)
        if columns is None:
            columns = active_collection.default_columns
        self.selection = selection
        self.selection_channel = selection_channel
        self.active_collection = active_collection
        
        self.make_actions()
        self.make_menu()
        
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.menu_bar)
        
        self.table_view = QtGui.QTableView()
        data_model = MappedListModel(active_collection, columns=columns)
        self.data_model = data_model
        #connect to the data model signals
        active_collection.reset.connect(data_model.on_reset)
        active_collection.extended.connect(data_model.on_extended)
        self.table_view.setModel(data_model)
        self.table_view.setSelectionBehavior(1)#1==select rows
        self.table_view.setSelectionMode(1)#single selection?
        self.local_selection_model = self.table_view.selectionModel()
        if not selection_channel is None:
            self.selection.channels[selection_channel].changed.connect(self.on_global_selection_changed)
            self.local_selection_model.selectionChanged.connect(self.on_local_selection_changed)
        layout.addWidget(self.table_view)
        
        self.setLayout(layout)
    
    def make_actions(self):
        self.clear_act = QtGui.QAction("clear", self)
        self.clear_act.setToolTip("empty the collection")
        self.clear_act.triggered.connect(self.on_clear)
        
        self.load_act = QtGui.QAction("load", self)
        self.load_act.setToolTip("load objects from file")
        self.load_act.triggered.connect(self.on_load)
        
        self.query_act = QtGui.QAction("query db", self)
        self.query_act.setToolTip("populate collection from the database via a SQLAlchemy query")
        self.query_act.triggered.connect(self.on_query)
        
        self.add_all_act = QtGui.QAction("add to db", self)
        self.add_all_act.setToolTip("persist collection instances to the database")
        self.add_all_act.triggered.connect(self.on_add_all)
    
    def make_menu(self):
        menu_bar = QtGui.QMenuBar(parent=self)
        self.menu_bar = menu_bar
        data_menu = self.menu_bar.addMenu("manage")
        data_menu.addAction(self.load_act)
        data_menu.addAction(self.query_act)
        data_menu.addAction(self.add_all_act)
        data_menu.addAction(self.clear_act)
    
    def on_add_all(self):
        self.active_collection.add_all()
    
    def on_clear(self):
        self.active_collection.set([])
    
    def on_load(self):
        ld = LoadDialog(self.active_collection)
        ld.exec_()
        result = ld.result
        if not result is None:
            self.active_collection.set(result)
    
    def on_local_selection_changed(self, selected, deslected):
        if len(selected) > 0:
            row_idx = selected.indexes()[0].row()
            selected_item = self.active_collection[row_idx]
            self.selection[self.selection_channel] = selected_item
    
    def get_local_selection(self):
        srows = self.local_selection_model.selectedRows()
        if len(srows) > 0:
            sel_idx = srows[0].row()
            return self.active_collection[sel_idx]
        return None
    
    def get_global_selection(self):
        return self.selection[self.selection_channel]
    
    def on_global_selection_changed(self):
        local_selection = self.get_local_selection()
        global_selection = self.get_global_selection()
        if local_selection != global_selection:
            local_idx = self.active_collection.indexer.get(global_selection)
            qsel = QtGui.QItemSelection()
            if not local_idx is None:
                start_qidx = self.data_model.index(local_idx, 0)
                end_qidx = self.data_model.index(local_idx, self.data_model.columnCount() -1)
                self.table_view.scrollTo(start_qidx)
                qsel.select(start_qidx, end_qidx)
            self.local_selection_model.select(qsel, QtGui.QItemSelectionModel.SelectCurrent)
    
    def on_query(self):
        qd = QueryDialog(self.active_collection, parent=self)
        qd.exec_()
        #q = qd.query
        #print(q)
        #print("on edit query")


class TransitionCollection(ActiveCollection):
    
    def __init__(
            name,
            db,
    ):
        super().__init__(
            name=name,
            db=db,             
            default_columns=base_transition_columns,
            defautl_read_func="tmb.io.read_linelist",
            default_query=default_tq,
        )

class TransitionCollectionView(ActiveCollectionView):
    
    def __init__(
            self,
            active_collection,
            selection,
            #columns = None,
            #selection_channel=None,
            parent=None
    ):
        super().__init__(
            active_collection,
            selection,
            parent
        )
