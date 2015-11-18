
from thimblesgui import QtCore, QtGui, Qt
from thimblesgui.object_creation_dialogs import NewObjectDialog
import thimbles as tmb
QModelIndex = QtCore.QModelIndex

class QueryDialog(QtGui.QDialog):
    _global_namespace = tmb.wds.__dict__
    _query = None
    _query_valid = False
    
    def __init__(self, active_collection, default_query, parent=None):
        super().__init__(parent=parent)
        
        layout = QtGui.QGridLayout()
        self.setLayout(layout)
        
        self.active_collection = active_collection
        query_expr = self.active_collection._last_query_expression
        if query_expr is None:
            query_expr = default_query
        self.query_edit = QtGui.QTextEdit(query_expr)
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
            self.active_collection._last_query_expression = self._query_text
            self.accept()


class ActiveCollection(QtCore.QObject):
    values_reset = QtCore.pyqtSignal()
    _last_query_expression = None
    
    def __init__(self, db, values=None):
        super(ActiveCollection, self).__init__()
        if values is None:
            values = []
        self.values = values
        self.db = db
        
        self.indexer = {}
        self.set(values, )
    
    def __getitem__(self, index):
        return self.values[index]
    
    def set(self, values):
        self.indexer = {}
        for idx, val in enumerate(values):
            self.indexer[val] = idx
        self.values = values
        self.values_reset.emit()
    
    def extend(self, values):
        extend_vals = []
        for val in values:
            if not val in self.indexer:
                extend_vals.append(val)
                self.indexer[val] = len(values)
                self.values.append(val)
        if len(extend_vals) > 0:
            self.changed.emit()
            self.extended.emit(extend_vals)
    

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
    def on_values_reset(self):
        self.beginResetModel()
        self.endResetModel()
    
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
            columns = [repr_column]
        self.selection = selection
        self.selection_channel = selection_channel
        self.active_collection = active_collection
        
        layout = QtGui.QVBoxLayout()
        
        button_box = QtGui.QGroupBox(parent=self)
        box_layout = QtGui.QHBoxLayout()
        #make buttons
        self.query_btn = QtGui.QPushButton("query")
        self.query_btn.clicked.connect(self.on_edit_query)
        box_layout.addWidget(self.query_btn)
        self.edit_columns_btn = QtGui.QPushButton("doesn't work")
        box_layout.addWidget(self.edit_columns_btn)
        button_box.setLayout(box_layout)
        layout.addWidget(button_box)
        
        self.table_view = QtGui.QTableView()
        data_model = MappedListModel(active_collection, columns=columns)
        self.data_model = data_model
        active_collection.values_reset.connect(data_model.on_values_reset)
        self.table_view.setModel(data_model)
        self.table_view.setSelectionBehavior(1)#1==select rows
        self.table_view.setSelectionMode(1)#single selection?
        self.local_selection_model = self.table_view.selectionModel()
        if not selection_channel is None:
            self.selection.channels[selection_channel].changed.connect(self.on_global_selection_changed)
            self.local_selection_model.selectionChanged.connect(self.on_local_selection_changed)
        layout.addWidget(self.table_view)
        
        self.setLayout(layout)
    
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
    
    def on_edit_query(self):
        qd = QueryDialog(self.active_collection, "db.query(Star)", parent=self)
        qd.exec_()
        #q = qd.query
        #print(q)
        #print("on edit query")
