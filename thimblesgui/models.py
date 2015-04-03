
import thimblesgui as tmbg
from thimblesgui import QtGui, QtCore, Qt
QModelIndex = QtCore.QModelIndex

import numpy as np
import matplotlib.pyplot as plt


class Column(object):
    
    def __init__(
            self, 
            name, 
            getter_dict,
            setter_dict=None,
            editable=False, 
            selectable=True,
            enabled=True,
            checkable=False
    ):
        self.name = name
        self.getter_dict = getter_dict
        if setter_dict == None:
            setter_dict = {}
        self.setter_dict = setter_dict
        self.editable = editable
        self.selectable=selectable
        self.enabled=enabled
        self.checkable=checkable
        self._init_qt_flag()
    
    def _init_qt_flag(self):
        self.qt_flag = Qt.NoItemFlags
        if self.editable:
            self.qt_flag |= Qt.ItemIsEditable
        if self.selectable:
            self.qt_flag |= Qt.ItemIsSelectable
        if self.enabled:
            self.qt_flag |= Qt.ItemIsEnabled
        if self.checkable:
            self.qt_flag |= Qt.ItemIsUserCheckable
    
    def get(self, data_obj, role):
        getter_func = self.getter_dict.get(role)
        if getter_func != None:
            return getter_func(data_obj)
        return None
    
    def set(self, data_obj, value,  role):
        setter_func = self.setter_dict.get(role)
        if setter_func != None:
            isset = setter_func(data_obj, value)
            return isset
        else:
            return False

class ConfigurableTableModel(QtCore.QAbstractTableModel):
    
    def __init__(self, data_list, column_list):
        super(ConfigurableTableModel, self).__init__()
        self._data = data_list
        self.columns = column_list
    
    def rowCount(self, parent=QModelIndex()):
        return len(self._data)
    
    def columnCount(self, parent=QModelIndex()):
        return len(self.columns)
    
    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self.columns[section].name
    
    def flags(self, index):
        col = index.column()
        return self.columns[col].qt_flag
    
    def data(self, index, role=Qt.DisplayRole):
        row, col = index.row(), index.column()
        data_obj = self._data[row]
        col_obj = self.columns[col]
        return col_obj.get(data_obj, role)
    
    def setData(self, index, value, role=Qt.EditRole):
        row, col = index.row(), index.column()
        data_obj = self._data[row]
        col_obj = self.columns[col]
        try:
            col_obj.set(data_obj, value, role)
            return True
        except Exception as e:
            print(e)
            return False

class MainTableRow(object):
    
    def __init__(self, data, name="horse with no name"):
        self.data = data
        self.name = name
        self.type_id = "misc"
        self.widgets = {}
    
    def on_double_click(self):
        pass

class SpectraRow(MainTableRow):
    
    def __init__(self, data, name="some spectra"):
        super(SpectraRow, self).__init__(data, name)
        self.type_id = "spectra"
    
    def on_double_click(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(len(self.data)):
            ax.plot(self.data[i].wv, self.data[i].flux, c="b")
            ax.plot(self.data[i].wv, self.data[i].norm, c="g")
        plt.show()

class LineListRow(MainTableRow):
    
    def __init__(self, data, name="some line list"):
        super(LineListRow, self).__init__(data, name)
        self.type_id = "line list"

class FeaturesRow(MainTableRow):
    
    def __init__(self, data, name="some features", parent_widget=None):
        super(FeaturesRow, self).__init__(data, name)
        self.type_id = "features"
        self.parent_widget = parent_widget
        self.widget = None
    
    def on_double_click(self):
        features = self.data
        fw = tmbg.widgets.FeatureFitWidget(features, 0, self.parent_widget)
        self.fit_widget = fw
        self.fit_widget.show()

class MainTableModel(QtCore.QAbstractTableModel):
    
    def __init__(self, rows=None):
        super(MainTableModel, self).__init__()
        if rows == None:
            rows = []
        self.rows = rows
        self.header_text = ["----------object name--------", "------type-------"]
    
    def rowCount(self, parent=QModelIndex()):
        return len(self.rows)
    
    def columnCount(self, parent=QModelIndex()):
        return 2
    
    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self.header_text[section]
    
    def addRow(self, row):
        if isinstance(row, MainTableRow):
            nrows = self.rowCount()
            self.beginInsertRows(QModelIndex(), nrows, nrows)
            self.rows.append(row)
            self.endInsertRows()
        else:
            raise ValueError
    
    def addRows(self, rows):
        nrows = self.rowCount()
        nrows_add = len(rows)
        self.beginInsertRows(QModelIndex(), nrows, nrows+nrows_add)
        self.rows.extend(rows)
        self.endInsertRows()
    
    def flags(self, index):
        fl = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        if index.column() == 0:
            fl |= Qt.ItemIsEditable
        return fl 
    
    def data(self, index, role=Qt.DisplayRole):
        row, col = index.row(), index.column()
        if col == 0:
            return self.rows[row].name
        elif col == 1:
            return self.rows[row].type_id
    
    def setData(self, index, value, role=Qt.EditRole):
        row, col = index.row(), index.column()
        if role == Qt.EditRole:
            if row == 0:
                self.rows[row].name = value
                return True
        return False

class NameTypeTableModel(QtCore.QAbstractTableModel):
    
    def __init__(self):
        super(NameTypeTableModel, self).__init__()
        self.names = []
        self._data = []
        self.types = []
        self.headers = ["-------------------object name--------------", "---------------type---------------"]
    
    def addItem(self, name, type_, data):
        nrows = self.rowCount()
        self.beginInsertRows(QModelIndex(), nrows, nrows)
        self.names.append(name)
        self.types.append(type_)
        self._data.append(data)
        self.endInsertRows()

    def internalData(self, row):
        return self._data[row]
    
    def rowCount(self, parent=QModelIndex()):
        return len(self._data)
    
    def columnCount(self, parent=QModelIndex()):
        return 2
    
    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self.headers[section]
    
    def flags(self, index):
        fl = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        if index.column() == 0:
            fl |= Qt.ItemIsEditable
        return fl 
    
    def data(self, index, role=Qt.DisplayRole):
        row, col = index.row(), index.column()
        if col == 0:
            return self.names[row]
        elif col == 1:
            return self.types[row]
    
    def setData(self, index, value, role=Qt.EditRole):
        row, col = index.row(), index.column()
        if role == Qt.EditRole:
            if row == 0:
                self.names[row] = value


class TreeNode(object):
    
    def __init__(self, name, obj, parent_item, max_depth=100, depth=0):
        self.name = name
        self._obj = obj
        self.parent_item = parent_item
        self.depth = depth
        self.max_depth = max_depth
        self.depth
        self._children=[]
        self._children_explored = False
    
    @property
    def options(self):
        if not self._children_explored:
            self.refresh_children()
        return self._children
    
    def refresh_children(self):
        self._children = []
        if self.depth >= self.max_depth:
            return
        if hasattr(self._obj, "__dict__"):
            sub_names = list(self._obj.__dict__.keys())
            for key in sub_names:
                if key[0] == "_":
                    continue
                val = self._obj.__dict__[key]
                self._children.append(TreeNode(key, val, self, self.max_depth, self.depth+1))
        elif isinstance(self._obj, dict):
            for key in self._obj:
                self._children.append(TreeNode(str(key), self._obj[key], self, self.max_depth, self.depth+1))
        elif isinstance(self._obj, list):
            #TODO: make a ... child for long lists
            list_max_n = min(100, len(self._obj))
            for key in range(list_max_n):
                self._children.append(TreeNode(str(key), self._obj[key], self, self.max_depth, self.depth+1))
        self._children_explored = True
    
    def __repr__(self):
        return "node:"+repr(self._obj)
    
    def __len__(self):
        return len(self.options)


def attribute_node_generator(obj):
    pass

class ObjectTree(QtCore.QAbstractItemModel):
    
    def __init__(self, obj, max_depth=15):
        super(ObjectTree, self).__init__()
        self.root_item = TreeNode("root node", obj, None, max_depth=max_depth)
    
    def columnCount(self, parent):
        return 2
    
    def headerData(self, section, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            if section == 0:
                return "Name"
            if section == 1:
                return "Value"
        return None
    
    def data(self, index, role):
        if not index.isValid():
            return None
        
        if role == Qt.DisplayRole:
            item = index.internalPointer()
            col = index.column()
            #row = index.row()
            if item is None:
                item = self.root_item
            if col == 0:
                data_str = item.name
            elif col == 1:
                try:
                    data_str = repr(item._obj)
                except Exception as e:
                    data_str = "repr failed with error {}".format(e)
            return data_str
        
        return None
    
    def flags(self, index):
        if not index.isValid():
            return Qt.NoItemFlags
        
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable
    
    def index(self, row, column, parent):
        if not self.hasIndex(row, column, parent):
            return QModelIndex()
        
        if not parent.isValid():
            parent_item = self.root_item
        else:
            parent_item = parent.internalPointer()
        
        child = parent_item.options[row]
        out_index = self.createIndex(row, column, child)
        return out_index
    
    def parent(self, index):
        if not index.isValid():
            return QModelIndex()
        
        child = index.internalPointer()
        parent = child.parent_item
        
        if parent == self.root_item:
            return QModelIndex()
        
        return self.createIndex(parent.options.index(child), 0, parent)
    
    def rowCount(self, parent):
        internal_pointer = parent.internalPointer()
        if internal_pointer is None:
            parent_node = self.root_item
        else:
            parent_node = internal_pointer
        nrows = len(parent_node)
        return nrows

if __name__ == "__main__":
    line_obj ,= plt.plot(list(range(10)))
    #import numpy as np
    #import thimbles as tmb
    #spec = tmb.Spectrum(np.arange(100), np.arange(100))
    
    #build a QApplication
    qap = QtGui.QApplication([])
    
    #build the model tree
    top_node = ObjectTree(line_obj)
    
    #make a view and set its model
    qtv = QtGui.QTreeView()
    qtv.setModel(top_node)
    
    #run
    qtv.show()
    qap.exec_()
