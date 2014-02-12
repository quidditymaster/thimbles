
import matplotlib
matplotlib.use('Qt4Agg')
try: 
    from PySide.QtCore import *
    from PySide.QtGui import *
    matplotlib.rcParams['backend.qt4'] = 'PySide'
except ImportError:
    from PyQt4.QtCore import *
    from PyQt4.QtGui import *
    matplotlib.rcParams['backend.qt4'] = 'PyQt4'
import matplotlib.pyplot as plt

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure

from widgets import *

class Column(object):
    
    def __init__(self, 
                 name, 
                 getter_dict,
                 setter_dict=None,
                 editable=False, 
                 selectable=True,
                 enabled=True,
                 checkable=False):
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

class ConfigurableTableModel(QAbstractTableModel):
    
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
            print e
            return False

class MainTableRow(object):
    
    def __init__(self, data, name="horse with no name"):
        self.data = data
        self.name = name
        self.type_id = "misc"
    
    def on_double_click(self):
        pass

class SpectraRow(MainTableRow):
    
    def __init__(self, data, name="some spectra"):
        super(SpectraRow, self).__init__(self, data, name)
        self.type_id = "spectra"
    
    def on_double_click(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(len(self.data)):
            ax.plot(self.data[i].wv, self.data[i].flux, c="b")
        plt.show()

class LineListRow(MainTableRow):

    def __init__(self, data, name="some line list"):
        super(LineListRow, self).__init__(self, data, name)
        self.type_id = "line list"

class FeaturesRow(MainTableRow):
    
    def __init__(self, data, name="some features"):
        super(FeaturesRow, self).__init__(self, data, name)
        self.type_id = "features"
        self.widget = None
    
    def on_double_click(self):
        if self.widget == None:
            spec, features, feat_spec_idxs = self.data
            fw = FeatureFitWidget(spec, features, 0, feat_spec_idxs, 3.0, None)
            self.widget = fw
            self.widget.show()
        else:
            self.widget.show()
    
class MainTableModel(QAbstractTableModel):
    
    def __init__(self, rows=None):
        super(MainTableModel, self).__init__()
        if rows == None:
            rows = []
        self.rows = rows
        self.header_text = ["----------object name--------", "------type-------"]
    
    def rowCount(self):
        return len(self.rows)
    
    def columnCount(self):
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

class NameTypeTableModel(QAbstractTableModel):
    
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

class TypedGroupableItemModel(QAbstractItemModel):
    
    def __init__(self, group_types=None):
        super(TypedGroupableItemModel, self).__init__()
        if group_types == None:
            group_types = []
        if "misc" not in [gt.name for gt in group_types]:
            group_types.append(GroupType("misc", []))
        self.group_types = {}
        self.groups = {}
        for gt in group_types:
            self.group_types[gt.name] = gt
            self.groups[gt.name] = []
    
    def rowCount(self, parent=QModelIndex()):
        if parent==QModelIndex():
            return len(self.group_types)
        ip = parent.internalPointer()
        return ip.rowCount()
    
    def columnCount(self, parent=QModelIndex()):
        return 2
    
    def add_group(self, group, type_name):
        self.groups[type_name].append(group)

class GroupType(object):
    
    def __init__(self, type_name, columns):
        self.name = type_name
        self.columns = columns

class ItemGroup(object):
    
    def __init__(self, name, items):
        self.name = group_name
        self.items = items
    
    def rowCount(self):
        return len(self.items)

class GroupItem(object):
    
    def __init__(self, name, data):
        self.name = name
        self.data = data

class GroupableItemModel(QAbstractItemModel):
    
    def __init__(self, data, groups):
        self._data = data
        self._groups = groups
    
    def rowCount(self, parent=QModelIndex()):
        if parent == QModelIndex():
            return len(self._groups)
        else:
            group_row = parent.row()
            return self._groups[group_row].rowCount()
    
    def columnCount(self, parent=QModelIndex()):
        return 1
    
    def index(row, column, parent=QModelIndex()):
        if not self.hasIndex(row, column, parent):
            return QModelIndex()
        if parent == QModelIndex():
            return self.createIndex(row, column, QModelIndex())
        if childItem:
            pass

    #def headerData(self, section, orientation, role):
    #    if role == Qt.DisplayRole:
    #        if orientation == Qt.Horizontal:
    #            return self.columns[section].name
    #
    #def flags(self, index):
    #    col = index.column()
    #    return self.columns[col].qt_flag
    #
    def data(self, index, role=Qt.DisplayRole):
        parent = index.parent()
        row, col = index.row(), index.column()
        if parent == QModelIndex():
            return self._groups[row].name
        else:
            parent_row = index.parent().row()
            return col_obj.get(data_obj, role)
    
    def setData(self, index, value, role=Qt.EditRole):
        row, col = index.row(), index.column()
        data_obj = self._data[row]
        col_obj = self.columns[col]
        try:
            col_obj.set(data_obj, value, role)
        except:
            return False

