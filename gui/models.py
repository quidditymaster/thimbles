
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

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure


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
        except:
            return False

class LineListView(QTableView):
    
    def __init__(self, parent=None):
        super(LineListView, self).__init__(parent)

