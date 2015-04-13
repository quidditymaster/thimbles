

from thimblesgui import QtCore, QtGui, Qt
QModelIndex = QtCore.QModelIndex

class Node(object):
    _children = None
    
    def __init__(self, name, obj, parent, node_generator=None):
        self.name = name
        self.obj = obj
        self.parent = parent
        if node_generator is None:
            node_generator = attribute_node_generator
        self.node_generator = node_generator
    
    def refresh_children(self):
        self._children = None
    
    @property
    def children(self):
        if self._children is None:
            self._children = self.node_generator(self)
        return self._children
    
    def __len__(self):
        return len(self.children)


def attribute_node_generator(parent_node):
    obj = parent_node.obj
    child_pairs = []
    if hasattr(obj, "__dict__"):
        sub_names = list(obj.__dict__.keys())
        for key in sub_names:
            if key[0] == "_":
                continue
            val = obj.__dict__[key]
            child_pairs.append((key, val))
    elif isinstance(obj, list):
        for idx in range(len(obj)):
            child_pairs.append((idx, obj[idx]))
    children = []
    for name, child_obj in child_pairs:
        node = Node(
            name=name, 
            obj=child_obj, 
            parent=parent_node,
            node_generator=attribute_node_generator
        )
        children.append(node)
    return children


class NodeColumn(object):
    
    def __init__(self, generator, formatter=None, qt_flags=None, setter=None, column_name=""):
        self.generator = generator
        if formatter is None:
            formatter = "{}"
        self.formatter = formatter
        if qt_flags is None:
            qt_flags = Qt.ItemIsSelectable
            qt_flags |= Qt.ItemIsEnabled
        self.qt_flags = qt_flags
        self.setter=setter
        self.column_name = column_name
    
    def get(self, node, role):
        if role == Qt.DisplayRole:
            col_obj = self.generator(node)
            return self.formatter.format(col_obj)
        return None
    
    def set(self, node, value, role):
        if role == Qt.EditRole:
            isset = self.setter(node, value)
            return isset


class MappedTreeModel(QtCore.QAbstractItemModel):
    
    def __init__(
            self, 
            root_object,
            columns,
            node_generator=None, 
            root_name="root"
    ):
        super(MappedTreeModel, self).__init__()
        self.columns = columns
        self.root_item = Node(
            name=root_name, 
            obj=root_object,
            parent=None,
            node_generator=node_generator
        )
    
    def rowCount(self, parent):
        pointer = parent.internalPointer()
        if pointer is None:
            parent_node = self.root_item
        else:
            parent_node = pointer
        nrows = len(parent_node)
        return nrows
    
    def columnCount(self, parent):
        return len(self.columns)
    
    def data(self, index, role):
        if not index.isValid():
            return None
        
        node = index.internalPointer()
        col_idx = index.column()
        column = self.columns[col_idx]
        return column.get(node, role)
    
    def parent(self, index):
        if not index.isValid():
            return QModelIndex()
        
        child = index.internalPointer()
        parent = child.parent
        if parent == self.root_item:
            return QModelIndex()
        child_row = parent.children.index(child)
        return self.createIndex(child_row, 0, parent)
    
    def index(self, row, column, parent):
        if not self.hasIndex(row, column, parent):
            return QModelIndex()
        
        if not parent.isValid():
            parent_item = self.root_item
        else:
            parent_item = parent.internalPointer()
        
        child=parent_item.children[row]
        out_index = self.createIndex(row, column, child)
        return out_index
    
    def flags(self, index):
        if not index.isValid():
            return Qt.NoItemFlags
        col = index.column()
        return self.columns[col].qt_flags
    
    def headerData(self, section, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.columns[section].column_name
        return None
    

if __name__ == "__main__":
    
    qap = QtGui.QApplication([])
    
    class Doot(object):
        pass
    d1 = Doot()
    d2 = Doot()
    d1.blam = d2
    d2.kaboom = d1
    rand_obj = [3, 5, "a", d1]
    
    def node_obj(node):
        return node.obj
    
    def node_name(node):
        return node.name
    
    name_column = NodeColumn(generator=node_name)
    repr_column = NodeColumn(generator=node_obj)
    
    mtm = MappedTreeModel(
        root_object=rand_obj,
        columns = [name_column, repr_column],
    )
    
    tview = QtGui.QTreeView()
    tview.setModel(mtm)
    
    tview.show()
    
    qap.exec_()
        
