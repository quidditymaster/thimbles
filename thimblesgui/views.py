from thimblesgui.models import ObjectTree
import thimbles as tmb

from thimblesgui import QtGui, QtWidgets, QtCore, Qt

class LineListView(QtWidgets.QTableView):
    
    def __init__(self, parent):
        super(LineListView, self).__init__(parent)
    
    def minimumSizeHint(self):
        return QtCore.QSize(500, 150)

class NameTypeTableView(QtWidgets.QTableView):
    
    def __init__(self, parent):
        super(NameTypeTableView, self).__init__(parent)
    
    def minimumSizeHint(self):
        return QtCore.QSize(500, 500)

class RepresentationEngine(object):
    
    def __init__(self):
        pass

class ObjectTreeWidget(QtWidgets.QWidget):
    
    def __init__(self, obj, parent):
        super(ObjectTreeWidget, self).__init__(parent=parent)
        
        self.layout = QtWidgets.QGridLayout()
        self.setLayout(self.layout)
        self.tree_model = ObjectTree(obj)
        
        self.tree_view = QtWidgets.QTreeView()
        self.tree_view.setModel(self.tree_model)
        self.layout.addWidget(self.tree_view, 0, 0, 1, 2)
        self.tree_view.doubleClicked.connect(self.on_double_click)
        
        self.refresh_btn = QtWidgets.QPushButton("refresh")
        self.refresh_btn.clicked.connect(self.on_refresh)
        self.layout.addWidget(self.refresh_btn, 1, 1, 1, 1)
    
    def minimumSizeHint(self):
        return QtCore.QSize(300, 500)
    
    def on_refresh(self):
        self.tree_model.root_item.refresh_children()
        self.tree_model.reset()
    
    def on_double_click(self, index):
        obj = index.internalPointer()._obj
        if isinstance(obj, list):
            if isinstance(obj[0], tmb.Spectrum):
                import matplotlib.pyplot as plt 
                plt.plot(obj[0].wv, obj[0].flux)
                plt.show()
