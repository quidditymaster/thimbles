from widgets import *
from models import *
import matplotlib


class StarTreeView(QTreeView):
    
    def __init__(self, parent=None):
        QTreeView.__init__(self, parent)


class LineListView(QTableView):
    
    def __init__(self, parent):
        super(LineListView, self).__init__(parent)
