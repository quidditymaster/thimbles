from widgets import *
from models import *
import matplotlib


class LineListView(QTableView):
    
    def __init__(self, parent):
        super(LineListView, self).__init__(parent)

    def minimumSizeHint(self):
        return QSize(500, 150)

class NameTypeTableView(QTableView):
    
    def __init__(self, parent):
        super(NameTypeTableView, self).__init__(parent)

    def minimumSizeHint(self):
        return QSize(500, 500)
