from thimblesgui import QtGui, QtCore, Qt
QModelIndex = QtCore.QModelIndex

import thimbles as tmb

class SelectionChannel(QtCore.QObject):
    changed = QtCore.pyqtSignal()
    obj = None
    
    def __init__(self):
        super(SelectionChannel, self).__init__()
    
    def select(self, obj):
        self.obj = obj
        self.changed.emit()


class GlobalSelection(QtCore.QObject):
    changed = QtCore.pyqtSignal(list)
    
    def __init__(self, channels):
        super(GlobalSelection, self).__init__()
        self.channels = {}
        for ch_name in channels:
            self.add_channel(ch_name)
    
    def add_channel(self, channel_name):
        self.channels[channel_name] = SelectionChannel()
            
    def __getitem__(self, index):
        return self.channels[index].obj
    
    def __setitem__(self, index, value):
        self.channels[index].select(value)
