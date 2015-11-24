
from thimblesgui import QtGui, Qt
from thimblesgui.expressions import PythonExpressionLineEdit

from sqlalchemy.orm.exc import MultipleResultsFound, NoResultFound
import thimbles as tmb

class ContextualizationRegistry(object):
    
    def __init__(self):
        self.contextualizers = []
        self._gui_connected = False
    
    def connect_gui(self, main_window):
        self.main_window = main_window
        self._gui_connected = True
        for contextualizer in self.contextualizers:
            contextualizer.connect_gui(self.main_window)
    
    def register(self, contextualizer):
        self.contextualizers.append(contextualizer)
        if self._gui_connected:
            contextualizer.connect_gui(self.main_window)

contextualizations = ContextualizationRegistry()

class ContextualizationEngine(object):
    gui = None
    
    def __init__(
            self,
            context_class,
            filter_generator,
            object_creation_dialog,
            auto_create=True,
    ):
        self.context_class = context_class
        self.filter_generator = filter_generator
        self.object_creation_dialog = object_creation_dialog
        contextualizations.register(self)
    
    def __call__(self, context_tag):
        query = self.gui.db.query(self.context_class)
        query = self.filter_generator(query, context_tag)
        try:
            result = query.one()
        except MultipleResultsFound:
            print("context filter not specific enough")
        except NoResultFound:
            result = self.object_creation_dialog.get_new(self.gui)
        return result
    
    def connect_gui(self, gui):
        self.gui = gui


#TODO: different behaviors for multiple and no matching.
class MultipleMatchingContextDialog(QtGui.QDialog):
    
    def __init__(
            self,
            creation_dialog,
            parent,
    ):
        super().__init__(parent=parent)        
        PythonExpressionLineEdit()


class NoMatchingContextDialog(QtGui.QDialog):
    
    def __init__(
            self,
            creation_dialog,
            parent,
    ):
        super().__init__(parent=parent)        
        PythonExpressionLineEdit()
