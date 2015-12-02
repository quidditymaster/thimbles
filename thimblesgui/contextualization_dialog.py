
from thimblesgui import QtGui, Qt
from thimblesgui.expressions import PythonExpressionLineEdit

from sqlalchemy.orm.exc import MultipleResultsFound, NoResultFound
import thimbles as tmb

class ContextualizationRegistry(object):
    global_contexts = {}
    
    def __init__(self):
        self.contextualizers = []
        self._db_connected = False
    
    def connect_db(self, db):
        self.db = db
        self._db_connected = True
        for contextualizer in self.contextualizers:
            contextualizer.connect_db(self.main_window)
    
    def register_contextualizer(self, contextualizer):
        self.contextualizers.append(contextualizer)
        if self._db_connected:
            contextualizer.connect_db(self.main_window)
    
    def register_global_context(self, context_name, context):
        self.global_contexts[context_name] = context


contextualization_registry = ContextualizationRegistry()

class ContextualizationEngine(object):
    db = None
    
    def __init__(
            self,
            context_spine,
            filter_generator,
            global_contexts=None,
            on_none_found=None,
            on_multiple_found=None,
    ):
        self.context_spine = context_spine
        self.filter_generator = filter_generator
        self.object_creation_dialog = object_creation_dialog
        self.on_none_found = on_none_found
        self.on_multiple_found=on_multiple_found
        #contextualizations.register_contextualizer(self)
    
    def find(self, context_tag):
        query = self.db.query(self.context_spine)
        query = self.filter_generator(query, context_tag)
        try:
            result = query.one()
        except MultipleResultsFound:
            if not self.on_multiple_found is None:
                self.on_multiple_found()
        except NoResultFound:
            if not self.on_none_found is None:
                self.on_none_found()
        return result
    
    def expand(self, backbone_instance):
        pass
    
    def connect_db(self, db):
        self.db = db


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
