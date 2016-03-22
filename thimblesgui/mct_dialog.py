from thimblesgui import QtGui, Qt
from thimblesgui.expressions import PythonExpressionLineEdit
import thimbles as tmb
from thimbles.analysis import component_templates
from thimbles.contexts import model_spines as contextualizers#, ContextualizationEngine

class ModelComponentTemplateApplicationDialog(QtGui.QDialog):
    
    def __init__(
            self,
            parent,
    ):
        super().__init__(parent=parent)
        layout = QtGui.QGridLayout()
        self.setLayout(layout)
        
        context_dd = QtGui.QComboBox(parent=self)
        context_label = QtGui.QLabel("context")
        layout.addWidget(context_label, 0, 0, 1, 1)
        layout.addWidget(context_dd,    0, 1, 1, 2)
        self.context_dd = context_dd
        comp_opts = component_templates.spines
        for component_name in comp_opts:
            context_dd.addItem(component_name)
        context_dd.currentIndexChanged.connect(self.update_template_dd)
        
        template_dd = QtGui.QComboBox(parent=self)
        template_label = QtGui.QLabel("template name")
        layout.addWidget(template_label, 1, 0, 1, 1)
        layout.addWidget(template_dd,    1, 1, 1, 2)
        self.template_dd = template_dd
        self.update_template_dd(0)
        
        apply_btn = QtGui.QPushButton("Apply")
        apply_btn.clicked.connect(self.on_apply)
        layout.addWidget(apply_btn, 2, 2, 1, 1)
    
    @property
    def current_spine_name(self):
        cidx = self.context_dd.currentIndex()
        return self.context_dd.itemData(cidx, 0)
    
    @property
    def current_template_name(self):
        tidx = self.template_dd.currentIndex()
        return self.template_dd.itemData(tidx, 0)
    
    def update_template_dd(self, component_index):
        self.template_dd.clear()
        comp_opts = component_templates.spines
        templates = comp_opts[self.current_spine_name]
        for template_name in templates:
            self.template_dd.addItem(template_name)
    
    def on_apply(self):
        cspine = self.current_spine_name
        ctemp = self.current_template_name
        
        template = component_templates.spines[cspine][ctemp]
        #collection_instances = self.spine_pele.value
        context_engine = contextualizers.spines[cspine]
        template.search_and_apply(context_engine)
        self.accept()

