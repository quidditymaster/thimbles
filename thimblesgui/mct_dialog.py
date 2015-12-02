from thimblesgui import QtGui, Qt
from thimblesgui.expressions import PythonExpressionLineEdit
import thimbles as tmb
from thimbles.analysis import component_templates
from thimbles.contexts import contextualizers, ContextualizationEngine

global_ce = ContextualizationEngine(
    table_spec=tmb.analysis.SharedParameterSpace,
    filter_factory=lambda q: q.filter(tmb.analysis.SharedParameterSpace.name == "global"),
)
contextualizers.register("global", global_ce)

class ModelComponentTemplateApplicationDialog(QtGui.QDialog):
    
    def __init__(
            self,
            parent,
    ):
        super().__init__(parent=parent)
        layout = QtGui.QGridLayout()
        self.setLayout(layout)
                
        component_dd = QtGui.QComboBox(parent=self)
        component_label = QtGui.QLabel("component")
        layout.addWidget(component_label, 0, 0, 1, 1)
        layout.addWidget(component_dd,    0, 1, 1, 2)
        self.component_dd = component_dd
        comp_opts = component_templates.component_options
        for component_name in comp_opts:
            component_dd.addItem(component_name)
        component_dd.currentIndexChanged.connect(self.update_template_dd)
        
        template_dd = QtGui.QComboBox(parent=self)
        template_label = QtGui.QLabel("template")
        layout.addWidget(template_label, 1, 0, 1, 1)
        layout.addWidget(template_dd,    1, 1, 1, 2)
        self.template_dd = template_dd
        self.update_template_dd(0)
        
        context_spine_dd = QtGui.QComboBox(parent=self)
        context_spine_label = QtGui.QLabel("modeling spine")
        layout.addWidget(context_spine_label, 2, 0, 1, 1)
        layout.addWidget(context_spine_dd,    2, 1, 1, 2)
        self.context_spine_dd = context_spine_dd
        for cspine_name in contextualizers.spines:
            context_spine_dd.addItem(cspine_name)
        
        apply_btn = QtGui.QPushButton("Apply")
        apply_btn.clicked.connect(self.on_apply)
        layout.addWidget(apply_btn, 3, 2, 1, 1)
    
    @property
    def current_component_name(self):
        cidx = self.component_dd.currentIndex()
        return self.component_dd.itemData(cidx, 0)
    
    @property
    def current_template_name(self):
        tidx = self.template_dd.currentIndex()
        return self.template_dd.itemData(tidx, 0)
    
    @property
    def current_spine_name(self):
        sidx = self.context_spine_dd.currentIndex()
        return self.context_spine_dd.itemData(sidx, 0)
    
    def update_template_dd(self, component_index):
        self.template_dd.clear()
        comp_opts = component_templates.component_options
        templates = comp_opts[self.current_component_name]
        for template_name in templates:
            self.template_dd.addItem(template_name)
    
    def on_apply(self):
        ccomp = self.current_component_name
        ctemp = self.current_template_name
        
        template = component_templates.component_options[ccomp][ctemp]
        #collection_instances = self.spine_pele.value
        context_engine = contextualizers.spines[self.current_spine_name]
        template.search_and_apply(context_engine)
        self.accept()

