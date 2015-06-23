
from thimbles.sqlaimports import *
from thimbles.thimblesdb import ThimblesTable, Base
from thimbles.modeling import ParameterGroup, Parameter
from sqlalchemy.orm.collections import collection


class NamedParameters(ParameterGroup):
    
    def __init__(self):
        self._aliases = []
        self.groups = {}
        self.param_to_index = {}
        self.parameters = []
    
    def __getitem__(self, index):
        return self.groups[index]
    
    def __len__(self):
        return len(self.parameters)
    
    def parameter_index(self, parameter):
        return self.param_to_index[parameter]
    
    @collection.appender
    def append(self, param_alias):
        pname = param_alias.name
        if param_alias._param_temp is None:
            param = param_alias.parameter
        else:
            param = param_alias._param_temp
        is_compound = param_alias.is_compound
        pgroup = self.groups.get(pname)
        if pgroup is None:
            if is_compound:
                self.groups[pname] = [param]
            else:
                self.groups[pname] = param
        else:
            if is_compound:
                self.groups[pname].append(param)
            else:
                print("WARNING: redundant non-compound InputAlias objects for model {} and parameter {}\n former alias is unreachable by name but will still show up in the .parameters collection".format(param_alias.model, param))
                self.groups[pname] = param
        self.param_to_index[param] = len(self.parameters)
        self.parameters.append(param)
        self._aliases.append(param_alias)
    
    @collection.remover
    def remove(self, param_alias):
        pname = param_alias.name
        param = param_alias.parameter
        if param_alias.is_compound:
            pgroup = self.groups[pname]
            pgroup.remove(param)
            if len(pgroup) == 0:
                self.groups.pop(pname)
        else:
            self.groups[pname].pop(pname)
        rem_pidx = self.param_to_index.pop(param)
        self.parameters.pop(rem_pidx)
        for param in self.parameters[rem_pidx:]:
            cidx = self.param_to_index[param]
            self.param_to_index[param] = cidx-1
        self._aliases.remove(param_alias)
    
    @collection.iterator
    def _iter_aliases(self):
        for alias in self._aliases:
            yield alias
    
    def __iter__(self):
        for p in self.parameters:
            yield p

class InputAlias(ThimblesTable, Base):
    _parameter_id = Column(Integer, ForeignKey("Parameter._id"))
    parameter = relationship("Parameter", back_populates="models")
    _model_id = Column(Integer, ForeignKey("Model._id"))
    model = relationship("Model", foreign_keys=_model_id, back_populates="inputs")
    name = Column(String)
    is_compound = Column(Boolean)
    
    _param_temp = None
    
    def __init__(self, name, model, parameter, is_compound=False):
        self.name = name
        #an assignment that does not trigger the back populate
        #so that when we assign to model we have access to the param
        self.is_compound=is_compound
        self._param_temp = parameter
        self.model = model
        self.parameter = parameter


class ParameterAlias(ThimblesTable, Base):
    _parameter_id = Column(Integer, ForeignKey("Parameter._id"))
    parameter = relationship("Parameter")
    _context_id = Column(Integer, ForeignKey("ParameterContext._id"))
    context = relationship("ParameterContext", foreign_keys=_context_id, back_populates="params")
    name = Column(String)
    is_compound = Column(Boolean)
    
    _param_temp = None
    
    def __init__(self, name, context, parameter, is_compound=False):
        self.name = name
        #an assignment that does not trigger the back populate
        #so that when we assign to model we have access to the param
        self.is_compound=is_compound
        self._param_temp = parameter
        self.context = context
        self.parameter = parameter


class ParameterContext(ThimblesTable, Base):
    params = relationship(
        "ParameterAlias",
        collection_class=NamedParameters,
    )
    
    def __getitem__(self, index):
        return self.params[index]

class HasParameterContext(object):
    
    @declared_attr
    def _context_id(self):
        return Column(Integer, ForeignKey("ParameterContext._id"))
    
    @declared_attr
    def context(self):
        return relationship("ParameterContext")
    
    def __init__(self, context_dict=None):
        self.context = ParameterContext()
        if context_dict is None:
            context_dict = {}
        for pname in context_dict:
            mapped_param = context_dict[pname]
            if isinstance(mapped_param, list):
                for param in mapped_param:
                    self.add_parameter(pname, param, is_compound=True)
            else:
                self.add_parameter(pname, mapped_param)
        
    
    def add_parameter(self, name, parameter, is_compound=False):
        p_alias = ParameterAlias(name=name, context=self.context, parameter=parameter, is_compound=is_compound)
    
    def __getitem__(self, index):
        val = self.context.params[index]
        return val


