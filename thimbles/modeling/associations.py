
from thimbles.sqlaimports import *
from thimbles.thimblesdb import ThimblesTable, Base
from sqlalchemy.orm.collections import collection


class NamedParameters(object):
    
    def __init__(self):
        self._aliases = []
        self.groups = {}
        self.param_to_index = {}
        self.parameters = []
    
    def get_pvec(self, pdict=None):
        """convert from a dictionary of parameter mapped values to
        a parameter vector in this embedding space.
        
        """
        parameters = self.parameters
        if pdict is None:
            pdict = {}
        pvals = []
        for p in parameters:
            pval = pdict.get(p)
            if pval is None:
                pval = p.get()
            flat_pval = np.asarray(pval).reshape((-1,))
            pvals.append(flat_pval)
        return np.hstack(pvals)
    
    def set_pvec(self, pvec, attr=None, as_delta=False):
        parameters = self.parameters
        pshapes = [p.shape for p in parameters]
        nvals = [flat_size(pshape) for pshape in pshapes]
        break_idxs = np.cumsum(nvals)[:-1]
        if as_delta:
            pvec = pvec + self.get_pvec()
        flat_params = np.split(pvec, break_idxs)
        if attr is None:
            for p_idx in range(len(parameters)):
                param = parameters[p_idx]
                pshape = pshapes[p_idx]
                flat_val = flat_params[p_idx]
                if pshape == tuple():
                    to_set = float(flat_val)
                else:
                    to_set = flat_val.reshape(pshape)
                param.set(to_set)
        elif isinstance(attr, str):
            for p_idx in range(len(parameters)):
                param = parameters[p_idx]
                pshape = pshapes[p_idx]
                flat_val = flat_params[p_idx]
                if pshape == tuple():
                    setattr(param, attr, float(flat_val))
                else:
                    setattr(param, attr, flat_val.reshape(pshape))
        else:
            raise ValueError("attr must be a string if set")
    
    def get_pdict(
            self, 
            value_replacements=None, 
            attr=None, 
            name_as_key=False
    ):
        
        if free_only:
            parameters = self.free_parameters
        else:
            parameters = self.parameters
        if name_as_key:
            keys = [p.name for p in parameters]
        else:
            keys = parameters
        if attr is None:
            values = [p.get() for p in parameters]
        else:
            values = [getattr(p, attr) for p in parameters]
        pdict = dict(list(zip(keys, values))) 
        return pdict
    
    def set_pdict(self, val_dict, attr=None):
        for p in val_dict:
            if attr is None:
                p.set(val_dict[p])
            else:
                setattr(p, attr, val_dict[p])
    
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
                raise ValueError("Redundant non-compound InputAlias objects for model {} and parameter {}\n".format(param_alias.model, param))
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
            self.groups.pop(pname)
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


class InformedContexts(object):
    _contexts = None
    
    def __init__(self):
        self._aliases = []
    
    @property
    def contexts(self):
        if self._contexts is None:
            self._contexts = [alias.context for alias in self._aliases]
        return self._contexts
    
    def append(self, param_alias):
        self._contexts = None
        self._aliases.append(param_alias)
    
    def remove(self, param_alias):
        self._contexts = None
        self._aliases.remove(param_alias)
    
    def __len__(self):
        return len(self._aliases)
    
    def __getitem__(self, index):
        return self.contexts[index]
    
    def __repr__(self):
        return "<InformedContexts: {}>".format(self.contexts)
    
    @collection.iterator
    def _iter_aliases(self):
        for alias in self._aliases:
            yield alias


class NamedContexts(object):
    _groups = None
    
    def __init__(self):
        self._aliases = []
    
    def __getitem__(self, index):
        return self.groups[index]
    
    def __len__(self):
        return len(self.contexts)
    
    @property
    def contexts(self):
        return [alias.context for alias in self._aliases]
    
    @property
    def groups(self):
        if self._groups is None:
            groups = {}
            for alias in self._aliases:
                pname = alias.name
                context = alias.context
                pgroup = groups.get(pname)
                if pgroup is None:
                    if alias.is_compound:
                        pgroup = [alias.context]
                    else:
                        pgroup = alias.context
                else:
                    if alias.is_compound:
                        pgroup.append(alias.context)
                    else:
                        print("Warning redundant non-compound context alias ignored")
                groups[pname] = pgroup
            self._groups = groups
        return self._groups
    
    @collection.appender
    def append(self, param_alias):
        self._groups = None
        self._aliases.append(param_alias)
    
    @collection.remover
    def remove(self, param_alias):
        self._groups = None
        self._aliases.remove(param_alias)
    
    @collection.iterator
    def _iter_aliases(self):
        for alias in self._aliases:
            yield alias


class ParameterAliasMixin(object):
    
    @declared_attr
    def _parameter_id(cls):
        return Column(Integer, ForeignKey("Parameter._id"))
    
    @declared_attr
    def parameter(cls):
        return relationship("Parameter")
    
    @declared_attr
    def name(cls):
        return Column(String)
    
    @declared_attr
    def is_compound(cls):
        return Column(Boolean)
    
    _param_temp = None
    
    def __init__(self, name, context, parameter, is_compound=False):
        self.name = name
        #print(name)
        #an assignment that does not trigger the back populate
        #so that when we assign to model we have access to the param
        self.is_compound=is_compound
        self._param_temp = parameter
        self.context = context
        self.parameter = parameter


class HasParameterContext(object):
    
    def __init__(self, context_dict=None):
        if context_dict is None:
            context_dict = {}
        if not isinstance(context_dict, dict):
            raise ValueError("expected context dictionary")
        for pname in context_dict:
            mapped_param = context_dict[pname]
            if isinstance(mapped_param, list):
                for param in mapped_param:
                    self.add_parameter(pname, param, is_compound=True)
            else:
                self.add_parameter(pname, mapped_param)
    
    
    def __getitem__(self, index):
        val = self.context[index]
        return val


