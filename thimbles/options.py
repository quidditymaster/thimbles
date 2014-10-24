import sys
import pyparsing
from thimbles import workingdataspace as wds
import os

class OptionSpecificationError(Exception):
    pass

config_path = os.environ.get("THIMBLESCONFIGPATH", os.path.join(os.environ["HOME"], ".config", "thimbles", "config.txt"))
config_strings = {}
if os.path.isfile(config_path):
    lines = open(config_path, "r").readlines()
    for line in lines:
        spl = line.split()
        if spl[0][0] == "#":
            continue
        #TODO: instead of splitting on white space and rejoining just find first white space
        config_strings[spl[0]] = " ".join(spl[1:])

class OptionTree(object):
    config_strings = config_strings
    
    def __init__(self):
        self.option_path = ""
        self.options = {}
    
    def traverse_tree(self, index):
        if not isinstance(index, basestring):
            raise ValueError("option index should be a string not {}".format(type(index)))
        spl = index.split(".")
        cur_opt = self
        for opt_name in spl:
            cur_opt = getattr(cur_opt, opt_name)        
        return cur_opt
    
    def __getitem__(self, index):
        return self.options[index].value
    
    def __setitem__(self, index, value):
        self.options[index].value = value
    
    def parse_options(self):
        argv = sys.argv[1:] #first value is program name
        arg_dict = {}
        val_idx = 0
        while val_idx < len(argv):
            val = argv[val_idx]
            if "--" == val[:2]:
                arg_dict[val[2:]] = argv[val_idx+1]
                val_idx +=1
            val_idx += 1
        for opt_path in self.options:
            if opt_path in arg_dict:
                self.options[opt_path].set_runtime_str(arg_dict[opt_path])

opts = OptionTree()

class Option(object):
    eval_ns = wds.__dict__
    
    def __init__(self, 
                 name, 
                 parent=None,  
                 envvar=None, 
                 runtime_str=None, 
                 help="no help string specified",
                 use_cached=True, 
                 option_tree=opts,
                 **kwargs):
        """an option which can be variously specified either by a default value
        in the code, set using the value of a environment variable, evaluated
        dynamically at runtime using a string read from a config file or 
        specified on the command line
        """
        default = None
        self.default_specified = False
        self.runtime_str = None
        self.use_cached = use_cached
        if "default" in kwargs:
            default = kwargs["default"]
            self.default_specified = True
        self.default = default
        self._value = None
        self._valuated = False
        if self.default_specified:
            self._value = default
            self._valuated = True
        self.envvar = envvar
        self.try_load_envvar()
        self.help = help
        self.option_tree = option_tree
        self.name = name
        self.register_option(name, parent)
    
    def try_load_envvar(self):
        if not self.envvar is None:
            if self.envvar in os.environ:
                self.set_runtime_str(os.environ[self.envvar])
    
    def set_runtime_str(self, value):
        if not isinstance(value, basestring):
            raise TypeError("runtime_str must be of type string not type {}".format(type(value)))
        self.runtime_str = value
        self._valuated = False
    
    def evaluate(self):
        if not self.runtime_str is None:
            res = eval(self.runtime_str, self.eval_ns)
            self._value = res
            self._valuated = True
        else:
            raise OptionSpecificationError("runtime string not specified")
        return res
    
    @property
    def value(self):
        if (not self._valuated) or (not self.use_cached):
            self.evaluate()
        return self._value
    
    def register_option(self, name, parent):
        if not isinstance(name, basestring):
            raise ValueError("option name must be a string! not {}".format(type(name)))
        if parent is None:
            parent = self.option_tree
        elif isinstance(parent, basestring):
            parent = opts.traverse_tree(parent)
        if hasattr(parent, name):
            raise OptionSpecificationError("option with name {} already specified!".format(self.name))
        if not (isinstance(parent, Option) or isinstance(parent, OptionTree)):
            raise ValueError("parent option value of type {} not understood".format(type(parent)))
        setattr(parent, name, self)
        self.parent = parent
        self.option_tree.options[self.option_path] = self
    
    @property
    def option_path(self):
        parent_path = self.parent.option_path
        if parent_path == "":
            return self.name
        else:
            return "{}.{}".format(parent_path, self.name)
    
    def __repr__(self):
        return "Option {}, value={}".format(self.option_path, repr(self.value))
