import sys
from thimbles import workingdataspace as wds
import os
from copy import copy

class OptionSpecificationError(Exception):
    pass

config_path = os.environ.get("THIMBLESCONFIGPATH", os.path.join(os.environ["HOME"], ".config", "thimbles", "config.txt"))

class OptionTree(object):
    
    def __init__(self, config_path=config_path):
        self.option_path = ""
        self.config_path=config_path
        self.options = {}
    
    @property
    def children(self):
        children = {}
        for poss_opt_key in self.__dict__:
            poss_opt = getattr(self, poss_opt_key)
            if isinstance(poss_opt, Option):
                children[poss_opt.name] = poss_opt.value 
        return children
    
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
        rt_str_dict = {}
        #populate from environment variables
        for option in self.options.values():
            if not option.envvar is None:
                if option.envvar in os.environ:
                    rt_str_dict[option] = os.environ[option.envvar]
        #parse the config file
        parent_opt_path = None
        sub_path = None
        if os.path.isfile(self.config_path):
            lines = open(config_path, "r").readlines()
            for line in lines:
                crun_str = None
                spl = line.split("#")[0].split()
                if len(spl) < 1:
                    continue
                elif spl[0][:2] == "--":
                    parent_opt_path = spl.pop(0)[2:]
                    sub_path = None
                if len(spl) > 0 and spl[0][:1] == "-":
                    sub_path = spl.pop(0)[1:]
                if len(spl) > 0:
                    crun_str = " ".join(spl)
                
                #figure out where we are in the option tree
                if sub_path is None:
                    cur_path = parent_opt_path
                else:
                    cur_path = "{}.{}".format(parent_opt_path, sub_path)
                cur_opt = self.traverse_tree(cur_path)
                if cur_opt.option_style == "flag":
                    rt_str_dict[cur_opt] = "True"
                elif not crun_str is None:
                    rt_str_dict[cur_opt] = crun_str
        #parse the command line arguments
        argv = copy(sys.argv[1:]) #first value is program name
        parent_opt_path = None
        sub_path = None
        while len(argv) > 0:
            crun_str = None
            if "--" == argv[0][:2]:
                parent_opt_path = argv.pop(0)[2:]
                sub_path = None
            elif len(argv) > 0 and argv[0][0] == "-":
                sub_path = argv.pop(0)[1:]
            elif len(argv) > 0:
                crun_str = argv.pop(0)
            #figure out where we are in the option tree
            if sub_path is None:
                cur_path = parent_opt_path
            else:
                cur_path = "{}.{}".format(parent_opt_path, sub_path)
            cur_opt = self.traverse_tree(cur_path)
            if cur_opt.option_style == "flag":
                rt_str_dict[cur_opt] = "True"
            elif not crun_str is None:
                rt_str_dict[cur_opt] = crun_str
        #set the runtime values
        for option in rt_str_dict:
            option.set_runtime_str(rt_str_dict[option])

opts = OptionTree()
wds.opts = opts

class Option(object):
    eval_ns = wds.__dict__
    
    def __init__(self, 
                 name, 
                 parent=None,
                 option_style=None,
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
        self.name = name
        self.option_style = option_style
        self.default_specified = False
        if self.option_style == "flag":
            if not "default" in kwargs:
                kwargs["default"] = True
        self.runtime_str = runtime_str
        self.use_cached = use_cached
        default = None
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
        self.help = help
        self.option_tree = option_tree
        self.register_option(name, parent)
    
    @property
    def children(self):
        children = {}
        for poss_opt_key in self.__dict__:
            poss_opt = getattr(self, poss_opt_key)
            if isinstance(poss_opt, Option):
                children[poss_opt.name] = poss_opt 
        return children
        
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
        if self.option_style == "parent_dict":
            children = self.children
            return {k:children[k].value for k in children}
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

#general behavior options
_help = "path to prepend to relative paths when searching for input data"
data_dir = Option("data_dir", default=os.getcwd(), help=_help)

_help = "path to prepend to relative paths when writing out files"
output_dir = Option("output_dir", default=os.getcwd(), help=_help)

#matplotlib options
_help = "parent option for setting matplotlib style related options"
mpl_style = Option("mpl_style", option_style="parent_dict", help=_help)
lw = Option(name="line_width", default=1.5, parent=mpl_style, help="default line width")

del _help