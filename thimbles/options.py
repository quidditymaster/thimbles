import os, sys, re
from collections import OrderedDict
from copy import copy

import thimbles as tmb
from thimbles import workingdataspace as wds

class OptionSpecificationError(Exception):
    pass

class EvalError(Exception):
    pass


class OptionTree(object):
    eval_ns = wds.__dict__
    full_path_pattern = re.compile(r"--[A-z]")
    sub_path_pattern = re.compile(r"-[A-z]")
    
    def __init__(self, global_config_file=None, local_config_file=None):
        self.option_path = ""
        if global_config_file is None:
            default_config_dir = os.path.join(
                os.environ["HOME"], 
                ".config", 
                "thimbles"
            )
            global_config_dir = os.environ.get("THIMBLESCONFIGPATH", default_config_dir)
            global_config_file = os.path.join(global_config_dir, "tmb_config.py")            
        else:
            global_config_dir = os.path.dirname(global_config_file)
        self.global_config_dir = global_config_dir
        if not os.path.exists(global_config_dir):
            os.makedirs(global_config_dir)
        
        self.global_config_file=global_config_file
        if local_config_file is None:
            cwd = os.getcwd()
            local_cfpath = os.path.join(cwd, "tmb_config.py")
            if os.path.isfile(local_cfpath):
                local_config_file = local_cfpath
        self.local_config_file = local_config_file
        #options one layer deep children["child_name"]-->child_option
        self.children = OrderedDict()
        #options['opt1.opt2.opt3']-->Option
        self.options = OrderedDict()
    
    def run_config(self):
        """execute the config.py file in the 
        working data name space.
        """
        if os.path.isfile(self.global_config_file):
            cfile = open(self.global_config_file)
            exec(cfile.read(), self.eval_ns)
            cfile.close()
        if not self.local_config_file is None:
            cfile = open(self.local_config_file)
            exec(cfile.read(), self.eval_ns)    
            cfile.close()
    
    def __getattr__(self, opt_name):
        return self.children[opt_name]
    
    def traverse_tree(self, index):
        if not isinstance(index, str):
            raise ValueError("option index should be a string not {}".format(type(index)))
        spl = index.split(".")
        cur_opt = self
        for opt_name in spl:
            cur_opt = cur_opt.children[opt_name]
        return cur_opt
    
    def __getitem__(self, index):
        return self.options[index].value
    
    def __setitem__(self, index, value):
        self.options[index].value = value       
    
    def parse_commands(self, argv):
        """parse arguments passed as commandline arguments
        program name is expected to have been removed
        """
        #import pdb; pdb.set_trace()
        self._current_path = None
        self.consume_tasks(argv)
    
    def consume_tasks(self, argv):
        """execute the top task in the argv stack
        """
        if len(argv) == 0:
            return
        elif argv[0] in tmb.tasks.task_registry:
            task_name = argv.pop(0)
            self._current_path = task_name
            task = tmb.tasks.task_registry[task_name]
        argv = self.consume_options(argv)
        task.run()
        self.consume_tasks(argv)
    
    def consume_options(self, argv):
        if len(argv) == 0:
            return []
        cpath = None
        if self.full_path_pattern.match(argv[0]):
            self._current_path = argv.pop(0)[2:].replace("-", "_")
            cpath = self._current_path
        #if the next thing in argv is a sub_path specifier apply it
        if self.sub_path_pattern.match(argv[0]):
            rel_path = argv.pop(0)[1:].replace("-", "_")
            if not self._current_path is None:
                cpath = "{}.{}".format(self._current_path, rel_path)
            else:
                cpath = rel_path
        #consume an option value specifier if necessary
        if cpath is None:
            if argv[0] in tmb.tasks.task_registry:
                return argv
            else:
                raise OptionSpecificationError("option {} is not a task name".format(argv[0]))
        elif len(argv) < 1:
            raise OptionSpecificationError("no value specifier following option {}".format(cpath))
        else:
            runtime_str = argv.pop(0)
            try:
                option = self.options[cpath]
            except KeyError:
                raise ValueError("no option found at option path {}".format(cpath))
            option.set_runtime_str(runtime_str)
        return self.consume_options(argv)
    
    def register_option(self, name, option, parent):
        if not isinstance(name, str):
            raise ValueError("option name must be a string! not {}".format(type(name)))
        if parent is None:
            parent = self
        elif isinstance(parent, str):
            parent = self.traverse_tree(parent)
        if name in list(parent.children.keys()):
            print("Warning: overwriting option {} in {} with option of same name".format(name, parent))
        if not isinstance(parent, (Option, OptionTree)):
            raise ValueError("parent option value of type {} not understood".format(type(parent)))
        parent.children[name] = option
        option.parent = parent
        parent_path  = parent.option_path
        if parent_path == "":
            option_path = name
        else:
            option_path = "{}.{}".format(parent_path, name)
        option.option_path = option_path
        self.options[option_path] = option


opts = OptionTree()

class Option(object):
    eval_ns = wds.__dict__
    
    def __init__(
            self, 
            name, 
            parent=None,
            option_style=None,
            envvar=None,
            runtime_str=None, 
            description="",
            help_="",
            use_cached=True, 
            option_tree=opts,
            editor_style=None,
            **kwargs
    ):
        """an option which can be variously specified either by a default value
        in the code, set using the value of a environment variable, evaluated
        dynamically at runtime using a string read from the command line, or set in config.py
        """
        self.children = OrderedDict()
        name_spl = name.split(".")
        self.name = name_spl[0]
        if len(name_spl) > 1:
            if not parent is None:
                raise ValueError("cannot both specify parent and have a chained name (e.g. parent.child)")
            parent = ".".join(name_spl[1:])
        self.option_style = option_style
        self.editor_style = editor_style
        self.description = description
        self.help = help_
        opt_style_poss = "parent_dict".split()
        if not option_style is None:
            if not option_style in opt_style_poss:
                raise ValueError("option_sytle must be one of {} received {}".format(opt_style_poss, option_style))
        self.default_specified = False
        self.runtime_str = runtime_str
        self.use_cached = use_cached
        default = None
        self.default_specified = False
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
        #populate from environment variables
        if not self.envvar is None:
            if self.envvar in os.environ:
                self.runtime_str = os.environ[self.envvar]
        self.option_tree = option_tree
        self.option_tree.register_option(name, self, parent)
    
    def __getattr__(self, opt_name):
        return self.children[opt_name]
    
    def set_runtime_str(self, value):
        if not isinstance(value, str):
            raise TypeError("runtime_str must be of type string not type {}".format(type(value)))
        self.runtime_str = value
        self._valuated = False
    
    def evaluate(self):
        if not self.runtime_str is None:
            try:
                res = eval(self.runtime_str, self.eval_ns)
                self._value = res
                self._valuated = True
                return res
            except SyntaxError as e:
                raise OptionSpecificationError("Evaluation of string:\n{}\nfailed with error {}".format(self.runtime_str, e))
            except OptionSpecificationError as ose:
                raise ose
            except Exception as e:
                raise OptionSpecificationError("Unknown valuation error {}".format(e))
        else:
            raise OptionSpecificationError("runtime string is None")
    
    @property
    def value(self):
        if self.option_style == "parent_dict":
            children = self.children
            return {k:children[k].value for k in children}
        if (not self._valuated) or (not self.use_cached):
            self.evaluate()
        return self._value
    
    @value.setter
    def value(self, value):
        if self.option_style == "parent_dict" or (not self.use_cached):
            raise NotImplementedError("attribute setting for these option styles not implemented")
        self._value = value
        self._valuated = True
    
    def __repr__(self):
        val = "valuation failed"
        try:
            val = self.value
        except OptionSpecificationError:
            pass
        return "Option {}, value={}".format(self.option_path, val)


