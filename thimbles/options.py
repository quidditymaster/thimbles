import sys
import os
from collections import OrderedDict
from copy import copy

from thimbles import thimbles_header_str
from thimbles import workingdataspace as wds

class OptionSpecificationError(Exception):
    pass

class EvalError(Exception):
    pass

config_dir = os.environ.get("THIMBLESCONFIGPATH", os.path.join(os.environ["HOME"], ".config", "thimbles"))
if not os.path.exists(config_dir):
    os.makedirs(config_dir)
config_file = os.path.join(config_dir, "config.txt")


class OptionTree(object):
    
    def __init__(self, config_file=config_file):
        self.option_path = ""
        self.config_file=config_file
        self.children = OrderedDict()
        self.options = OrderedDict()
    
    def __getattr__(self, opt_name):
        return self.children[opt_name]
    
    def traverse_tree(self, index):
        if not isinstance(index, basestring):
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
        if os.path.isfile(self.config_file):
            lines = open(config_file, "r").readlines()
            for line in lines:
                crun_str = None
                spl = line.split("#")[0].split()
                if len(spl) < 1:
                    continue
                elif spl[0][:2] == "--":
                    parent_opt_path = spl.pop(0)[2:].replace("-", "_")
                    sub_path = None
                if len(spl) > 0 and spl[0][:1] == "-":
                    sub_path = spl.pop(0)[1:].replace("-", "_")
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
                    if cur_opt.option_style == "parent_dict":
                        raise OptionSpecificationError("cannot assign runtime strings directly to options with option_style==parent_dict, attempted to assign value={} to option {}".format(crun_str, cur_opt.name))
                    rt_str_dict[cur_opt] = crun_str
        #parse the command line arguments
        argv = copy(sys.argv[1:]) #first value is program name
        parent_opt_path = None
        sub_path = None
        while len(argv) > 0:
            crun_str = None
            if "--" == argv[0][:2]:
                parent_opt_path = argv.pop(0)[2:].replace("-", "_")
                sub_path = None
            elif len(argv) > 0 and argv[0][0] == "-":
                sub_path = argv.pop(0)[1:].replace("-", "_")
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
                if cur_opt.option_style == "parent_dict":
                    raise OptionSpecificationError("cannot assign runtime strings directly to options with option_style==parent_dict, attempted to assign value={} to option {}".format(crun_str, cur_opt.name))
                rt_str_dict[cur_opt] = crun_str
        #set the runtime values we have collected
        for option in rt_str_dict:
            option.set_runtime_str(rt_str_dict[option])
        
        for option in rt_str_dict:
            if not option.on_parse is None:
                option.on_parse()

opts = OptionTree()

class Option(object):
    eval_ns = wds.__dict__
    
    def __init__(self, 
                 name, 
                 parent=None,
                 option_style=None,
                 on_parse=None,
                 envvar=None,
                 runtime_str=None, 
                 help_="",
                 use_cached=True, 
                 option_tree=opts,
                 editor_style=None,
                 **kwargs):
        """an option which can be variously specified either by a default value
        in the code, set using the value of a environment variable, evaluated
        dynamically at runtime using a string read from a config file or 
        specified on the command line
        """
        self.children = OrderedDict()
        self.name = name
        self.on_parse = on_parse
        self.option_style = option_style
        self.editor_style = editor_style
        opt_style_poss = "parent_dict flag raw_string existing_file new_file".split()
        if not option_style is None:
            if not option_style in opt_style_poss:
                raise ValueError("option_sytle must be one of {} received {}".format(opt_style_poss, option_style))
        self.default_specified = False
        if self.option_style == "flag":
            if not "default" in kwargs:
                kwargs["default"] = False
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
        self.help = help_
        self.option_tree = option_tree
        self.register_option(name, parent)
    
    def __getattr__(self, opt_name):
        return self.children[opt_name]
    
    def set_runtime_str(self, value):
        if not isinstance(value, basestring):
            raise TypeError("runtime_str must be of type string not type {}".format(type(value)))
        self.runtime_str = value
        self._valuated = False
    
    def evaluate(self):
        if not self.runtime_str is None:
            try:
                if self.option_style == "raw_string":
                    res = self.runtime_str
                elif self.option_style == "existing_file":
                    if os.path.exists(self.runtime_str):
                        res = self.runtime_str
                    else:
                        raise OptionSpecificationError("file {} does not exist".format(self.runtime_str))
                elif self.option_style == "new_file":
                    res = self.runtime_str
                else:
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
    
    def register_option(self, name, parent):
        if not isinstance(name, basestring):
            raise ValueError("option name must be a string! not {}".format(type(name)))
        if parent is None:
            parent = self.option_tree
        elif isinstance(parent, basestring):
            parent = opts.traverse_tree(parent)
        if name in parent.children.keys():
            print "overwriting option {} in {} with option of same name".format(name, parent)
        if not (isinstance(parent, Option) or isinstance(parent, OptionTree)):
            raise ValueError("parent option value of type {} not understood".format(type(parent)))
        parent.children[name] = self
        self.parent = parent
        parent_path  = self.parent.option_path
        if parent_path == "":
            option_path = self.name
        else:
            option_path = "{}.{}".format(parent_path, name)
        self.option_path = option_path
        self.option_tree.options[self.option_path] = self
    
    def __repr__(self):
        val = "valuation failed"
        try:
            val = self.value
        except OptionSpecificationError:
            pass
        return "Option {}, value={}".format(self.option_path, val)

#general behavior options
_help = "path to prepend to relative paths when searching for input data"
data_dir = Option("data_dir", default=os.getcwd(), help_=_help)

_help = "path to prepend to relative paths when writing out files"
output_dir = Option("output_dir", default=os.getcwd(), help_=_help)

#matplotlib options
_help = "parent option for setting matplotlib style related options"
mpl_style = Option("mpl_style", option_style="parent_dict", help_=_help)
lw = Option(name="line_width", default=1.5, parent=mpl_style, help="default line width")

#spectrum display related options
_help=\
"""options relating to how spectra will be displayed by default
"""
Option(name="spec_display", option_style="parent_dict", help_=_help)

_help=\
"""The logarithm of the ratio of default display window in angstroms
to the central wavelength being displayed.
"""
Option(name="window_width", default=-4.5, parent="spec_display", help_=_help)


def print_option_help():
    print thimbles_header_str
    help_str = "{name}  :  {help}"#\n  value: {value}\n  runtime string:{run_str}" 
    print "Top Level Options"
    top_opts = opts.children
    for op_name in top_opts:
        help_ = top_opts[op_name].help
        print help_str.format(name=op_name, help=help_)
    #for op in opts.options.values():
    #    try:
    #        value = op.value
    #    except OptionSpecificationError:
    #        value = "no value"
    #    if op.runtime_str is None:
    #        run_str = "runtime string unspecified"
    #    else:
    #        run_str = op.runtime_str
    #    print help_str.format(name=op.name, help=op.help)# value=value, run_str=run_str)

_help=\
"""print the help message
"""
Option(name="help", option_style="flag", help_=_help, on_parse=print_option_help)

#TODO: set the on_parse function for the ThimblesLogger options.


del _help
