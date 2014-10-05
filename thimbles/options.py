import sys

import pyparsing

class OptionSpecificationError(Exception):
    pass

class OptionTree(object):
    
    def __init__(self):
        pass
    
    def traverse_tree(self, index):
        if not isinstance(index, basestring):
            raise ValueError("option index should be a string not {}".format(type(index)))
        spl = index.split(".")
        cur_opt = self
        for opt_name in spl:
            cur_opt = getattr(cur_opt, opt_name)        
        return cur_opt
    
    def __getitem__(self, index):
        cur_opt = self.traverse_tree(index)
        return cur_opt.value
    
    def __setitem__(self, index, value):
        cur_opt = self.traverse_tree(index)
        cur_opt.value = value

opts = OptionTree()

class Option(object):
    
    def __init__(self, name, parent, default=None, converter=None, help="no help string specified"):
        self.default = default
        self.value = default
        self.help = help
        self.register_option(name, parent)
    
    def register_option(self, name, parent):
        if not isinstance(name, basestring):
            raise ValueError("option name must be a string! not {}".format(type(name)))
        if parent is None:
            parent = opts
        elif isinstance(parent, basestring):
            parent = opts.traverse_tree(parent)
        if hasattr(parent, name):
            raise OptionSpecificationError("option with name {} already specified!".format(self.name))
        if not isinstance(parent, Option):
            raise ValueError("parent option value of type {} not understood".format(type(parent)))
        setattr(parent, name, self)
    
    def __repr__(self):
        return repr(self.value)
    
    