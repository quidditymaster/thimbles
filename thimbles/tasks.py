#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PURPOSE: For handling tasks in Thimbles
AUTHOR: dylangregersen
DATE: Mon Aug 25 14:39:11 2014
"""
# ########################################################################### #

# import modules 


import os 
import sys 
import re 
import time
import inspect
from collections import OrderedDict
from types import LambdaType

import pickle
import numpy as np 

from . import workingdataspace as wds
from thimbles.options import Option, opts, EvalError
import collections

# ########################################################################### #

task_registry = {}
def task(name=None, result_name="return_value", option_tree=opts, registry=task_registry, sub_kwargs=None):
    new_task = Task(name=name, result_name=result_name, option_tree=option_tree, registry=registry, sub_kwargs=sub_kwargs)
    return new_task.set_func

def argument_dict(func, filler_value=None, return_has_default=False):
    argspec = inspect.getargspec(func)
    n_args_total = len(argspec.args)
    if argspec.defaults is None:
        n_defaults = 0
    else:
        n_defaults = len(argspec.defaults)
    defaults = [filler_value for i in range(n_args_total-n_defaults)]
    if not argspec.defaults is None:
        defaults.extend(argspec.defaults)
    arg_dict = OrderedDict(list(zip(argspec.args, defaults)))
    
    if return_has_default:
        has_defaults = [False for i in range(n_args_total-n_defaults)]
        has_defaults.extend([True for i in range(n_defaults)])
        has_defaults_dict = OrderedDict(list(zip(argspec.args, has_defaults)))
        return arg_dict, has_defaults_dict
    else:
        return arg_dict 

class Task(Option):
    target_ns = wds.__dict__
    
    def __init__(self, result_name, name, option_tree, registry, sub_kwargs=None, func=None):
        if sub_kwargs is None:
            sub_kwargs = {}
        self.sub_kwargs = sub_kwargs
        self.name = name
        self.result_name = result_name
        self.option_tree = option_tree
        self.registry = registry
        if not isinstance(self.result_name, str):
            raise ValueError("result_name must be of type string not type {}".format(type(result_name))) 
        if not func is None:
            self.set_func(func)
    
    def set_func(self, func):
        self.func = func
        if not self.func is None:
            if self.name is None:
                self.name = self.func.__name__
            super(Task, self).__init__(self.name, default=func)
            self._generate_child_options()
            self.registry[self.name] = self
        return func
    
    def _generate_child_options(self):
        arg_dict, arg_has_default = argument_dict(self.func, return_has_default=True)
        self.task_kwargs = list(arg_dict.keys())
        for arg_key in arg_dict:
            opt_kwargs = self.sub_kwargs.pop(arg_key, {})
            if arg_has_default[arg_key]:
                opt_kwargs["default"] = arg_dict[arg_key]
            Option(name=arg_key, parent=self, **opt_kwargs)
        if len(self.sub_kwargs) > 0:
            print("Warning, not all sub_kwargs consumed! in Task.generate_child_options for task {} \n, {} left unconsumed".format(self.name, self.sub_kwargs))
    
    def run(self, **kwargs):
        task_kwargs = {kw:getattr(self, kw).value for kw in self.task_kwargs}
        task_kwargs.update(**kwargs)
        func_res = self.func(**task_kwargs)
        self.target_ns[self.result_name] = func_res
        return func_res

def fprint (func,max_lines=100,exclude_docstring=True,show=True):
    """ function print : Prints out the source code (from file) for a function
    
    inspect.getsourcelines(func)
    
    """
    import inspect  
    filepath = inspect.getsourcefile(func)
    code_lines,num = inspect.getsourcelines(func)
    
    # ----------------------- 
    to_print = []    
    to_print.append("from: '{}'\n".format(filepath))
    to_print.append("line: {}\n\n".format(num))
    to_print += code_lines    
    to_print = str("".join(to_print[:max_lines]))
    
    # -----------------------
    if exclude_docstring:
        msg = ' <docstring see help({})> '.format(func.__name__)
        to_print = to_print.replace(func.__doc__,msg)
    
    if show:
        print(to_print) 
    else:
        return to_print 

