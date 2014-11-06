#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PURPOSE: For handling tasks in Thimbles
AUTHOR: dylangregersen
DATE: Mon Aug 25 14:39:11 2014
"""
# ########################################################################### #

# import modules 

from __future__ import print_function, division
import os 
import sys 
import re 
import time
import inspect
from collections import OrderedDict
from types import LambdaType

import cPickle
import numpy as np 

import workingdataspace as wds
from thimbles.options import Option, opts

# ########################################################################### #

class TaskRegister(object):
    
    def __init__(self):
        self.registry = {}
    
    def register_task(self, task):
        self.registry[task.name] = task
    
    def __getitem__(self, index):
        return self.registry[index]

task_registry = TaskRegister()

def task(name=None, result_name="return_value"):
    new_task = Task(name=name, result_name=result_name)
    return new_task.set_func

#def task(task_func):
#    task_registry[task_func.__name__] = Task(task_func)

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
    arg_dict = OrderedDict(zip(argspec.args, defaults))
    
    if return_has_default:
        has_defaults = [False for i in range(n_args_total-n_defaults)]
        has_defaults.extend([True for i in range(n_defaults)])
        has_defaults_dict = OrderedDict(zip(argspec.args, has_defaults))
        return arg_dict, has_defaults_dict
    else:
        return arg_dict 

class Task(Option):
    target_ns = wds.__dict__
    
    def __init__(self, result_name, name, func=None):
        self.name = name
        self.result_name = result_name
        if not isinstance(self.result_name, basestring):
            raise ValueError("result_name must be of type string not type {}".format(type(result_name))) 
        if not func is None:
            self.set_func(func)
    
    def set_func(self, func):
        self.func = func
        if not self.func is None:
            if self.name is None:
                self.name = self.func.__name__
            super(Task, self).__init__(name=self.name, default=self.func)
            self.generate_child_options()
            task_registry.register_task(self)
        return func
    
    def generate_child_options(self):
        arg_dict, arg_has_default = argument_dict(self.func, return_has_default=True)
        self.task_kwargs = arg_dict.keys()
        for arg_key in arg_dict:
            opt_kwargs = {}
            if arg_has_default[arg_key]:
                opt_kwargs["default"] = arg_dict[arg_key]
            Option(name=arg_key, parent=self, **opt_kwargs)
        #add a special option for result_name
        #new_opt = Option(name="result_name", parent=self, option_style="raw_string", default=self.result_name) 
        #Option(name="task_order", default=-1, parent=self)
    
    def run(self):
        task_kwargs = {kw:getattr(self, kw).value for kw in self.task_kwargs}
        func_res = self.func(**task_kwargs)
        self.target_ns[self.result_name] = func_res
        return func_res

class EvalError (Exception):

    def __init__ (self,argname,msg):
        self.argname = argname        
        Exception.__init__(self,self.argname,msg)

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

class DylanTask (object):
    """ A task is based on some function
    
    This task : {target}
    
    This is used to match argument names in the function with namespaces and 
    expressions to be evaluated in a namespace
    
    
    Example
        # have some function
        def target (x,y,a=2.3,b=4.1)
            return a*x+b*y
        
        # set these by hand
        argstrings['x'] = "np.array(3)"
        argstrings['y'] = "3*np.cos(xpts)"
        argstrings['a'] = "2+3"
        argstrings['b'] = "b"
    
        # have some namespace
        namespace = dict(np=numpy,xpts=np.array([-0.3,0,0.3]),b=1.2)
        
        # by hand this is what's happening:
        result = target(x=eval(argstrings['x'],namespace), # x=array([0, 1, 2])
                        y=eval(argstrings['y'],namespace), # y=array([ 2.86600947,  3.        ,  2.86600947])
                        a=eval(argstrings['a'],namespace), # a=5                
                        b=eval(argstrings['b'],namespace), # b=1.2
                        )
            
        # is the same as:
        task = Task(target)
        task.argstrings['x'] = "np.array(3)"
        task.argstrings['y'] = "3*np.cos(xpts)"
        task.argstrings['a'] = "2+3"
        task.argstrings['b'] = "4.1"
        namespace['b'] = 2
        result = task.eval(namespace)
    
      
    """ 
    defaultNamespace = wds.__dict__
    
    def __init__ (self, target=None, argstrings=None, result_name="result"):
        """
        Parameters
        target : callable
            This is the target function for this task
        argstrings : dict or optional
            The keys are arguments for the target function. The values are strings
            which can be evaluated for the argument. I.e. 
            def func(x):
                ...
            argstrings["x"] = "a*2.0"
            func(eval("a*2.0"))
        
        """        
        # store target
        #if not callable(target):
        #    raise TypeError("target function must be callable")
        self.target = target
        self._target_id = id(target)
        self.result_name=result_name
        # initialize argstrings for target
        self.reset_argstrings(argstrings=argstrings)
        if self.star_args is not None:
            raise ValueError("can't accept *args right now")
        # update doc string with target
    
    def reset_argstrings (self,argstrings=None):
        """ Creates argstrings for the target function.
        
        >>> def func (x,y,name="hello"):
        >>>    ...                
        >>> argstrings = OrderedDict()
        >>> argstrings["x"] = "x"
        >>> argstrings["y"] = "y"
        >>> argstrings["name"] = self.get_repr("hello")
                
        Parameters
        argstrings : dict or None
            keys with have the names of variables for the target function
            the values are strings which can be evaluated, i.e. eval(value)
                
        """
        # get the argspec of target
        self.argspec = inspect.getargspec(self.target)
        argspec = self.argspec
        
        self.argstrings = OrderedDict()        
        
        # define argstrings as an ordered dictionary
        nargs = len(self.argspec.args)
        if self.argspec.defaults is None:
            nkws = 0
        else:
            nkws = len(self.argspec.defaults)
        dargs = nargs-nkws
        argnames = []
        keynames = []
        for i in range(nargs):
            # get argument name
            argname = argspec.args[i]
            if argname in self.argstrings:
                continue            
            # get argument default value
            if i >= dargs:
                argvalue = self.get_repr(argspec.defaults[i-dargs])
                keynames.append(argname)
            else:
                argvalue = argname
                argnames.append(argname)                
            # store the previous input
            self.argstrings[argname] = argvalue 
        
        # store variables            
        self._argnames = argnames
        self._keynames = keynames           
        self.star_args = argspec.varargs
        self.star_kwargs = argspec.keywords
        
        if argstrings is not None:        
            self.update_argstrings(argstrings)
    
    def update_argstrings (self,argstrings=None,namespace=None):
        """ Update what value is evaluated
        
        Parameters
        argstrings : dict of string values
            Each key is an argument for the target function (keys which are not
            are ignored) and each value is a string to evaluate to get that
            argument value (e.g.  arg_x = eval(input['arg_x']))
        namespace : dict
            If given for each argument (aka key) in self.argstrings if that argument
            name appears in the namespace, then that name is used. e.g.
            def func (a=2):
                return a
            argstrings['a'] = '2'
            eval(namespace={'a':3.14}) # -> 2            
            update(namespace={'a':3.14})  # -> argstrings['a'] = 'a' because 'a' is a name in the namespace
            eval(namespace={'a':3.14}) # -> 3.14
            eval(namespace={'a':4.00}) # -> 4.00            
        
        """
        if namespace is not None:
            for key in self.argstrings:
                if key in namespace:
                    self.argstrings[key] = key 
        
        if argstrings is not None:
            for key in argstrings:
                if not isinstance(argstrings[key],basestring):
                    raise TypeError("every value in argstrings must be a string which can be evaluated")
                if key in self.argstrings:
                    self.argstrings[key] = argstrings[key]
    
    def get_repr (self,value):
        """ Take a value to a representation which can be evaluated
                
        rep = self.get_repr(obj_in)
        obj_out = eval(rep)
        obj_in == obj_out # True
        
        This function is used whenever argstrings are given to one of these methods
        Parameter
        value : obj
            If lambda function then the one line string evaluation is returned
            Else repr(obj) is returned. If you have a custom object you want to
            evaluate this way then please specify __repr__ for it
            Any function (callable) can't be given
        
        Raises
        EvalError : if value is a callable function then there is not a simple 
                    repr for it. 
                                    
        """    
        if isinstance(value,(basestring,float,int,complex,list,dict)):
            return repr(value)
        if isinstance(value,LambdaType):
            rep = inspect.getsource(hey).rstrip()
            return rep[rep.find("=")+1:]        
        elif callable(value):
            # perhaps use the memory address to instanciate a python object
            raise EvalError("Don't know how to turn function into a re-evaluatable thing")
        else:
            return repr(value)
    
    def eval_argstring (self,key,namespace=None):
        """ return eval(self.input[key],namespace) 
        
        Raises
        KeyError : if key not in self.input
        EvalError : if NameError, SyntaxError, Exception
                
        """
        ns = self._get_namespace(namespace)
        # try to evaluate the argstrings with the namespace values
        try:
            value = eval(self.argstrings[key],ns)
            # KeyError : this should actually error as a key error
            return value
                
        # if the name does not exist
        except NameError as e:
            raise EvalError(key,e.message)
        
        # if you have a syntax error
        except SyntaxError as e:
            raise EvalError(key,e.message)            
        
        # raise any other exception 
        except Exception as e:
            if isinstance(e,KeyError):
                raise e
            else:
                raise EvalError(key,e.message) 
    
    def _get_namespace (self,namespace=None,**kwargs):
        # ======================= get the namespace
        if namespace is None:            
            ns = self.defaultNamespace
        else:
            ns = namespace
        # if kwargs are given overwrite with them
        if len(kwargs):
            for key in namespace:
                if key in kwargs:
                    ns[key] = kwargs[key]
                else:
                    ns[key] = namespace[key]
            for key in kwargs:
                if key not in ns:
                    ns[key] = kwargs[key]
        return ns     
            
    def get_args (self,namespace=None,**kwargs):
        """ Use the namespace and kwargs dicts to evaluate the self.argstrings
        and return arguments for the target function
        
        Parameters
        namespace : dict
            This is the primary names to use
        kwargs : dict
            If a name in kwargs is given, then this (name,value) is used instead
            of the pair in namespace
        
        Returns
        args : list
            *args for the target function
        kwargs : dict
            **kwargs for the target function
            
            
        Example
        >>> task = Task(target)
        >>> namespace = dict(a=3.14)
        >>> args,kws = task.get_args(namespace)
        >>> target(*args,**kwargs) == task.eval(namespace)
        
        """ 
        # double check        
        keys = self._argnames + self._keynames
        for key in self.argstrings:
            if key not in keys:
                raise KeyError("Missing argument {}".format(key))
                               
        ns = self._get_namespace(namespace,**kwargs)
                                
        # ======================= check that each thing evals
        
        # BASIC: kws[key] = eval(self.argstrings[key],ns)        
        args = []
        for key in self._argnames:
            args.append(self.eval_argstring(key,ns))
                        
        kws = {}  
        for key in self._keynames:
            kws[key] = self.eval_argstring(key,ns)
        
        # ======================= **kwargs for self.target
        if self.star_kwargs is not None and self.star_kwargs in ns:
            kws.update(ns[self.star_kwargs])
        #if self.star_args is not None and self.star_args in ns:
        #    args += list(*ns[self.star_args])        
        return args,kws             
                                
    def eval (self,namespace=None,**kwargs):
        """ Evaluate the target function with variables from this namespace
        over-ridden by kwargs with defaults taken from argstrings
                
        
        Parameters
        namespace : dict
        **kwargs : key=value
            If specified you can overwrite what name is matched to the argument
            name. 
            
        
        """
        # eval target with namespace
        args,kws = self.get_args(namespace,**kwargs)
        return self.target(*args,**kws)
    
    @property
    def help (self):
        """ Returns a help string """
        # could include a repr of self.argstrings for help
        return str(self.target.__doc__)
        
    def func_code (self,max_lines=100,exclude_docstring=True,show=False):
        """ Prints out the source code (from file) for target function
    
        inspect.getsourcelines(self.target)
    
        """
        return fprint(self.target,max_lines=max_lines,
                      exclude_docstring=exclude_docstring,show=show)
        
    
    def run_task(self):
        eval_result = self.eval()
        self.defaultNamespace[self.result_name] = eval_result
