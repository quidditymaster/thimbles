#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PURPOSE: 
AUTHOR: dylangregersen
DATE: Mon Aug 25 14:42:00 2014
"""
# ########################################################################### #

# import modules 

from __future__ import print_function, division
import os 
import sys 
import re 
import time
import numpy as np 
import unittest
try:
    import ipdb as pdb
except ImportError:
    import pdb
from thimbles.tasks import DylanTask as Task,EvalError

# ########################################################################### #
class _TestTask (object): # unittest.TestCase
    
    def setUp (self):
        # test function/task
        def func (x,y,a=2):
            return a*x+y          
        self.func1 = func  
        self.task1 = Task(func)
        
        def func (x,a=2,**kwargs):
            # kwargs.update({"i{}".format(i):i for i in xrange(len(3))})
            kwargs['a'] = a
            return kwargs
        self.func2 = func
        self.task2 = Task(func)
        
    def runTests (self):
        self.setUp()
        for attr in dir(self):
            if "test_" in attr:
                print("testing {} ...".format(attr))
                getattr(self,attr)()
        print("success!")
        
    def test_error (self):
        task = self.task1
        # check for namespace error
        namespace = {'x':3} # missing y
        try:
            task.eval(namespace)
            raise NameError("Should have been missing y")
        except EvalError as e:
            pass    

    def test_arguments (self):
        task = self.task1
        func = self.func1
        namespace = {'y':1.1,'x':2,'a':3}
        sol_args = [2,1.1]
        sol_kws = {'a':3}
        
        task.argstrings['a'] = 'a' # so that it'll take 'a' from the namespace
        # argstrings for 'x' and 'y' should already be 'x','y' to be taken from namespace

        
        ans_args,ans_kws = task.get_args(namespace)
        
        self.assertEqual(sol_args,ans_args)
        self.assertEqual(sol_kws,ans_kws)        
        

    def test_appropriate_namespace (self):
        return 
        task = self.task1
        func = self.func1
        namespace = {'x':2,'y':1.1}
            # check argstrings
        sol = func(2,1.1)
        ans = task.eval(namespace)
        self.assertEqual(sol,ans)

    def test_keywords (self):
        func = self.func1
        task = self.task1
        namespace = {'x':3,'y':1.1,'a':3}
        sol = 10.1
        ans = task.eval(namespace)
        self.assertEqual(sol,ans)                   

    def test_repetition (self):
        func = self.func1
        task = self.task1

        namespace = {'x':3,'y':1.1,'a':3}
        sol = 10.1
        ans = task.eval(namespace)
        self.assertEqual(sol,ans)           
          
        # if eval isn't given a namespace then it takes them from globals
        for key in namespace:
            globals()[key] = namespace[key]                        
        ans = task.eval()
        self.assertEqual(sol,ans)                      

#     def test_star_args_kwargs (self):
#         func = self.func2 
#         task = self.task2
#         
#         namespace = {'args':['a',3,2],'x':2,'a':3.4,'kwargs':dict(hello='world')}
#         
#         sol_args = [2,'a',3,2]
#         sol_kws = {'a':3.4,'hello':'world'}
#         
#         ans_args,ans_kws = task.get_args(namespace)
#         
#         self.assertEqual(sol_args,ans_args)
#         self.assertEqual(sol_kws,ans_kws)        
        
#     def test_result_star_args (self):
#         func = self.func2 
#         task = self.task2
#         namespace = {'args':['a',3,2],'x':2,'a':3.4,'kwargs':dict(hello='world')}
#         # can't handle *args because I need to track order much better. 
#         # 
#         # func(*given_args,*given_kws,*args,*kws)
#         # 
#         # all the given keywords are actually positional too.
#         #        
#         sol = func(2,'a',hello='world')
#         ans = task.eval(namespace)
#         self.assertEqual(sol,ans)                      
        
        
    def test_modify_argstrings (self):
        func = self.func1
        task = self.task1
        
        namespace = {'xpts':np.arange(10),'scale':2.3,'y':3,'a':1.3}
        
        argstrings = {}
        argstrings['x'] = "xpts*scale"
        argstrings['y'] = "scale/a"
        argstrings['a'] = "3.3"
        task.update_argstrings(argstrings)      
        
        x = np.arange(10)*2.3
        y = 2.3/1.3
        a = 3.3
        
        sol = func(x,y,a=a)  
        ans = task.eval(namespace)
        self.assertTrue(np.all(sol==ans))
   
    def test_modify_argstrings2 (self):
        func = self.func1
        task = self.task1

        namespace = {'xpts':np.arange(10),'scale':2.3,'y':3}
        
        task.argstrings['x'] = "xpts*scale"
        task.argstrings['y'] = "scale + 2"
        task.argstrings['a'] = "3.3"
        
        x = np.arange(10)*2.3
        y = 2.3 + 2
        a = 3.3
        
        sol = func(x,y,a=a)        
        ans = task.eval(namespace)
        self.assertTrue(np.all(sol==ans))        
   
pass 
# ########################################################################### #
if __name__ == "__main__":
    #TestTasks().runTests()
    pass
