#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DATE: Mon Aug 25 14:42:00 2014
"""
# ########################################################################### #


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

from thimbles.tasks import Task, task
from thimbles.options import OptionTree, Option, OptionSpecificationError
import thimbles.workingdataspace as wds


# ########################################################################### #
class _TestTask (object): # unittest.TestCase
    
    def setUp (self):
        self.opts = OptionTree()
        self.register = {}
        
        @task(name="task1", result_name="task1_result", option_tree=self.opts, registry=self.register)
        def func1(x,y,a=2):
            return a*x+y
    
    def test_task_decorator(self):
        @task(option_tree=self.opts, registry=self.register)
        def random_task_name():
            return "TADA!"
        self.opts.random_task_name.name = "random_task_name"
    
    def test_children(self):
        top_cdict = self.opts.children
        self.assertTrue(len(top_cdict) == 1)
        self.assertTrue("task1" in top_cdict)
        xcdict = self.opts.task1.x.children
        ycdict = self.opts.task1.y.children
        acdict = self.opts.task1.a.children
        self.assertTrue(len(xcdict) == 0)
        self.assertTrue(len(ycdict) == 0)
        self.assertTrue(len(acdict) == 0)
    
    def test_empty_runtime_string(self):
        self.opts.task1.x.set_runtime_str("")
        #import pdb; pdb.set_trace()
        with self.assertRaises(OptionSpecificationError):
            self.opts.task1.x.value
    
    def test_run_task(self):
        task1 = self.opts.task1
        wds.bloot = 3
        task1.x.set_runtime_str("bloot")
        task1.y.set_runtime_str("5")
        task1.a.set_runtime_str("-2")
        task1.run()
        self.assertEqual(wds.task1_result, -2*3 + 5)
    
    def test_option_path(self):
        self.assertTrue(self.opts.task1.option_path == "task1")
        self.assertTrue(self.opts.task1.x.option_path == "task1.x")
        self.assertTrue(self.opts.task1.y.option_path == "task1.y")
        self.assertTrue(self.opts.task1.a.option_path == "task1.a")
    
    def test_default(self):
        self.assertEqual(self.opts.task1.a.value, self.opts["task1.a"])
        self.assertEqual(self.opts.task1.a.value, 2)
    
    def test_fancy_eval(self):
        self.opts.task1.x.set_runtime_str("np.arange(10)")
        self.assertTrue(np.all(np.arange(10) == self.opts["task1.x"]))


if __name__ == "__main__":
    unittest.main()
    
