#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PURPOSE: 
AUTHOR: Dylan Gregersen
DATE: Sun Nov  9 19:43:40 2014
"""
# ########################################################################### #

# import modules 

from __future__ import print_function, division, unicode_literals
import os 
import sys 
import re 
import time
import numpy as np 
import pylab as plt
import unittest

from thimbles.radtran.moog_engine import (\
    _moog_par_format_synlimits,_moog_par_format_fluxlimits,
    _moog_par_format_plot,_moog_par_format_plotpars,_moog_par_format_synlimits,
    _moog_par_format_abundances,get_model_name)


# ########################################################################### #


class TestMOOGParFormat (unittest.TestCase):

    def setUp (self):
        pass 
    
    def test_get_model_name (self):
        pars = [5000,4.10,-2.13,1.10,"ODFNEW"]
        sol = "5000p410m213v110.ODFNEW"
        ans = get_model_name(*pars)
        assert sol==ans
    
    def test_moog_par_format_plotpars (self):
        plotpars = [\
            [5555.0,5600.0,0.2,1],
            [-100,0.0,0.0,1.0],
            ['gs',1.0,1.0,1.0,1.0,1.0]
            ]
        sol = "\n".join((\
            "plotpars       1",
            " 5555.00 5600.00 0.20 1.00",
            " -100.00 0.00 0.00 1.00",
            "    gs 1.00 1.00 1.00 1.00 1.00",
            ))
        ans = _moog_par_format_plotpars(plotpars)
        assert sol==ans 
    
    def test_format_synlimits (self):
        synlimits = [5555.0,5600.0,1,1]
        sol = "\n".join((\
            "synlimits ",
            "  5555.00 5600.00 1.00 1.00",
            ))
        ans = _moog_par_format_synlimits(synlimits)
        assert sol,ans 
    
    def test_moog_format_abundances (self):
        abundances = [[26.0,-9, -1.2, 0],
                      [8.0, -9,   -1, 0],
                      [7.0, -9,  1.2, 2],
                      [6.0, -9,   -9, -9]]
        sol = "\n".join((\
            "abundances     3     3",
            "   26.0    -9.0    -1.2     0.0",
            "    8.0    -9.0    -1.0     0.0",
            "    7.0    -9.0     1.2     2.0",
            "    6.0    -9.0    -9.0    -9.0",                
            ))
        ans = _moog_par_format_abundances(abundances)
        assert sol==ans


pass 
# ########################################################################### #
if __name__ == "__main__":
    unittest.main()
    
