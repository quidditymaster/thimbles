#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Purpose: For dependencies
# Author: Dylan Gregersen
# Date: Jan 18, 2014

# ########################################################################### #

# Standard Library
from __future__ import division, print_function, absolute_import
import os

# ########################################################################### #

has_package = {'numpy':True}
# TODO: make this be a yaml file which has the data. 

try:
    import numpy
except ImportError:
    has_package['numpy'] = False

# TODO: perhaps add in version checking. could have has_package['numpy'] = False if wrong
# version?? or should it give warning?? "may not work"

# ########################################################################### #
def load_package_dependencies ():
    """
    read the package dependencies yaml file and return the has_package dictionary
    """
    pass

def evaluate_dependencies ():
    """
    Runs through all the files in Thimbles and evaluates what dependencies
    it has and checks if it's included here
    """
    # TODO: create
    pass

def require (package_name,raise_='error'):
    """
    Given a package_name it checks if it has is
    
    Parameters
    ----------
    package_name : string
        has the identity of a package
    raise_ : string
        * error : will raise error
        * warning : will raise warning
        * ignore : no raise will be given
    
    Returns
    -------
    has_package : bool
        True if has the package
    
    """
    has_package = True
    errmsg = "Package {package_name} is required to run this function"
    errmsg_kws = dict(package_name=package_name)
    
    # TODO: write this
    
    if not has_package:
        if raise_ == 'error':
            raise ImportError(errmsg.format(**errmsg_kws))
        # TODO: write others, 'warning','ignore'
    return has_package
  
  
if __name__ == "__main__":
    # TODO: check ALL dependencies of Thimbles
    pass
        
