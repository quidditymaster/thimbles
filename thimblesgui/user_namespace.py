
import numpy as np
import matplotlib.pyplot as plt

from thimblesgui import tmb
from thimbles.io import *

from options import options
command_str = options.startup

def eval_(cmd_str):
    return eval(cmd_str)

if len(command_str) > 3 and command_str[-3:]==".py":
    execfile(command_str)
else:
    try:
        exec(command_str)
    except Exception as e:
        print("Error with user given --exec argument:")
        print("    {} : {}".format(type(e).__name__,e.message))

