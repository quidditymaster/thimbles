import os
import numpy as np
import matplotlib.pyplot as plt

from thimblesgui import tmb
from thimbles.io import *

from options import options
command_str = options.startup
if len(command_str) > 3 and command_str[-3:]==".py":
    execfile(command_str)
else:
    try:
        exec(command_str)
    except Exception as e:
        print("Error with user given --startup argument:")
        print("    {} : {}".format(type(e).__name__,e.message))


template_dir = os.path.join(os.path.join(os.path.dirname(tmb.__file__), "resources"), "templates")

#load specified template spectra
for template_name in options.templates:
    store_name = template_name.split(".")[0]
    if os.path.isfile(template_name):
        try:
            exec("%s = read(%s)" % (store_name, template_name) )
        except Exception as e:
            print "unable to read template %s" % template_name
            print e
    elif os.path.isfile(os.path.join(template_dir, template_name)):
        tfname = os.path.isfile(os.path.join(template_dir, template_name))
        exec("%s = read(%s)" % (store_name, tfname))

def eval_(cmd_str):
    return eval(cmd_str)




