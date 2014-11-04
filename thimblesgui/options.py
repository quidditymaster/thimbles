#the parsing for runtime arguments.
#import argparse
import thimbles as tmb
from thimbles.options import Option, opts

#description = "a spectrum processing and analysis GUI"

_help=\
"""if set the positional arguments are assumed to contain 
a list of files to run serially.
setting this option also automatically sets --no-window"""
#Option(name="batch_mode", help=_help, option_style="flag")

_help=\
"""
"""
Option(name="")

_help=\
"""don't display a GUI window just run tasks then quit.
"""
Option(name="no_window", option_style="flag", help=_help)

_help=\
"""suppress the splash screen
"""
Option(name="no-splash", option_style="flag", help=_help)

#st_help = "a valid python script or expression to execute in the user name space on startup"
#parser.add_argument("--startup", default="", help=st_help)

_help=\
"""
"""

if opts.no_window:
    opts.no_splash = True


