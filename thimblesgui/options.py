#the parsing for runtime arguments.
import argparse

desc = "a spectrum processing and analysis GUI"
parser = argparse.ArgumentParser(description=desc)
file_help = "paths to spectra files or other files to read in"
parser.add_argument("files", nargs="*", help=file_help)
line_list_help = "the path to a linelist file to load"
parser.add_argument("--line-list", "--ll", help=line_list_help)
fwhelp = "the number of angstroms on either side of the current feature to display while fitting" 
parser.add_argument("--fwidth", "--fw", type=float, default=3.0, help=fwhelp)
parser.add_argument("--read_func", default="read")
parser.add_argument("--rv", type=float, default=0.0, help="optional radial velocity shift to apply")
norm_help="type of normalization to apply to spectra"
parser.add_argument("--norm", default="ones", help=norm_help)
fit_help="the type of spectral fit to run"
parser.add_argument("--fit", default="none", help=fit_help)
nwhelp="suppress the GUI window"
parser.add_argument("--no-window", "--nw", action="store_true", help=nwhelp) 
parser.add_argument("--startup", default="")

options = parser.parse_args()
