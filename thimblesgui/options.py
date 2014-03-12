#the parsing for runtime arguments.
import argparse

desc = "a spectrum processing and analysis GUI"
parser = argparse.ArgumentParser(description=desc)

file_help = "paths to spectra files or other files to read in"
parser.add_argument("files", nargs="*", help=file_help)

line_list_help = "the path to a linelist file to load"
parser.add_argument("--line-list", "--ll", help=line_list_help)

dwhelp = "default number of angstroms to display" 
parser.add_argument("--display-width", "--dw", type=float, default=3.0, help=dwhelp)

#TODO: replace these options with a kwarg for the individual mode using --fit
fwhelp = "default individual line fit width"
parser.add_argument("--fit-width", "--fw", type=float, default=0.3, help=fwhelp)

dgamma_help = "penalize lorentz width values above this threshold"
parser.add_argument("--gamma-max", "--gm", type=float, default=0.0)

read_help = "name of function to load the data files with"
parser.add_argument("--read_func", default="read")

rv_help = "optional radial velocity shift to apply"
parser.add_argument("--rv", type=float, default=0.0, help=rv_help)

norm_help="type of normalization to apply to spectra"
parser.add_argument("--norm", default="ones", help=norm_help)

fit_help="the type of spectral fit to run"
parser.add_argument("--fit", default="none", help=fit_help)

benchmark_help = "run in benchmark mode"
parser.add_argument("--benchmark", help=benchmark_help)

nwhelp="suppress the GUI window"
parser.add_argument("--no-window", "--nw", action="store_true", help=nwhelp) 

st_help = "a valid python script or expression to execute in the user name space on startup"
parser.add_argument("--startup", default="", help=st_help)

options = parser.parse_args()
