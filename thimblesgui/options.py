#the parsing for runtime arguments.
#import argparse
import thimbles as tmb
from thimbles.options import Option

#description = "a spectrum processing and analysis GUI"

_help=\
"""if set the positional arguments are assumed to contain 
a list of files to run serially.
setting this option also automatically sets --no-window"""
#Option(name="batch_mode", help=_help, option_style="flag")

pre_cullh="method for culling the line list before measurement"
parser.add_argument("--pre-cull", default="snr", help=pre_cullh)

cullthreshh="a threshold for quick culling of lines for strength"
parser.add_argument("--cull-threshold", default=-2.0, type=float, help=cullthreshh)

post_cullh="method for culling the output measurements"
parser.add_argument("--post-cull", default="none", help=post_cullh)

tgh="a guess at the effective temperature"
parser.add_argument("--start_teff", default=5500.0, type=float, help=tgh)

dwhelp = "default number of angstroms to display" 
parser.add_argument("--display-width", "--dw", type=float, default=3.0, help=dwhelp)

#TODO: replace these options with a kwarg for the individual mode using --fit
fwhelp = "default individual line fit width in angstroms"
parser.add_argument("--fit-width", "--fw", default="average", help=fwhelp)

dgamma_help = "penalize lorentz width values above this threshold"
parser.add_argument("--gamma-max", "--gm", type=float, default=0.0)

read_help = "name of function to load the data files with"
parser.add_argument("--read-func", default="read_spec")

rv_help = "optional radial velocity shift to apply set to 'cc' to estimate via cross correlation with a template"
parser.add_argument("--rv", default=0.0, help=rv_help)

dmaxh="maximum velocity shift to search for by default (in Km/s)"
parser.add_argument("--max-rv", default=500, help=dmaxh)

norm_help="type of normalization to apply to spectra options are ones and auto"
parser.add_argument("--norm", default="auto", help=norm_help)

ctmw_help="how heavily to weight the global continuum when determinging local continuum"
parser.add_argument("--continuum-weight", default=10.0, type=float, help=ctmw_help)

fit_help="the type of spectral fit to run"
parser.add_argument("--fit", default="individual", help=fit_help)

outmulth="consider differences from the median of greater than 1.4*(outlier multiplier)*(median absolute deviation) to constitute an outlier and exclude them from consideration"
parser.add_argument("--outlier-threshold", type=float, default=5.0, help=outmulth)

inmulth="consider differences from the median of less than 1.4*(inlier multiplier)*(median absolute deviation) to constitute core trusted values"
parser.add_argument("--inlier-threshold", type=float, default=1.0, help=inmulth)

iter_help="the fit iteration strategy"
parser.add_argument("--iteration", default="2", help=iter_help)

mout_help="automatically output a moog readable ew file if set"
parser.add_argument("--moog-out", action="store_true", help=mout_help)

featout_help="output a features.pkl if set"
parser.add_argument("--features-out", action="store_true", help=featout_help)

ddir_help="directory of the data to read in"
parser.add_argument("--data-dir", default="", help=ddir_help)

outdir_help="directory to store outputs in"
parser.add_argument("--output-dir", default="", help=outdir_help)

nwhelp="suppress the GUI window"
parser.add_argument("--no-window", "--nw", action="store_true", help=nwhelp)

nsplashh="suppres the splash screen"
parser.add_argument("--no-splash", "--nsplash", "--ns", action="store_true", help=nsplashh)

templ_help="specify template spectra to load from the thimbles/resources/templates directory"
parser.add_argument("--templates", nargs="*", default=[], help=templ_help)

st_help = "a valid python script or expression to execute in the user name space on startup"
parser.add_argument("--startup", default="", help=st_help)

verbose_help = "set verbosity of thimbles (default is False)"
parser.add_argument("-v","--verbosity", action='store_true', dest='verbosity_level', help=verbose_help)

options = parser.parse_args()

if options.verbosity_level:
    tmb.verbosity.set_level('verbose')
else:
    tmb.verbosity.set_level('silent')

if options.no_window:
    options.no_splash = True

#in batch mode always give some output and never open the window
if options.batch_mode:
    options.no_window=True
    options.features_out=True

