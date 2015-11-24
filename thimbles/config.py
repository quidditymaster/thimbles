from .options import Option, opts
import thimbles as tmb


Option("wavelengths", option_style="parent_dict")
Option("medium", default="vacuum", parent="wavelengths")
Option("units", default="Angstroms", parent="wavelengths")


Option("spectra", option_style="parent_dict")
Option("io", parent="spectra", option_style="parent_dict")
Option("read_default", parent="spectra.io", runtime_str="tmb.io.read_spec")
Option("write_default", parent="spectra.io", runtime_str="tmb.io.write_spec")


Option("modeling", option_style="parent_dict")
Option("min_wv", parent="modeling")
Option("max_wv", parent="modeling")
Option("resolution", parent="modeling", default=3e5)


Option("database", option_style="parent_dict")
Option("dialect", default="sqlite", parent="database")
Option("echo_sql", default=False, parent="database")


#matplotlib options
_help = "parent option for setting matplotlib style related options"
mpl_style = Option("mpl_style", option_style="parent_dict", help_=_help)
lw = Option(name="line_width", default=1.5, parent=mpl_style, help="default line width")

#spectrum display related options
_help=\
"""options relating to how spectra will be displayed by default
"""
Option(name="spec_display", option_style="parent_dict", help_=_help)

_help=\
"""The logarithm of the ratio of default display window in angstroms
to the central wavelength being displayed.
"""
Option(name="log_window_width", default=-4.0, parent="spec_display", help_=_help)


Option("GUI", option_style="parent_dict")
Option(
    name="show_splash",
    default=True,
    help="show/suppress the splash screen\n",
    parent="GUI"
)
Option(
    name="project_path",
    default="",
    parent="GUI"
)
