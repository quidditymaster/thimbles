
from thimblesgui.active_collections import ItemMappedColumn, repr_column
from thimbles.contexts import model_spines
import thimbles as tmb
from . import object_creation_dialogs

namec = ItemMappedColumn(
    "name",
    getter=lambda x: x.name,
    setter=lambda x, y: setattr(x, "name", y),
    string_converter = lambda x: x,
    value_converter=lambda x: x,
)

def none_safe_float_str(val):
    if val is None:
        return "None"
    else:
        return "{:8.3f}".format(val)


#columns for stars
teffc = ItemMappedColumn(
    "Teff",
    getter=lambda x: x.teff,
    value_converter=lambda x: "{: 8.1f}".format(x),
    setter=lambda x, y: setattr(x, "teff", y),
    string_converter=float,
)
loggc = ItemMappedColumn(
    "log(g)",
    getter=lambda x: x.logg,
    value_converter=lambda x: "{: 8.3f}".format(x),
    setter=lambda x, y: setattr(x, "logg", y),
    string_converter=float,
)
metalicityc = ItemMappedColumn(
    "[M/H]",
    getter=lambda x: x.metalicity,
    value_converter=lambda x: "{: 8.3f}".format(x),
    setter=lambda x, y: setattr(x, "metalicity", y),
    string_converter=float,
)
massc = ItemMappedColumn(
    "Mass",
    getter=lambda x: x.mass,
    value_converter=none_safe_float_str,
    setter=lambda x, y: setattr(x, "mass", y),
    string_converter=float,
)
vmicroc = ItemMappedColumn(
    "vmicro",
    getter=lambda x: x.vmicro,
    value_converter=lambda x: "{: 8.3f}".format(x),
    setter=lambda x, y: setattr(x, "vmicro", y),
    string_converter=float,
)
vmacroc = ItemMappedColumn(
    "vmacro",
    getter=lambda x: x.vmacro,
    value_converter=lambda x: "{: 8.3f}".format(x),
    setter=lambda x, y: setattr(x, "vmacro", y),
    string_converter=float,
)
vsinic = ItemMappedColumn(
    "vsini",
    getter=lambda x: x.vsini,
    value_converter=lambda x: "{: 8.3f}".format(x),
    setter=lambda x, y: setattr(x, "vsini", y),
    string_converter=float,
)
ldarkc = ItemMappedColumn(
    "ldark",
    getter=lambda x: x.ldark,
    value_converter=lambda x: "{: 8.3f}".format(x),
    setter=lambda x, y: setattr(x, "ldark", y),
    string_converter=float,
)

star_columns = [namec, teffc, loggc, metalicityc, vmicroc, vmacroc, vsinic, ldarkc, massc]


#columns for electronic transitions
wvc = ItemMappedColumn(
    "wavelength",
    getter=lambda x: x.wv,
    value_converter=lambda x: "{: 8.3f}".format(x),
    setter=lambda x, y: setattr(x, "wv", y),
    string_converter=float,
)
ionc = ItemMappedColumn(
    "Ion",
    getter=lambda x: x.ion,
    value_converter=lambda x:x.symbol,
)
epc = ItemMappedColumn(
    "E.P.",
    getter=lambda x: x.ep,
    value_converter=lambda x: "{: 8.3f}".format(x)
)
loggfc = ItemMappedColumn(
    "log(gf)", getter=lambda x: x.loggf,
    value_converter=lambda x: "{: 8.3f}".format(x)
)

base_transition_columns = [wvc, ionc, epc, loggfc]

isotopec = ionc = ItemMappedColumn(
    "isotope",
    getter=lambda x: x.ion.isotope,
    value_converter=lambda x:str(x),
)

def float_str_or_empty(x):
    if x is None:
        return ""
    else:
        return "{:8.3f}".format(x)

starkdc = ItemMappedColumn(
    "stark damp",
    getter=lambda x: x.damp.stark,
    value_converter=float_str_or_empty
)
waalsdc = ItemMappedColumn(
    "Vanderwaals damp",
    getter=lambda x: x.damp.waals,
    value_converter=float_str_or_empty
)
raddc = ItemMappedColumn(
    "radiative damp",
    getter=lambda x: x.damp.rad,
    value_converter=float_str_or_empty
)
empiricalc = ItemMappedColumn(
    "empirical damp",
    getter=lambda x: x.damp.empirical,
    value_converter=float_str_or_empty
)

full_transition_columns = [
    wvc,
    ionc,
    isotopec,
    epc,
    loggfc,
    starkdc,
    waalsdc,
    raddc,
    empiricalc
]

def extract_bound(wvs):
    return wvs[0], wvs[-1]

wvboundc = ItemMappedColumn(
    "wv span",
    getter=lambda x: extract_bound(x.wv),
    value_converter=lambda x: "{:8.0f} < wv < {:8.0f}".format(*x),
)

star_contextualizer = model_spines["stars"]
starc = ItemMappedColumn(
    "star",
    getter=lambda x:x.source,
    value_converter = lambda x : "{}".format(x),
    string_converter = lambda x: star_contextualizer.find(tag=x)[0],
    setter=lambda x, y: setattr(x, "source", y),
)


Aperture = tmb.spectrographs.Aperture
aperture_contextualizer = model_spines["apertures"]
aperturec = ItemMappedColumn(
    "aperture",
    getter=lambda x:x.aperture,
    value_converter = lambda x : "{}".format(x),
    string_converter= lambda x: aperture_contextualizer.find(tag=x)[0],
    setter=lambda x, y: setattr(x, "aperture", y),
)


Order = tmb.spectrographs.Order
order_contextualizer = model_spines["orders"]
orderc = ItemMappedColumn(
    "order",
    getter=lambda x:x.order,
    value_converter = lambda x : "{}".format(x),
    string_converter=lambda x : order_contextualizer.find(tag=x)[0],
    setter=lambda x, y: setattr(x, "order", y)
)


Chip = tmb.spectrographs.Chip
chipc = ItemMappedColumn(
    "chip",
    getter=lambda x:x.chip,
    value_converter = lambda x : "{}".format(x),
    string_converter=lambda x: chip_contextualizer.find(tag=x)[0],
    setter=lambda x, y: setattr(x, "chip", y)
)

Exposure = tmb.observations.Exposure
exposure_contextualizer = model_spines["exposures"]
exposurec = ItemMappedColumn(
    "exposure",
    getter=lambda x:x.exposure,
    value_converter = lambda x : "{}".format(x),
    string_converter = lambda x : exposure_contextualizer.find(tag=x)[0],
    setter = lambda x, y: setattr(x, "exposure", y)
)

snrc = ItemMappedColumn(
    "median SNR",
    getter = lambda x: x.snr_estimate(),
    value_converter = lambda x: "{:8.2f}".format(x),
)

resolutionc = ItemMappedColumn(
    "median resolution",
    getter = lambda x: x.resolution_estimate(),
    value_converter = lambda x: "{:8.2f}".format(x),
)

rvc = ItemMappedColumn(
    "radial velocity",
    getter = lambda x: x["rv"].value,
    setter = lambda x, v: setattr(x["rv"], "value" , v),
    value_converter = lambda x: "{:5.2f} Km/s".format(x),
    string_converter = float
)

vhelioc = ItemMappedColumn(
    "rv helio",
    getter = lambda x: x["delta_helio"].value,
    setter = lambda x, v: setattr(x["delta_helio"], "value" , v),
    value_converter = lambda x: "{:5.2f} Km/s".format(x),
    string_converter = float
)

spectrum_columns = [
    wvboundc,
    snrc,
    starc,
    resolutionc,
    aperturec,
    orderc,
    chipc,
    exposurec,
    rvc,
    vhelioc
]
