
import thimbles as tmb
import pandas as pd
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("db_url")
parser.add_argument("--threshold_rel_flux", type=float)
parser.add_argument("--max_rel_flux", type=float)

if __name__ == "__main__":
    args = parser.parse_args()
    db = tmb.ThimblesDB(args.db_url)#"m67_region_1deg.db")
    
    spectra = db.query(tmb.Spectrum).all()
    
    for spec in spectra:
        flux = spec.flux
        norm = spec["norm"].value
        nflux = flux/norm
        ivar_weights = np.where(nflux >= args.max_rel_flux, 0, np.ones(len(flux)))
        ivar_weights = np.where(nflux > args.threshold_rel_flux, args.max_rel_flux-nflux, 1.0)
        
        spec.ivar = spec.ivar*ivar_weights
    
    db.commit()
