
import numpy as np
import astropy.io.fits as fits
import os, sys
import argparse
import scipy

import thimbles as tmb

parser = argparse.ArgumentParser()
parser.add_argument("--ra", required=True, nargs="*")
parser.add_argument("--dec", required=True, nargs="*")
parser.add_argument("--selection-radius", default=0.2, type=float)
parser.add_argument("--fallback-rv", default=0.0, type=float)

redux_dir = "/uufs/chpc.utah.edu/common/home/sdss01/apogeework/apogee/spectro/redux/r6"
all_visit_fpath = "/uufs/chpc.utah.edu/common/home/sdss01/apogeework/apogee/spectro/redux/r6/allVisit-l30e.2.fits"

#redux_dir = "/uufs/chpc.utah.edu/common/home/sdss00/bosswork/apogee/spectro/redux/r5"
#all_visit_fpath = "/uufs/chpc.utah.edu/common/home/sdss00/bosswork/apogee/spectro/redux/r5/allVisit-v603.fits"

def dms_to_dd(deg, mins=0, secs=0):
    return deg + mins/60.0 + secs/3600.0

def hms_to_dd(h, m=0, s=0):
    return (360.0/24.0)*(h+m/60.0 + s/3600.0)

def extract_info(row, columns):
    info = {}
    for col in columns:
        info[col] = row[col]
    return info

phot_cols = "J H K WASH_M WASH_T2 DDO51 IRAC_3_6 IRAC_4_5 IRAC_5_8 IRAC_8_0 WISE_4_5 TARG_4_5".split()
phot_errs = "J_ERR H_ERR K_ERR SRC_H WASH_M_ERR WASH_T2_ERR DDO51_ERR IRAC_3_6_ERR IRAC_4_5_ERR IRAC_5_8_ERR IRAC_8_0_ERR WISE_4_5_ERR TARG_4_5_ERR SFD_EBV".split()
star_info_cols = "PMRA PMDEC PM_SRC FIELD".split()

if __name__ == "__main__":
    args = parser.parse_args()
    
    all_visit_hdul = fits.open(all_visit_fpath)
    all_visit = all_visit_hdul[1].data
    
    #col_names = [all_fisit_hdul[1].header[hv] for hv in all_visit_hdul[1].header if isinstance(hv, str) and "TTYPE" in hv]
    
    ras = all_visit["RA"]
    decs = all_visit["DEC"]
    if len(args.ra) > 1:
        center_ra = hms_to_dd(*map(float, args.ra))
    else:
        center_ra = args.ra[0]
    center_dec = dms_to_dd(*map(float, args.dec))
    ddeg_offset = np.sqrt((ras-center_ra)**2 + (decs - center_dec)**2)
    pmask = ddeg_offset <= args.selection_radius
    
    #import pdb; pdb.set_trace()
    #pmask = np.zeros(len(all_visit), dtype=bool)
    #for plate_str in args.plates:
    #    pmask += all_visit["PLATE"] == plate_str
    
    visit_idxs = np.where(pmask)[0]
    if len(visit_idxs) == 0:
        print("no visits found, quitting")
        sys.exit()
    else:
        print("found {} visits".format(len(visit_idxs)))    
    
    sources = {}
    #slices = {}
    exposures = {}
    fibers = {}
    chips = [tmb.spectrographs.Chip("ccd{}".format(i)) for i in range(3)]
    
    spectra = []
    for vis_idx in visit_idxs:
        crow = all_visit[vis_idx]
        
        cmjd = "{:5d}".format(crow["MJD"])
        cplate = crow["PLATE"]
        cfname = crow["FILE"]
        cur_p = os.path.join(redux_dir, "apo25m", cplate, cmjd, cfname)
        try:
            hdul = fits.open(cur_p)
        except FileNotFoundError:
            print("Warning no file {} skipping".format(cur_p))
        
        csource_name = crow["APOGEE_ID"]
        csource = sources.get(csource_name)
        if csource is None:
            cra, cdec = crow["RA"], crow["DEC"]
            #csource = tmb.sources.Source(name=csource_name, ra=cra, dec=cdec)
            star_info = {}
            star_info["phot"] = extract_info(crow, phot_cols)
            star_info["phot_err"] = extract_info(crow, phot_errs)
            star_info["misc"] = extract_info(crow, star_info_cols)
            star_info["ap_starflags"] = crow["STARFLAGS"] 
            csource = tmb.star.Star(name=csource_name, ra=cra, dec=cdec, info=star_info)
            sources[csource_name] = csource
        
        exposure_name = "{}".format((cplate, cmjd))
        c_exp = exposures.get(exposure_name)
        if c_exp is None:
            c_exp = tmb.observations.Exposure(name=exposure_name)
            exposures[exposure_name] = c_exp
        
        cfiber_id = crow["FIBERID"]
        cfiber = fibers.get(cfiber_id)
        if cfiber is None:
            cfiber = tmb.spectrographs.Aperture("fiber{}".format(cfiber_id))
            fibers[cfiber_id] = cfiber
        
        vrel_tot = crow["VREL"]
        vhelio_center = crow["VHELIO"]
        delta_helio = vrel_tot - vhelio_center
        
        if np.isnan(delta_helio):
            delta_helio = args.fallback_rv
        if np.isnan(vhelio_center):
            vhelio_center = args.fallback_rv
        
        for chip_idx in range(3):
            c_chip = chips[chip_idx]
            flux = hdul[1].data[chip_idx, ::-1].copy()
            wvs = hdul[4].data[chip_idx, ::-1].copy()
            var = hdul[2].data[chip_idx, ::-1]**2
            mean_pixel_delta = np.median(scipy.gradient(wvs))
            new_spec = tmb.Spectrum(
                wvs, 
                flux, 
                var=var, 
                source=csource,
                delta_helio=delta_helio,
                helio_shifted=False,
                rv=vhelio_center,
                rv_shifted=False,
                aperture=cfiber,
                exposure=c_exp,
                chip=c_chip,
                cdf_type=2,
                cdf_kwargs={"box_width":mean_pixel_delta},
            )
            spectra.append(new_spec)
            if len(spectra) % 100 == 0:
                print("{} spectra processed".format(len(spectra)))
        
        hdul.close()
