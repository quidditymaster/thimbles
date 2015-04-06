
import re
from datetime import datetime

import pandas as pd
import numpy as np

import thimbles as tmb
from thimbles import ptable, atomic_number, atomic_symbol
from thimbles.linelists import LineList

def float_or_nan(val):
    try:
        return float(val)
    except ValueError:
        return np.nan

def read_moog_linelist(fname):
    lines = open(fname).readlines()
    ldat = {"wv":[], "species":[], "ep":[], "loggf":[], "moog_damp":[], "D0":[], "ew":[]}
    for line in lines:
        line = line.split("#")[0]
        moog_cols = [line[i*10:(i+1)*10].strip() for i in range(7)]
        moog_cols = list(map(float_or_nan, moog_cols))
        lspl = line.split()
        if len(lspl) < 4:
            continue
        if np.isnan(moog_cols[0]):
            continue
        if np.isnan(moog_cols[1]):
            continue
        ldat["wv"].append(moog_cols[0])
        ldat["species"].append(moog_cols[1])
        ldat["ep"].append(moog_cols[2])
        ldat["loggf"].append(moog_cols[3])
        ldat["moog_damp"].append(moog_cols[4])
        ldat["D0"].append(moog_cols[5])
        ldat["ew"].append(moog_cols[6])
    ldf = pd.DataFrame(data=ldat)
    return LineList(ldf)

def write_moog_linelist(fname, linelist, comment=None):
    out_file = open(fname,'w')
    
    # write the header line if desired
    if comment is None:
        comment = "#{}".format(datetime.today())
        out_file.write(str(comment).rstrip()+"\n")
    
    fmt_string = "% 10.3f% 10.5f% 10.2f% 10.2f"
    
    for line_idx in range(len(linelist)):
        cline = linelist.iloc[line_idx]
        wv,species,ep,loggf = cline["wv"], cline["species"], cline["ep"], cline["loggf"]
        out_str = fmt_string % (wv, species, ep, loggf)
        for v_str in "moog_damp D0 ew".split():
            bad_value = False
            if not v_str in linelist.columns:
                bad_value = True
            elif np.isnan(cline[v_str]):
                bad_value = True
            if bad_value:
                out_str += 10*" "
            else:
                val = cline[v_str]
                out_str +="{: 10.4f}".format(val)
        out_file.write(out_str)
        out_file.write("\n")
    out_file.close()


def read_moog_ewfind_summary(fname):
    raise NotImplementedError("TODO:")

def read_moog_abfind_summary(fname):
    header = {'info':None,
              'teff':None,
              'logg':None,
              'feh':None,
              'vt':None}
    
    linedata = {}
    infile = open(fname, "rb")
    
    # define the expressions to find
    paramsexp = re.compile("(\d+\.[\d+, *]) +(\d*\.\d+) +([\ ,+,-]\d\.\d+) +(\d*\.\d+)") #teff,logg,feh,vt
    elemidentexp = re.compile(r"Abundance Results for Species [A-Z][a-z]* +I+")
    lineabexp = re.compile(r" *(\d+\.\d+) +(\d+\.\d+) +")
    #statlineexp = re.compile(r"average abundance = +\d\.\d\d +std\. +deviation = +\d\.\d\d")
    
    # modellineexp = re.compile(r"\d+g\d\.\d\dm-?\d\.\d+\v\d")
    currentspecies = None
    
    for i,line in enumerate(infile):
        p = paramsexp.search(line)
        if p is not None:
            teff,logg,feh,vt = [float(st) for st in p.groups()]
            header['teff'] = teff
            header['logg'] = logg 
            header['feh'] = feh 
            header['vt'] = vt 
            continue
        
        if i == 0:
            header['info'] = line.rstrip()
            continue
        
        _l = lineabexp.match(line)
        if _l is not None:
            #print "new linedata", line
            #line is ordered like wv ep logGF EW logrw, abund, del avg
            linedatum = [float(st) for st in line.split()]
            linedata[currentspecies].append(linedatum)
            continue
        
        m = elemidentexp.search(line)
        if m is not None:
            #print "new element", line
            elemline = m.group().split()
            currentspecies = elemline[-2] + " " + elemline[-1]
            #species_id_num = float(elemline[-2]) + 0.1*(int(elemline[-1])-1)
            linedata[currentspecies] = []
            continue
    
    infile.close()
    out_ldat = dict(wv=[],
                    species=[],
                    ep=[],
                    loggf=[],
                    ew=[],
                    abund=[],
                    )
    #batom = Batom()
    for cspecies in list(linedata.keys()):
        species_parts = cspecies.split()
        species_pnum = atomic_number[species_parts[0]]
        species_id = int(species_pnum) + 0.1*(len(species_parts[1])-1)
        for lidx in range(len(linedata[cspecies])):
            ldm = linedata[cspecies][lidx]
            #output line list format Wv, species, ep, logGF, EW, logRW, Abundance, del_avg
            out_ldat["wv"].append(ldm[0])
            out_ldat["species"].append(species_id)
            out_ldat["ep"].append(ldm[1])
            out_ldat["loggf"].append(ldm[2])
            out_ldat["ew"].append(ldm[3])
            out_ldat["abund"].append(ldm[5])
    out_ldat = LineList(pd.DataFrame(data=out_ldat))
    
    return out_ldat


def read_moog_synth_summary(fname, effective_snr=500.0):
    lines = open(fname).readlines()[1:]
    data = []
    for lidx in range(len(lines)):
        try:
            spl = lines[lidx].split()
            if len(spl) != 4:
                continue
            flvals = list(map(float, spl))
            min_wv = flvals[0]
            max_wv = flvals[1]
            wv_delta = flvals[2]
            break
        except Exception:
            pass
    for lidx in range(lidx+1, len(lines)):
        data.extend(list(map(float, lines[lidx].split())))
    flux = 1.0-np.array(data)
    wvs = np.linspace(min_wv, min_wv+(len(flux)-1)*wv_delta, len(flux))
    sflags = tmb.spectrum.SpectrumFlags()
    sflags["normalized"] = True
    eff_ivar = np.repeat(effective_snr**2, len(flux))
    #import pdb; pdb.set_trace()
    spec = tmb.Spectrum(wvs, flux, eff_ivar, flags=sflags)
    return spec
