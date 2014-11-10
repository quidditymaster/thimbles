import pandas as pd
import numpy as np

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
        moog_cols = map(float_or_nan, moog_cols)
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

def write_moog_linelist(fname, line_data, a_to_ma=True):
    out_file = open(fname,'w')
    
    # write the header line if desired
    if comment is None:
        comment = "#{}".format(datetime.today())
        out_file.write(str(comment).rstrip()+"\n")
            
    fmt_string = "% 10.3f% 10.5f% 10.2f% 10.2f"
    ew_scale_factor = 1.0
    if a_to_ma: # convert from anstroms to milli angstroms
        ew_scale_factor = 1000.0
        for line_idx in range(len(line_data)):
            cline = line_data.iloc[line_idx]
            wv,species,ep,loggf = cline["wv"], cline["species"], cline["ep"], cline["loggf"]
            out_str = fmt_string % (wv, species, ep, loggf)
            for v_str in "moog_damp D0 ew".split():
                bad_value = False
                if not v_str in line_data.columns:
                    bad_value = True
                elif np.isnan(cline[v_str]):
                    bad_value = True
                if bad_value:
                    out_str += 10*" "
                else:
                    if v_str == "ew":
                        val = cline[v_str]*ew_scale_factor
                    else:
                        val = cline[v_str]
                    out_str +="{: 10.4f}".format(val)
                out_str += "\n"
            out_file.write(out_str)
        out_file.close()
