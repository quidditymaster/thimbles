from datetime import datetime 
import re
import numpy as np
import pandas as pd
import thimbles as tmb

from thimbles.stellar_atmospheres import solar_abundance as ptable

def float_or_nan(val):
    try:
        return float(val)
    except ValueError:
        return np.nan

def read_linelist(fname, file_type="moog"):
    lines = open(fname).readlines()
    ldat = {"wv":[], "species":[], "Z":[], "ion":[], 
            "ep":[], "loggf":[], "ew":[],
            "rad_damp":[], "stark_damp":[], "waals_damp":[],
            "moog_damp":[], "D0":[],
            }
    if file_type.lower() == "moog":
        for line in lines:
            try:
                moog_cols = [line[i*10:(i+1)*10].strip() for i in range(7)]
                wv = float(moog_cols[0])
                species = float(moog_cols[1])
                sp_split = moog_cols[1].split(".")
                z = int(sp_split[0])
                ion = int(sp_split[1][0])
                #A=sp_split[1][1:]
                ep = float(moog_cols[2])
                loggf = float(moog_cols[3])
                if moog_cols[4] != "":
                    moog_damp = float(moog_cols[4])
                else:
                    moog_damp = np.nan
                if moog_cols[5] != "":
                    d0 = float(moog_cols[5])
                else:
                    d0 = np.nan
                if moog_cols[6]:
                    ew = float(moog_cols[6])
                else:
                    ew = 0
                z = int(species)
                ion = int(10*(species-z))
                
                rad_damp = np.nan
                stark_damp = np.nan
                waals_damp = np.nan
            except ValueError as e:
                print e
                continue
            ldat["wv"].append(wv)
            ldat["species"].append(z+(ion-1)*0.1)
            ldat["Z"].append(z)
            #TODO: add a nucleon number column "A"
            ldat["ion"].append(ion)
            ldat["ep"].append(ep)
            ldat["loggf"].append(loggf)
            ldat["ew"]=ew
            ldat["rad_damp"].append(rad_damp)
            ldat["stark_damp"].append(stark_damp)
            ldat["waals_damp"].append(waals_damp)
            ldat["moog_damp"].append(moog_damp)
            ldat["D0"] = d0
    elif file_type == "vald":
        input_re = re.compile("'[A-Z][a-z] [12]', ")
        for line in lines:
            m = input_re.match(line)
            if m is None:
                continue
            spl = line.rstrip().split(",")
            species_name, ion_number = spl[0].replace("'", "").split()
            ion_number = int(ion_number)
            proton_number = ptable[species_name]["z"]
            #species_id = proton_number + 0.1*(int(ion_number)-1)
            wv, loggf, elow, jlo, eup, jup = map(float, spl[1:7])
            l_lande, u_lande, m_lande = map(float_or_nan, spl[8:11])
            rad_damp, stark_damp, waals_damp = map(float_or_nan, spl[12:15])
            ldat["wv"].append(wv)
            ldat["species"].append(proton_number+(ion_number-1)*0.1)
            ldat["Z"].append(proton_number)
            #TODO: add a nucleon number column "A"
            ldat["ion"].append(ion_number)
            ldat["ep"].append(elow)
            ldat["loggf"].append(loggf)
            ldat["rad_damp"].append(rad_damp)
            ldat["stark_damp"].append(stark_damp)
            ldat["waals_damp"].append(waals_damp)
            #and the parameters not present
            ldat["moog_damp"].append(np.nan)
            ldat["ew"].append(0.0)
            ldat["D0"].append(np.nan)
    return pd.DataFrame(data=ldat)


def write_linelist(line_data, filename, file_type="moog", comment=None):
    """write out a linelist"""
    if file_type == "moog":
        out_file = open(filename,'w')
        
        # write the header line if desired
        if comment is None:
            comment = "#"+str(datetime.today())
        out_file.write(str(comment).rstrip()+"\n")
        
        fmt_string = "% 10.3f% 10.5f% 10.2f% 10.2f"
        for line_idx in range(len(line_data)):
            cline = line_data.iloc[line_idx]
            wv,species,ep,loggf = cline["wv"], cline["species"], cline["ep"], cline["loggf"]
            out_str = fmt_string % (wv, species, ep, loggf)
            for v_str in "vwdamp d0 ew".split():
                if cline[v_str] != np.nan:
                    out_str += 10*" "
            out_str += "\n"
            out_file.write(out_str)    
        out_file.close()

def write_moog_from_features(filename, features):
    llout = tmb.stellar_atmospheres.utils.moog_utils.write_moog_lines_in(filename)
    for feat in features:
        wv=feat.wv
        spe=feat.species
        loggf = feat.loggf
        ep = feat.ep
        ew = 1000*feat.eq_width
        if feat.flags["use"]:
            llout.add_line(wv, spe, ep, loggf, ew=ew, comment=feat.note)
    llout.close()
