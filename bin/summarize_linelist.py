
import numpy as np
import scipy.cluster.vq as vq
import argparse

import thimbles as tmb
import latbin

parser = argparse.ArgumentParser()
parser.add_argument("linelist")
parser.add_argument("--k-max", default=200, type=int)
parser.add_argument("--delta-log-wv", default=0.025, type=float)
parser.add_argument("--delta-ep", default=0.4, type=float)
parser.add_argument("--delta-pseudo-strength", default=0.6, type=float)
parser.add_argument("--teff", default=5500.0, type=float)
parser.add_argument("--match-isotopes", action="store_true")
parser.add_argument("--output", required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    full_ll = tmb.io.linelist_io.read_linelist(args.linelist)
    
    lbs = tmb.transitions.lines_by_species(full_ll, match_isotopes=args.match_isotopes)
    
    line_summaries = {}
    for species_id in lbs:
        species_ll = lbs[species_id]
        wvs = np.array([l.wv for l in species_ll])
        eps = np.array([l.ep for l in species_ll])
        psts = np.array([l.pseudo_strength(teff=args.teff) for l in species_ll])
        
        scaled_ll = np.array([
            np.log10(wvs)/args.delta_log_wv, 
            eps/args.delta_ep, 
            psts/args.delta_pseudo_strength,
        ]).transpose()
        alat = latbin.ALattice(3)
        
        binned_centers = alat.bin(scaled_ll).mean()
        
        k_keep = min(args.k_max, int(np.sqrt(len(binned_centers))))
        k_keep = max(1, k_keep)
        centroids, dist = vq.kmeans(binned_centers.values, k_keep)
        binned_ids, dist = vq.vq(scaled_ll, centroids)
        
        cur_summary = []
        exemplar_dists = []
        for i in np.unique(binned_ids):
            exemplar_dist = np.inf
            exemplar = None
            for j in range(len(species_ll)):
                if binned_ids[j] == i:
                    cur_dist = np.sum((scaled_ll[j] - centroids[i])**2)
                    if cur_dist < exemplar_dist:
                        exemplar_dist = cur_dist
                        exemplar = species_ll[j]
            cur_summary.append(exemplar)
            exemplar_dists.append(exemplar_dist)
        
        line_summaries[species_id] = sorted(cur_summary, key=lambda x:x.wv)
    
    output_transitions = []
    for sp_key in sorted(line_summaries):
        output_transitions.extend(line_summaries[sp_key])
    
    tmb.io.linelist_io.write_linelist(args.output, output_transitions, file_type="moog")
