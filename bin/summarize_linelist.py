
import numpy as np
import scipy.cluster.vq as vq
import argparse

import matplotlib as mpl
mpl.use("qt4Agg")
import matplotlib.pyplot as plt

import thimbles as tmb
import json
import latbin

parser = argparse.ArgumentParser()
parser.add_argument("linelist")
parser.add_argument("--k-max", default=300, type=int)
parser.add_argument("--delta-log-wv", default=0.025, type=float)
parser.add_argument("--resolution", default=50000, type=float)
parser.add_argument("--delta-ep", default=0.4, type=float)
parser.add_argument("--strength-offset", type=float, default=0.0)
parser.add_argument("--delta-rel-strength", default=0.5, type=float)
parser.add_argument("--max-rel-strength", default=100.0, type=float)
parser.add_argument("--teff", default=5500.0, type=float)
parser.add_argument("--match-isotopes", action="store_true")
parser.add_argument("--output", required=True)
parser.add_argument("--output-mapping")


if __name__ == "__main__":
    args = parser.parse_args()
    full_ll = tmb.io.linelist_io.read_linelist(args.linelist)
    
    lbs = tmb.transitions.lines_by_species(full_ll, match_isotopes=args.match_isotopes)
    
    line_summaries = {}
    ll_indexer = {full_ll[i]:i for i in range(len(full_ll))}
    mapping_dict = {}
    
    for species_id in lbs:
        species_ll = lbs[species_id]
        wvs = np.array([l.wv for l in species_ll])
        eps = np.array([l.ep for l in species_ll])
        psts = np.array([l.pseudo_strength(teff=args.teff) for l in species_ll])
        
        rel_strengths =  np.power(10.0, psts-args.strength_offset)
        max_rl = args.max_rel_strength
        rel_strengths = np.where(rel_strengths <= max_rl, rel_strengths, np.sqrt(rel_strengths - max_rl) + max_rl)

        if False:
            plt.scatter(psts, rel_strengths)
            plt.show()
        
        scaled_ll = np.array([
            np.log10(wvs)/args.delta_log_wv, 
            eps/args.delta_ep,
            rel_strengths/args.delta_rel_strength,
        ]).transpose()
        alat = latbin.ALattice(3)
        
        binned_centers = alat.bin(scaled_ll).mean()
        
        #choose the number of lines to keep
        k_keep = min(args.k_max, len(scaled_ll))
        
        #don't keep degenerate features
        n_unique_wvs_eff = np.unique(np.around(np.log(wvs)*args.resolution))
        k_keep = min(len(n_unique_wvs_eff), k_keep)
        
        #don't keep more features than we have the ability to detect
        
        strength_sum = int(np.sum(np.clip(rel_strengths, 0.0, 1.0)))
        strength_sum = max(1, strength_sum) 
        k_keep = min(strength_sum, k_keep) 
        
        #carry out k-means
        centroids, dist = vq.kmeans(binned_centers.values, k_keep)
        #quantize onto the centroids
        binned_ids, dist = vq.vq(scaled_ll, centroids)
        
        cur_summary = []
        #iterate through the transitions assigned to each centroid
        for i in np.unique(binned_ids):
            group_idxs = np.where(binned_ids == i)[0]
            exemplar_idx = np.argmax(rel_strengths[group_idxs])
            exemplar = species_ll[group_idxs[exemplar_idx]]
            cur_summary.append(exemplar)
            grouped_ll = [species_ll[gi] for gi in group_idxs]
            mapping_dict[ll_indexer[exemplar]] = [ll_indexer[trans] for trans in grouped_ll]
        
        line_summaries[species_id] = sorted(cur_summary, key=lambda x:x.wv)
    
    output_transitions = []
    for sp_key in sorted(line_summaries):
        output_transitions.extend(line_summaries[sp_key])
    
    if not args.output_mapping is None:
        map_out_fname = args.output_mapping
        
        map_file = open(map_out_fname+".json", "w")
        json.dump(mapping_dict, map_file)
        map_file.close()
    
    tmb.io.linelist_io.write_linelist(args.output, output_transitions, file_type="moog")

