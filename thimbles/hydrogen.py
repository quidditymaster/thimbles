import numpy as np
import pandas as pd
import os
from thimbles import resource_dir

data_cols = np.loadtxt(os.path.join(resource_dir, "transition_data", "Hydrogen_lines.txt"), usecols=[0, 1, 2, 3, 5])
hlines = pd.DataFrame(data=dict(wv=data_cols[:, 0], n_lo=data_cols[:, 1], n_up=data_cols[:, 2], ep=data_cols[:, 3], loggf=data_cols[:, 4]))

def get_H_mask(wvs, masking_radius=10.0):
    """a mask to remove wavelengths close to hydrogen features"""
    min_wv = np.min(wvs)
    max_wv = np.max(wvs)
    mask = np.ones(wvs.shape, dtype=bool)
    for line_idx in range(len(hlines)):
        lwv = hlines.iloc[line_idx]["wv"]
        if lwv < (min_wv - masking_radius):
            continue
        if lwv > (max_wv + masking_radius):
            continue
        mask *= np.abs(wvs-lwv) > masking_radius
    return mask
            
            