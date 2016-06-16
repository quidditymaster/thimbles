
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
import numpy as np
import scipy

import thimbles as tmb

def generate_effective_coordinatizer(
        coordinates,
        rounding_scale,
):
    unique_pos = np.unique(np.around(coordinates/rounding_scale))
    unique_pos *= rounding_scale
    return tmb.coordinatization.ArbitraryCoordinatization(unique_pos)


class SparseMatrixCoordinatizer(object):
    
    def __init__(
            self,
            matrix,
            row_x,
            col,
            rounding_scale=None,
    ):
        self.matrix = matrix.tocsc().sorted_indices()
        self.row_x = row_x
        self.col = col
        if rounding_scale is None:
            x_deltas = scipy.gradient(np.sort(row_x))
            rounding_scale = np.mean(x_deltas)
        assert rounding_scale >= 0
        self.rounding_scale = rounding_scale
        self._coordinatizers = {}
        self._nz_indexes = {}
    
    def get_nz_indexes(self):
        return self.matrix[:, self.col].indices
    
    def get_coordinatization(self):
        coorder = self._coordinatizers.get(self.col)
        if coorder is None:
            nz_x = self.row_x[self.get_nz_indexes()]
            coorder = generate_effective_coordinatizer(
                coordinates = nz_x
            )
            self._coordinatizers.get(self.col)
    
    def set_col(self, col):
        self.col = col
    
    def __call__(self, x):
        coo = self.get_coordinatization()
        return coo.get_index(x)


class SparseDerivativeChart(object):
    
    def __init__(self):
        pass

    
