import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
import numpy as np
from thimbles.charts import MatplotlibCanvas


class ChunkedLineChart(object):
    _plots_initialized = False
    
    def __init__(
            self,
            x,
            y,
            n_lag=21,
            n_chunks=5,
            n_overlap=1,
            chunk_reducer=None,
            pre_processor=None,
            post_processor=None,
            line_kwargs=None,
            ax=None,
    ):
        self.x = x
        self.y = y
        self.n_lag=n_lag
        self.n_chunks=n_chunks
        self.n_overlap=n_overlap
        if line_kwargs is None:
            line_kwargs = {}
        self.line_kwargs = line_kwargs
        
        if chunk_reducer is None:
            chunk_reducer = lambda x: np.mean(x, axis=0)
        self.chunk_reducer = chunk_reducer
            
        self.pre_processor = pre_processor
        self.post_processor = post_processor
        
        if ax is None:
            fig, ax = plt.subplots()
        self.ax = ax
        
        self.update()
    
    def get_chunk_bounding_indexes(self):
        bounds = []
        chunk_edges = np.linspace(0, len(self.y), self.n_chunks+self.n_overlap+1).astype(int)
        
        for i in range(self.n_chunks):
            lb = chunk_edges[i]
            ub = chunk_edges[i+self.n_overlap+1]
            bounds.append((lb, ub))
        return bounds
    
    def get_line_pts(self):
        npts_corr = 2*self.n_lag+1
        line_pts = np.zeros((self.n_chunks, npts_corr, 2))
        chunk_bound_idxs = self.get_chunk_bounding_indexes()
        for chunk_idx in range(self.n_chunks):
            lb, ub = chunk_bound_idxs[chunk_idx]
            min_x = self.x[lb]
            max_x = self.x[ub-1]
            x_approx = np.linspace(min_x, max_x, npts_corr)
            line_pts[chunk_idx, :, 0] = x_approx
            y_chunk = self.y[lb:ub]
            if not self.pre_processor is None:
                y_chunk = self.pre_processor(y_chunk)
            y_red = self.chunk_reducer(y_chunk)
            if not self.post_processor is None:
                y_red = self.post_processor(y_red)
            line_pts[chunk_idx, :, 1] = y_red
        return line_pts
    
    def _initialize_plots(self):
        if not self._plots_initialized:
            line_data = self.get_line_pts()
            self.lines = mpl.collections.LineCollection(
                line_data,
                **self.line_kwargs
            )
            self.ax.add_collection(self.lines)
        self._plots_initialzed = True
    
    def update(self):
        self._initialize_plots()
        self.lines.set_segments(self.get_line_pts())
        self.ax._tmb_redraw=True
    
    def set_x(self, x):
        self.set_data(x, self.y)
    
    def set_y(self, y):
        self.set_data(self.x, y)
    
    def set_data(self, x, y):
        self.x = x
        self.y = y
        self.update()
