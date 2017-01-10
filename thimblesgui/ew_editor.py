
import numpy as np
import scipy
import matplotlib as mpl
from matplotlib.collections import LineCollection

from thimblesgui import QtCore, QtGui, QtWidgets, Qt
from thimblesgui.mplwidget import MatplotlibWidget
from thimblesgui.prevnext import PrevNext
from thimblesgui.selection_charts import TransitionMarkerChart

import thimbles as tmb

def generate_effective_coordinatizer(
        coordinates,
        rounding_scale,
):
    unique_pos = np.unique(np.around(coordinates/rounding_scale))
    unique_pos *= rounding_scale
    return tmb.coordinatization.ArbitraryCoordinatization(unique_pos)


def sparse_indexes_to_dense_bounds(sparse_indexes, forced_breaks):
    fb_set = set(forced_breaks)
    bounds = []
    clb = None
    for i in range(len(sparse_indexes)-1):
        cind = sparse_indexes[i]
        if clb is None:
            clb = cind
        if (sparse_indexes[i+1] - cind) > 1:
            bounds.append((clb, cind))
            clb = None
        elif cind in fb_set:
            bounds.append((clb, cind))
            clb = cind
    return bounds

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
                coordinates = nz_x,
                rounding_scale=self.rounding_scale
            )
            self._coordinatizers[self.col] = coorder
        return coorder
    
    def set_col(self, col):
        self.col = col
    
    def __call__(self, x):
        coo = self.get_coordinatization()
        return coo.get_index(x)


class MultiLineChart(object):
    
    def __init__(
            self,
            segments,
            ax,
            line_kwargs=None,
    ):
        self.segments = segments
        if line_kwargs is None:
            line_kwargs = {}
        self.lines = LineCollection(segments=segments, **line_kwargs)
        self.ax = ax
        self.ax.add_collection(self.lines)
    
    def set_segments(self, segments):
        self.segments = segments
        self.update()
    
    def update(self):
        self.lines.set_segments(self.segments)
        self.ax.figure._tmb_redraw=True


class TinePicker(object):
    
    def __init__(self, tol=0.1):
        self.tol = tol
    
    def __call__(self, artist, mouseevent):
        print("picking")
        xp, yp = mouseevent.xdata, mouseevent.ydata
        line_segs = artist.get_segments()
        if len(line_segs) == 0:
            return None
        x_dists = np.abs(line_segs[:, -1, 0]-xp)
        min_dist_idx = np.argmin(x_dists)
        if x_dists[min_dist_idx] < self.tol:
            print("picked! {}".format(min_dist_idx))
            return min_dist_idx
        #for seg_idx in range(len(line_segs)):
        #    seg_pts = line_segs[seg_idx]
        #    x_loc = seg_pts[-1, 0]
        #    dist = np.abs(x_loc


class ExemplarForkDiagram(object):
    _segments = None
    
    def __init__(
            self,
            exemplar,
            transitions,
            locator_func,
            tine_lengths,
            tine_max,
            handle_max,
            handle_min,
            handle_kwargs,
            tine_kwargs,
            handle_picker,
            tine_picker,
            ax,
    ):
        self.ax = ax
        self.exemplar = exemplar
        self.transitions = transitions
        self.locator_func = locator_func
        self.tine_lengths = tine_lengths
        
        self.tine_max = tine_max
        self.handle_max = handle_max
        self.handle_min = handle_min
        self.handle_picker = handle_picker
        self.tine_picker = tine_picker
        
        self.handle ,= self.ax.plot(*self.get_handle_pts(), **handle_kwargs)
        self.tines = LineCollection(
            segments=self.get_segments(),
            **tine_kwargs
        )
        self.ax.add_collection(self.tines)
    
    def set_data(self, exemplar, transitions, tine_lengths):
        self.exemplar=exemplar
        self.transitions=transitions
        self.tine_lengths=tine_lengths
        self.update()
    
    def get_handle_pts(self):
        handle_x = self.locator_func(self.exemplar)
        return ([handle_x, handle_x], [self.handle_min, self.handle_max])
    
    def get_segments(self):
        handle_x = self.locator_func(self.exemplar)
        n_trans = len(self.transitions)
        x_vals = [self.locator_func(trans) for trans in self.transitions]
        y_bottom = [self.tine_max - self.tine_lengths[tine_idx] for tine_idx in range(len(self.transitions))]
        
        pts = np.zeros((n_trans, 3, 2))
        pts[:, 0, 0] = handle_x
        pts[:, 1, 0] = x_vals
        pts[:, 2, 0] = x_vals
        pts[:, 0, 1] = self.handle_min
        pts[:, 1, 1] = self.tine_max
        pts[:, 2, 1] = y_bottom
        return pts
    
    def update(self):
        self.handle.set_data(*self.get_handle_pts())
        self.tines.set_segments(self.get_segments())
        self.ax.figure._tmb_redraw=True


class WidthsEditor(QtWidgets.QMainWindow):
    
    def __init__(
            self,
            star,
            transition_indexer,
            exemplar_indexer,
            exemplar_map,
            selection,
            x_pad=0.2,
            parent=None,
    ):
        super().__init__(parent=parent)
        self.selection = selection
        self.x_pad = x_pad
        
        self.star = star
        self.setWindowTitle("EW GoF for {}".format(star.name))
        self.spectra = star.spectroscopy
        
        self.transition_indexer = transition_indexer
        self.exemplar_indexer = exemplar_indexer
        self.exemplar_map = exemplar_map
        self.exemplar_index = 0
        
        self.strength_matrix = star["strength_matrix"].value
        flux_params = [spec["obs_flux"] for spec in self.spectra]
        norm_params = [spec["norm"] for spec in self.spectra]
        wv_params = [spec["rest_wvs"] for spec in self.spectra]
        self.flux_params = flux_params
        self.norm_params = norm_params
        self.wv_params = wv_params
        
        parameter_break_indexes = [0]
        cur_break_idx = 0
        for i in range(len(flux_params)):
            cur_break_idx += len(flux_params[i].value)
            parameter_break_indexes.append(cur_break_idx)
        
        self.parameter_break_indexes = np.array(parameter_break_indexes)
        
        self.ew_param = star["thermalized_widths_vec"]
        deriv_matrix = tmb.modeling.derivatives.deriv(
            flux_params,
            [self.ew_param],
        )
        self.update_stacked_vecs()
        
        self.coordinate_map = SparseMatrixCoordinatizer(
            matrix=deriv_matrix,
            row_x=self.stacked_wvs,
            col=self.exemplar_index,
            rounding_scale=1.0,
        )
        
        self.flux_plot_widget = MatplotlibWidget(
            nrows=1,
            parent=self,
            mpl_toolbar=True,
        )
        self.flux_ax = self.flux_plot_widget.ax
        self.resid_plot_widget = MatplotlibWidget(
            nrows=1,
            parent=self,
            sharex=self.flux_ax,
            mpl_toolbar=False,
        )
        self.resid_ax = self.resid_plot_widget.ax
        self.setCentralWidget(self.flux_plot_widget)
        
        self.attach_as_dock("residuals", self.resid_plot_widget, Qt.BottomDockWidgetArea)
        
        self.prevnext = PrevNext(parent=self)
        self.prevnext.prev.connect(self.on_prev)
        self.prevnext.next.connect(self.on_next)
        self.attach_as_dock("ew controls", self.prevnext, Qt.RightDockWidgetArea)
        
        self.update_dense_bounds()
        self.make_charts()
        self.update_x_limits()
        self.flux_ax.set_ylim(0.5, 1.15)
        self.resid_ax.set_ylim(-5, 5)
        
        exemplar_channel = self.selection.channels["exemplar"]
        exemplar_channel.changed.connect(self.on_exemplar_changed)

        transition_channel = self.selection.channels["transition"]
        transition_channel.changed.connect(self.on_transition_changed)
        cid = self.flux_ax.figure.canvas.mpl_connect("button_press_event", self.on_click)
    
    def on_click(self, event):
        fork_segs = self.fork_chart.tines.get_segments()
        transitions = self.fork_chart.transitions
        nsegs = len(transitions)
        x_values = [fork_segs[i][-1, 0] for i in range(nsegs)]
        y_values = [fork_segs[i][-1, 1] for i in range(nsegs)]
        xpos, ypos = event.xdata, event.ydata
        xdists = np.abs(x_values - xpos)
        min_dist_idx = np.argmin(xdists)
        if xdists[min_dist_idx] < 0.1:
            trans = transitions[min_dist_idx]
            self.set_selected_transition(trans)
    
    def on_exemplar_changed(self):
        print("on exemplar changed called")
        exemplar = self.selection["exemplar"]
        if not exemplar is None:
            exemplar_index = self.exemplar_indexer[exemplar]
            self.set_exemplar_index(exemplar_index)
    
    def on_transition_changed(self):
        print("on transition changed")
        transition = self.selection["transition"]
        self.set_selected_transition(transition)
    
    def set_selected_transition(self, transition):
        self.trans_marker_flux.set_transition(transition)
        self.trans_marker_resid.set_transition(transition)
        self.selection["transition"] = transition
    
    def update_stacked_vecs(self):
        self.stacked_wvs = np.hstack([wvp.value.coordinates for wvp in self.wv_params])
        self.stacked_norms = np.hstack([normp.value for normp in self.norm_params])
        self.stacked_models = np.hstack([fp.value for fp in self.flux_params])
        self.stacked_models /= self.stacked_norms
        self.stacked_flux = np.hstack([spec.flux for spec in self.spectra])
        self.stacked_flux /= self.stacked_norms
        
        self.stacked_ivar = np.hstack([spec.ivar for spec in self.spectra])
        self.stacked_ivar *= self.stacked_norms**2
    
    def update_dense_bounds(self):
        cur_nzi = self.coordinate_map.get_nz_indexes()
        self.dense_bound_indexes = sparse_indexes_to_dense_bounds(cur_nzi, forced_breaks=self.parameter_break_indexes)
    
    def update_x_limits(self):
        flsegs = self.model_chart.lines.get_segments()
        if len(flsegs) > 0:
            x_min = np.min([seg[0, 0] for seg in flsegs if len(seg) > 0]) - self.x_pad
            x_max = np.max([seg[-1, 0] for seg in flsegs if len(seg) > 0]) + self.x_pad
        else:
            x_min = -self.x_pad
            x_max = self.x_pad
        self.flux_ax.set_xlim(x_min, x_max)
    
    def set_exemplar_index(self, index):
        self.exemplar_index = index
        self.coordinate_map.set_col(index)
        self.update_dense_bounds()
        self.update_charts()
        self.update_x_limits()
        exemplar = self.exemplar_indexer[self.exemplar_index]
        self.selection["exemplar"] = exemplar
    
    def dense_to_segments(self, vec):
        segments = []
        for lbi, ubi in self.dense_bound_indexes:
            eff_x = self.coordinate_map(self.stacked_wvs[lbi:ubi])
            y = vec[lbi:ubi]
            segments.append(np.stack([eff_x, y], axis=1))
        return segments
    
    def update_charts(self):
        flux_segs = self.dense_to_segments(self.stacked_flux)
        model_segs = self.dense_to_segments(self.stacked_models)
        resid_segs = self.dense_to_segments((self.stacked_models-self.stacked_flux)*np.sqrt(self.stacked_ivar))
        self.data_chart.set_segments(flux_segs)
        self.model_chart.set_segments(model_segs)
        self.resid_chart.set_segments(resid_segs)
        
        self.fork_chart.set_data(*self.get_fork_data())
    
    def get_exemplar(self):
        return self.exemplar_indexer[self.exemplar_index]
    
    def get_fork_data(self):
        exemplar = self.get_exemplar()
        smat_col = self.strength_matrix[:, self.exemplar_index].tocsc()
        transitions = self.exemplar_map.get(exemplar)
        transition_to_rel_strength = {self.transition_indexer[smat_col.indices[i]]:smat_col.data[i] for i in range(len(transitions))}
        tine_lengths = [0.25*transition_to_rel_strength[trans] for trans in transitions]
        return exemplar, transitions, tine_lengths
    
    def make_charts(self):
        flux_segs = self.dense_to_segments(self.stacked_flux)
        model_segs = self.dense_to_segments(self.stacked_models)
        resid_segs = self.dense_to_segments(
            (self.stacked_models-self.stacked_flux)*np.sqrt(self.stacked_ivar)
        )
        self.data_chart = MultiLineChart(
            flux_segs,
            ax=self.flux_ax,
            line_kwargs=dict(
                color="k",
                alpha=0.5,
                lw=2.0
            )
        )
        self.model_chart = MultiLineChart(
            model_segs,
            ax=self.flux_ax,
            line_kwargs=dict(
                color="orange",
                lw=2.0,
                alpha=0.6
            )
        )
        self.resid_chart = MultiLineChart(resid_segs, ax=self.resid_ax)
        
        exemplar, transitions, tine_lengths = self.get_fork_data()
        self.fork_chart = ExemplarForkDiagram(
            exemplar=exemplar,
            transitions=transitions,
            locator_func = self.transition_locator,
            tine_lengths=tine_lengths,
            handle_max=1.15,
            handle_min=1.05,
            tine_max=1.0,
            handle_kwargs=dict(
                color="r",
            ),
            tine_kwargs=dict(
                color="r",
            ),
            handle_picker=None,
            tine_picker=6,
            ax=self.flux_ax
        )
        selected_trans = self.selection["transition"]
        self.trans_marker_flux = TransitionMarkerChart(
            selected_trans,
            locator_func = self.transition_locator,
            y_min = 0.0,
            y_max = 1.5,
            ax=self.flux_ax,
        )
        self.trans_marker_resid = TransitionMarkerChart(
            selected_trans,
            locator_func=self.transition_locator,
            y_min=-10.0,
            y_max=10.0,
            ax=self.resid_ax,
        )
    
    def transition_locator(self, transition):
        return self.coordinate_map(transition.wv)
    
    def attach_as_dock(self, dock_name, widget, dock_area):
        dock = QtWidgets.QDockWidget(dock_name, self)
        dock.setAllowedAreas(Qt.AllDockWidgetAreas)
        dock.setWidget(widget)
        self.addDockWidget(dock_area, dock)
    
    def on_prev(self):
        prev_idx = max(self.exemplar_index-1, 0)
        print("prev idx", prev_idx)
        if prev_idx != self.exemplar_index:
            self.set_exemplar_index(prev_idx)
    
    def on_next(self):
        next_idx = min(self.exemplar_index+1, len(self.exemplar_indexer)-1)
        print("next idx", next_idx)
        if next_idx != self.exemplar_index:
            self.set_exemplar_index(next_idx)
