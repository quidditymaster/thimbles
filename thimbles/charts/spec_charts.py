import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection

import thimbles as tmb
from thimbles.charts import MatplotlibCanvas
from thimbles.options import Option, opts

spectra_mpl_kwargs=Option("spectra_mpl_kwargs", option_style="parent_dict", parent="charts")
spec_color = "#7EA057"
Option("color", default=spec_color, parent=spectra_mpl_kwargs)
Option("linewidth", default=2.0, parent=spectra_mpl_kwargs)
#Option("linestyle", default="steps-mid", parent=spectra_mpl_kwargs)
Option("linestyle", default="-", parent=spectra_mpl_kwargs)


class MarkerAnnotation(object):
    _annotation_initialized = False
    
    def __init__(
            self,
            locator_func=None,
            annotation_func=None,
            annotation_offset=None,
            ax=None,
    ):
        if ax is None:
            fig, ax = plt.subplots()
        self.ax = ax
        self.transition=transition
        self.annotation_func = annotation_func
        if locator_func is None:
            locator_func = lambda x: x.wv
        self.locator_func = locator_func
        if line_kwargs is None:
            line_kwargs = {}
        self.line_kwargs = line_kwargs
        if not transition is None:
            self.update()
    
    def get_line_pts(self):
        x=self.locator_func(self.transition)
        return [[x, x], [self.marker_top, self.marker_bottom]]
    
    def update(self):
        if not self._annotation_initialized:
            if self.annotation_func is None:
                self.annotation = self.ax.annotate(
                    self.annotation_func(self.transition),
                    xy = (trans_wv, self.marker_top),
                )
        else:
            pass
        if not self._self._line_initialized:
            x, y = self.get_line_pts()
            self.line = self.ax.plot(x, y, **self.line_kwargs)


def make_effective_sparse_coordinatizer():
    pass

class ExemplarWidthEffectChart(object):
    
    def __init__(
            star,
            exemplar,
            exemplar_indexer,
            exemlar_map,
            transition_indexer,
            flux_ax = None,
            resid_ax= None,
    ):
        self.star = star
        self.exemplar = exemplar
        self.exemplar_indexer = exemplar_indexer
        self.transition_indexer = transition_indexer
        
        if flux_ax is None and resid_ax is None:
            fig, axes = plt.subplots(sharex=True)
            flux_ax = axes[0]
            resid_ax = axes[1]
        self.flux_ax = flux_ax
        self.resid_ax = resid_ax
        
        spectra = star.spectroscopy
        flux_params = [spec["obs_flux"] for spec in spectra]
        mod_flux = [fp.value for fp in flux_params]
        obs_flux = [spec.flux for spec in spectra]
        obs_var = [spec.var for spec in spectra]
        obs_wvs = [spec.wvs for spec in spectra]
        norms = [spec["norm"].value for spec in spectra]
        
        ew_param = star["thermalized_widths_vec"]
        derivative_matrices = tmb.modeling.derivatives.driv(flux_params, [ew_param], combine_matrices=False)
        
        strength_matrix = star["strength_matrix"].value
        
        
        
        self.flux_markers = tmb.charts.VerticalMarkerChart(ax=self.flux_ax)
        self.resid_markers = tmb.charts.VerticalMarkerChart(ax=self.resid_ax)


class SpectraChart(object):
    _plots_initialized = False
    
    def __init__(
            self,
            spectra,
            bounds,
            ax,
            line_kwargs = None,
    ):
        self.spectra = spectra
        self.bounds = bounds
        self.ax=ax
        if line_kwargs is None:
            line_kwargs = {}
        self.lines = LineCollection(segments=self.get_segments(), **line_kwargs)
        self.ax.add_collection(self.lines)
    
    def get_segments(self):
        segments = []
        for spec in self.spectra:
            bspec = spec.sample(self.bounds, mode="bounded")
            if bspec is None:
                continue
            seg = np.stack((bspec.wvs, bspec.flux), axis=1)
            segments.append(seg)
        return segments
    
    def update(self):
        print("update called")
        self.lines.set_segments(self.get_segments())
        self.ax.figure._tmb_redraw=True
    
    def set_bounds(self, bounds):
        self.bounds = bounds
        self.update()
    
    def set_spectra(self, spectra):
        self.spectra = spectra
        self.update()


class SpectrumChart(object):
    _plots_initialized = False
    
    def __init__(
        self, 
        spectrum=None, 
        bounds=None,
        normalize=True,
        auto_zoom=True,
        label_axes=True,
        ax=None,
        **kwargs
    ):
        if ax is None:
            fig, ax = plt.subplots()
        self.ax = ax
        self.auto_zoom = auto_zoom
        
        self.ax.get_xaxis().get_major_formatter().set_useOffset(False)
        if label_axes:
            self.ax.set_xlabel("Wavelength")
            self.ax.set_ylabel("Flux")
        
        line_kwargs = kwargs
        default_kwargs = opts["charts.spectra_mpl_kwargs"]
        for key in default_kwargs:
            line_kwargs.setdefault(key, default_kwargs[key])
        self.line_kwargs = line_kwargs
        self.bounds = bounds
        self.normalize=normalize
        
        self.set_spectrum(spectrum)
    
    def set_spectrum(self, spectrum):
        self.spectrum = spectrum
        if not spectrum is None:
            self._initialize_plots()
            self.update()
    
    def set_normalize(self, normalize):
        self.normalize = normalize
        self.update()
    
    def cropped_spectrum(self):
        if not self.bounds is None:
            bspec = self.spectrum.sample(self.bounds, mode="bounded")
        else:
            bspec = self.spectrum
        return bspec
    
    def _initialize_plots(self):
        if not self._plots_initialized:
            bspec = self.cropped_spectrum()
            if not bspec is None:
                if self.normalize:
                    bspec = bspec.normalized()
            if not bspec is None:
                self.spec_line ,= self.ax.plot(bspec.wvs, bspec.flux, **self.line_kwargs)
            elif bspec is None:
                self.spec_line ,= self.ax.plot([0], [1], **self.line_kwargs)
                self.spec_line.set_visible(False)
        self._plots_initialized = True
    
    def set_bounds(self, bounds):
        self.bounds = bounds
        self.update()
    
    def update(self):
        bspec = self.cropped_spectrum()
        if bspec is None:
            self.spec_line.set_visible(False)
            return
        else:
            self.spec_line.set_visible(True)
        if self.normalize:
            bspec = bspec.normalized()
        if self.auto_zoom:
            bwv = bspec.wvs
            min_x, max_x = sorted([bwv[0], bwv[-1]])
            min_y = min(0, np.min(bspec.flux))
            mflux = np.median(bspec.flux)
            max_y_idx = np.argmax((bspec.flux > mflux)*(bspec.flux*bspec.ivar))
            max_y = bspec.flux[max_y_idx]
            self.ax.set_xlim(min_x, max_x)
            self.ax.set_ylim(min_y-0.01, max_y+(max_y-min_y)*0.15)
        self.spec_line.set_data(bspec.wvs, bspec.flux)
        self.spec_line.figure._tmb_redraw=True


if __name__ == "__main__":
    spec = tmb.Spectrum([0, 1, 2], [3, 5, 4])
    schart = SpectrumChart(spec)
    plt.show()
