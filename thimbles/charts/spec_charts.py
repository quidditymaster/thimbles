import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import thimbles as tmb
from thimbles.charts import MatplotlibCanvas
from thimbles.options import Option, opts

spectra_mpl_kwargs=Option("spectra_mpl_kwargs", option_style="parent_dict", parent="charts")
spec_color = "#7EA057"
Option("color", default=spec_color, parent=spectra_mpl_kwargs)
Option("linewidth", default=2.0, parent=spectra_mpl_kwargs)
#Option("linestyle", default="steps-mid", parent=spectra_mpl_kwargs)
Option("linestyle", default="-", parent=spectra_mpl_kwargs)

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
