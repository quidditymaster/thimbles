import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
import numpy as np
from thimbles.charts import MatplotlibCanvas


class ForkDiagram(object):
    _plots_initialized = False
    
    def __init__(
        self, 
        xvals=None, 
        depths=None, 
        curve=None, 
        handle_indexes=None, 
        handle_locator=None,
        nub_height=0.01, 
        handle_height=0.05, 
        spread_height=0.05, 
        text="", 
        ax=None,
    ):
        if ax is None:
            fig, ax = plt.subplots()
        self.ax = ax
        self.fig = ax.figure
        if handle_locator is None:
            handle_locator = np.mean
        if curve is None:
            curve = lambda x: np.ones(x.shape)
        self.depths=depths
        self.curve = curve
        self.nub_height=nub_height
        self.handle_height=handle_height
        self.spread_height=spread_height
        self.handle_indexes=handle_indexes
        self.text=text
        self.set_xvals(xvals)
    
    def _initialize_plots(self):
        if not self._plots_initialized:
            self._init_handle()
            self._init_tines()
            self._init_annotation()
            self.ax.add_line(self.handle)
            self.ax.add_collection(self.tines)
        self._plots_initialized = True
    
    def _init_handles(self):
        
        xvec = self.handle_x
        bot, top = self.handle_bottom, self.handle_top
        self.handle = mpl.lines.Line2D([x, x], [bot, top])
    
    def update_handle(self):
        x = self.handle_x
        bot, top = self.handle_bottom, self.handle_top
        self.handle.set_data([x, x], [bot, top])
    
    @property
    def handle_x(self):
        return self.handle_locator(self.xvals)
    
    @property
    def rel_handle_top(self):
        return self.rel_handle_bottom + self.handle_height
    
    @property
    def rel_handle_bottom(self):
        return 1.0 + self.spread_height + self.nub_height
    
    @property
    def handle_bottom(self):
        return self.curve(self.handle_x)*self.rel_handle_bottom
    
    @property
    def handle_top(self):
        return self.curve(self.handle_x)*self.rel_handle_top
    
    @property
    def nub_tops(self):
        return self.curve(self.xvals)*(1.0+self.nub_height)
    
    @property
    def tine_bottoms(self):
        return self.curve(self.xvals)*self.depths
    
    def _calc_tine_data(self):
        lvals = np.zeros((len(self.xvals), 3, 2))
        lvals[:, :2, 0] = self.xvals.reshape((-1, 1)) * np.ones((1, 2))
        lvals[:, 2, 0] = self.handle_x
        lvals[:, 0, 1] = self.tine_bottoms
        lvals[:, 1, 1] = self.nub_tops
        lvals[:, 2, 1] = self.handle_bottom
        return lvals
    
    def _init_tines(self):
        tine_data = self._calc_tine_data()
        self.tines = mpl.collections.LineCollection(tine_data)
    
    def update_tines(self):
        tine_data = self._calc_tine_data()
        self.tines.set_segments(tine_data)
    
    def _init_annotation(self):
        anot_xy = (self.handle_x, self.handle_top)
        self.annotation = mpl.text.Annotation(self.text, anot_xy)
        self.ax.add_artist(self.annotation)
    
    def update_annotation(self):
        self.annotation.set_text(self.text)
        self.annotation.set_x(self.handle_x)
        self.annotation.set_y(self.handle_top)
    
    def update(self):
        self.update_tines()
        self.update_handle()
        self.update_annotation()
    
    def set_xvals(self, xvals, update=True):
        if not xvals is None:
            self.xvals = np.asarray(xvals)
            if (self.depths is None) or self.depths.shape != self.xvals.shape:
                self.depths = np.zeros(xvals.shape)
            if (self.handle_indexes is None) or self.handle_indexes.shape != self.xvals.shape:
                self.handle_indexes = np.ones(xvals.shape)
            self._initialize_plots()
            if update:
                self.update()
    
    def set_curve(self, curve, update=True):
        self.curve = curve
        if update:
            self.update()
    
    def set_depths(self, depths, update=True):
        self.depths = depths
        if update:
            self.update()
    
    def set_handle_indexes(self, handle_indexes):
        pass


class TransitionsChart(object):
    _handles_initialized=False
    _fans_initialized=False
    _tines_initialized=False
    
    def __init__(
            self, 
            transitions,
            lmax=None,
            lmin=None,
            l_nub=0.02,
            grouping_dict=None,
            tine_min=0.0,
            tine_max=1.0,
            handle_max=1.35,
            fan_fraction=0.75,
            handle_picker=None,
            tine_picker=None,
            tine_tags=None,
            ax=None,
            **mpl_kwargs
    ):
        #import pdb; pdb.set_trace()
        self.tine_min=tine_min
        self.tine_max=tine_max
        self.handle_max=handle_max
        self.fan_fraction = fan_fraction
        self.lmax=lmax
        self.lmin=lmin
        self.l_nub = l_nub
        if grouping_dict is None:
            grouping_dict = {}
        self.grouping_dict=grouping_dict
        self.handle_picker=handle_picker
        self.tine_picker=tine_picker
        if tine_tags is None:
            tine_tags = {}
        self.tine_tags=tine_tags
        self.mpl_kwargs = mpl_kwargs
        if ax is None:
            fig, ax = plt.subplots()
        self.ax = ax
        self.set_transitions(transitions)
    
    def get_handle_pts(self):
        n_handles = len(self.handle_wvs)
        if n_handles == 0:
            return None
        dat = np.zeros((n_handles, 2, 2))
        dat[:, 0, 0] = self.handle_wvs
        dat[:, 1, 0] = self.handle_wvs
        dat[:, 0, 1] = self.tine_max + (self.handle_max-self.tine_max)*self.fan_fraction
        dat[:, 1, 1] = self.handle_max
        return dat
    
    def get_fan_pts(self):
        if len(self.handle_wvs)==0:
            return None
        fan_idxs = np.where(self.grouping_vec > -1)[0] 
        dat = np.zeros((len(fan_idxs), 2, 2))
        dat[:, 0, 0] = self.fan_bottom_wvs
        dat[:, 1, 0] = self.fan_top_wvs
        dat[:, 0, 1] = self.tine_max
        dat[:, 1, 1] = self.tine_max + (self.handle_max-self.tine_max)*self.fan_fraction
        return dat
    
    def get_tine_pts(self):
        dat = np.zeros((len(self.transitions), 2, 2))
        dat[:, 0, 0] = self.transition_wvs
        dat[:, 1, 0] = self.transition_wvs
        dat[:, 0, 1] = self.tine_min*self.tine_lengths+self.tine_max*(1.0-self.tine_lengths)
        dat[:, 1, 1] = self.tine_max
        return dat
    
    def _initialize_plots(self):
        if not self._handles_initialized:
            handle_dat = self.get_handle_pts()
            if not handle_dat is None:
                self.handles = mpl.collections.LineCollection(handle_dat, picker=self.handle_picker, **self.mpl_kwargs)
                self.ax.add_collection(self.handles)
                self._handles_initialized = True
        if not self._fans_initialized:
            fan_dat = self.get_fan_pts()
            if not fan_dat is None:
                self.fans = mpl.collections.LineCollection(fan_dat, **self.mpl_kwargs)
                self.ax.add_collection(self.fans)
                self._fans_initialized = True
        if not self._tines_initialized:
            tine_dat = self.get_tine_pts()
            if not tine_dat is None:
                self.tines = mpl.collections.LineCollection(tine_dat, picker=self.tine_picker, **self.mpl_kwargs)
                self.ax.add_collection(self.tines)
                self._tines_initialized = True
    
    def set_transitions(self, transitions):
        if not transitions is None:
            twvs = np.array([t.wv for t in transitions])
            tlens = np.array([t.x for t in transitions])
            if self.lmin is None:
                self.lmin = np.min(tlens)
            if self.lmax is None:
                self.lmax = np.max(tlens)
            tlens = (tlens-self.lmin)/(self.lmax-self.lmin)
            tlens = np.clip(tlens, self.l_nub, 1)
            
            grouping_vec = np.repeat(-1, len(transitions))
            group_to_idx = {}
            handle_wvs = []
            fan_top_wvs = []
            fan_bottom_wvs = []
            group_list = []
            for trans_idx in range(len(transitions)):
                trans = transitions[trans_idx]
                group = self.grouping_dict.get(trans)
                if not group is None:
                    group_idx = group_to_idx.get(group)
                    if group_idx is None:
                        group_idx = len(group_to_idx)
                        group_to_idx[group]=group_idx
                        group_list.append(group)
                        handle_wvs.append(group.aggregate(attr="wv", reduce_func=np.mean))
                    fan_bottom_wvs.append(trans.wv)
                    fan_top_wvs.append(handle_wvs[-1])
                    grouping_vec[trans_idx]=group_idx
            self.transitions = transitions
            self.transition_wvs = twvs
            self.tine_lengths = tlens
            self.group_to_idx = group_to_idx
            self.group_list = group_list
            self.handle_wvs = handle_wvs
            self.fan_top_wvs = fan_top_wvs
            self.fan_bottom_wvs = fan_bottom_wvs
            self.grouping_vec = grouping_vec
            self._initialize_plots()
            self.update()
    
    def update(self):
        if self._handles_initialized:
            self.handles.set_segments(self.get_handle_pts())
            metdat = dict(
                kind="groups",
                groups=self.group_list,
            )
            self.handles._md = metdat
        if self._fans_initialized:
            self.fans.set_segments(self.get_fan_pts())
        if self._tines_initialized:
            self.tines.set_segments(self.get_tine_pts())
            metdat = dict(
                kind="transitions",
                transitions=self.transitions,
            )
            metdat.update(self.tine_tags)
            self.tines._md = metdat
        self.ax._tmb_redraw=True
    
    def set_bounds(self, bounds):
        pass
