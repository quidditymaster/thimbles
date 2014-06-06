
import thimbles as tmb
import numpy as np
from copy import copy
import scipy
import matplotlib.pyplot as plt
import h5py
import argparse

#parser = argparse.ArgumentParser("processes collections of spectra carrying out fits to be examined inside the thimbles gui.")
#
#parser.add_argument("batch_file")

class BossData(object):
    
    def __init__(self):
        self.hf = h5py.File("/home/tim/data/boss_standards/regularized_standards.h5", "r")
    
    def get_spec(self, spec_idx):
        wvs = np.array(np.log10(self.hf["wvs"]))
        res = 0.0001*np.array(self.hf["resolution"])
        wvsol = tmb.spectrum.WavelengthSolution(wvs, lsf=res)
        flux = np.array(self.hf["flux"][spec_idx])
        inv_var = np.array(self.hf["inv_var"][spec_idx])
        return tmb.Spectrum(wvsol, flux, inv_var)


class TransitionGrouper(object):
    
    def __init__(self, transitions, wvs):
        self.transitions = transitions
    
    def groups(self):
        min_wv = self.wvs[0]
        max_wv = self.wvs[-1]
        return [[tr] for tr in self.transitions if min_wv < tr.wv < max_wv]

class NeighborGrouper(TransitionGrouper):

    def __init__(self, transitions, wvs):
        self.transitions = transitions
    
    def groups(self):
        pass

class TransitionBox(object):
    
    def __init__(self, region, transitions=None):
        self.region = region
        if transitions == None:
            transitions = []
        self.transitions = transitions
    
    def __len__(self):
        return len(self.transitions)
    
    def add_transition(self, transition):
        self.transitions.append(transition)
    
    def remove_transition(self, transition):
        self.transitions.remove(transition)
    
    def grouped_by_ionization(self):
        tgroups = {}
        for trans in self.transitions:
            group_result = tgroups.get(trans.ion)
            if group_result is None:
                group_result = [trans]
            tgroups[trans.ion] = trans
        return tgroups

class GroupingMetric(object):
    
    def __init__(self, metric_functions, group_cost):
        self.metric_functions = metric_functions
        self.group_cost = group_cost
    
    def cost(self, box, transition):
        cost = 0
        trans_list = copy(box.transitions)
        if not transition in trans_list:
            trans_list.append(transition)
        for m_func in self.metric_functions:
            cost += m_func(trans_list)
        return cost

def wv_std(transitions):
    trans_wv = [trans.wv for trans in transitions]
    return np.std(trans_wv)

def ep_std(transitions):
    trans_eps = [trans.ep for trans in transitions]
    return np.std(trans_eps)

def ion_std(transitions):
    trans_ion = [trans.ion for trans in transitions]
    return np.std(trans_ion)

def species_similarity(transitions):
    tgroups = {}
    for trans in transitions:
        group_result = tgroups.get(trans.species)
        if group_result is None:
            group_result = []
        group_result.append(trans)
        tgroups[trans.species] = group_result
    value_lens = [len(vl) for vl in tgroups.values()]
    max_value = max(value_lens)
    value_lens.remove(max_value)
    return max_value - sum(value_lens)

class StandardGroupingMetric(GroupingMetric):
    
    def __init__(self, epsig=0.25, ionsig=0.25, wvsig=1.0, species_weight=2.0, new_cost=2.5):
        mfuncs = []
        mfuncs.append(lambda t:(wv_std(t)/wvsig)**2)
        mfuncs.append(lambda t:(ep_std(t)/epsig)**2)
        mfuncs.append(lambda t:(ion_std(t)/ionsig)**2)
        mfuncs.append(lambda t: -species_weight*species_similarity(t))
        super(StandardGroupingMetric, self).__init__(mfuncs, new_cost)

class BoxTransitionGrouper(TransitionGrouper):
    
    def __init__(self, transitions, wvs, box_size, grouping_metric, neighbor_dx=2):
        self.transitions=transitions
        self.wvs=wvs
        self.box_size = box_size
        self.neighbor_dx = neighbor_dx
        self.neighbor_deltas = np.arange(-self.neighbor_dx, self.neighbor_dx+1)
        self.grouping_metric = grouping_metric
        
        #initialize regions
        self.min_wv = wvs[0] - self.box_size/2.0
        self.n_regions = int(np.ceil((wvs[-1]-wvs[0]+self.box_size)/self.box_size))
        
        #start by putting exactly one box in each region with no overlap
        transition_region_idxs = [int(np.around((trans.wv-self.min_wv)/self.box_size)) for trans in self.transitions]
        belonging_transitions = [[] for i in range(self.n_regions)]
        for trans_idx in range(len(self.transitions)):
            reg_idx = transition_region_idxs[trans_idx]
            ctrans = self.transitions[trans_idx]
            belonging_transitions[reg_idx].append(ctrans)
        boxes = [TransitionBox(i, belonging_transitions[i]) for i in range(self.n_regions)]
        self._regions_to_boxes = dict(zip(range(self.n_regions), [[bx] for bx in boxes]))
    
    def add_box(self, region_idx, transitions):
        box = TransitionBox(region_idx, transitions)
        reg_res = self._regions_to_boxes[region_idx]
        reg_res.append(box)
    
    def groups(self):
        return self._regions_to_boxes.values()
    
    def iterate(self):
        for transition in self.transitions:
            self.reassign(transition)
    
    def neighboring_boxes(self, wv):
        center_idx = self.region_idx_from_wv(wv)
        neighbor_regions = [center_idx+off for off in self.neighbor_deltas\
                             if (0 <= center_idx+off <= self.n_regions-1)]
        neighbor_boxes = []
        for region_idx in neighbor_regions:
            boxes = self._regions_to_boxes.get(region_idx)
            if not boxes is None:
                neighbor_boxes.extend(boxes)
        return neighbor_boxes
    
    def reassign(self, transition):
        twv = transition.wv
        neighbor_boxes = self.neighboring_boxes(twv)
        n_neighbors = len(neighbor_boxes)
        costs = np.zeros(n_neighbors)
        for box_idx in range(n_neighbors):
            cbox = neighbor_boxes[box_idx]
            if transition in cbox.transitions:
                containing_box = cbox
            costs[box_idx] = self.grouping_metric.cost(cbox, transition)
        new_group_cost = self.grouping_metric.group_cost
        min_cost_idx = np.argmin(costs)
        if costs[min_cost_idx] < new_group_cost:
            containing_box.remove_transition(transition)
            neighbor_boxes[min_cost_idx].add_transition(transition)
        else:
            trans_reg_idx = self.region_idx_from_wv(twv)
            self.add_box(trans_reg_idx, [transition])
            #delete the old box if it is empty now
            if len(containing_box) == 0:
                self.remove_box(containing_box)
    
    def remove_box(self, box):
        box_list = self._regions_to_boxes[box.region_idx]
        box_list.remove(box)
    
    def region_idx_from_wv(self, wv):
        return int(np.around((wv-self.min_wv)/self.box_size))
    
    def wv_from_region_idx(self, idx):
        return self.min_wv + (idx+0.5)*self.box_size


class TransitionData(object):
    
    def __init__(self, foreground_file, bkground_file):
        pass
        

if __name__ == "__main__":
    data = BossData()
    #load in the spectra
    
    gmet = StandardGroupingMetric()
    ldat = tmb.io.linelist_io.read_vald_linelist("/home/tim/linelists/vald/TimothyAnderton.000991")
    minwv = ldat[0].wv
    maxwv = ldat[-1].wv
    
    #import pdb; pdb.set_trace()
    btg = BoxTransitionGrouper(ldat, wvs=(minwv, maxwv), box_size=5.0, grouping_metric=gmet, neighbor_dx=2)
    btg.iterate()
    #associate the spectra to source objects
    #this can hopefully be done with 
    
    #associate the spectra to an instrument model (many spectra to one instrument model)
    #
    
    #associate the spectra to an atmosphere model (many spectra to one atmosphere model)
    #decent mechanism at least at first, just associate everything to the same atmosphere model
    #appropriate for the sdss boss spectra since our atmosphere model will be all ones
    
    #
    
