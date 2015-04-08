import numpy as np

import thimbles as tmb
from thimbles.tasks import task
from thimbles.sqlaimports import *
from thimbles.thimblesdb import Base, ThimblesTable
from thimbles.modeling import Parameter, Model
from thimbles.abundances import Ion


class Damping(ThimblesTable, Base):
    stark = Column(Float)
    waals = Column(Float)
    rad   = Column(Float)
    empirical  = Column(Float)
    
    def __init__(self, stark=None, waals=None, rad=None, empirical=None):
        self.stark = stark
        self.waals = waals
        self.rad = rad
        self.empirical = empirical

class Transition(ThimblesTable, Base):
    wv = Column(Float)
    _ion_id = Column(Integer, ForeignKey("Ion._id"))
    ion = relationship("Ion")
    ep = Column(Float)
    loggf = Column(Float)
    _damp_id = Column(Integer, ForeignKey("Damping._id"))
    damp = relationship(Damping)
    
    def __init__(self, wv, ion, ep, loggf, damp=None):
        self.wv = wv
        self.ep = ep
        self.loggf = loggf
        
        if not isinstance(ion, Ion):
            if isinstance(ion, (tuple, list)):
                if len(ion) == 2:
                    speciesnum, charge = ion
                    isotope = 0
                elif len(ion) == 3:
                    speciesnum, charge, isotope = ion
                else:
                    raise ValueError("Ion specification {} not understood".format(ion))
            else:
                speciesnum = int(ion)
                charge = int(10*(ion%1))
                if speciesnum > 100:
                    iso_mult = 1e5
                else:
                    iso_mult = 1e3
                isotope = int(iso_mult*(ion%0.1))
            ion = Ion(speciesnum, charge, isotope)
        self.ion = ion
        
        if damp is None:
            damp = Damping()
        elif isinstance(damp, dict):
            damp = Damping(**damp)
        self.damp = damp
    
    @property
    def z(self):
        return self.ion.z
    
    @property
    def charge(self):
        return self.ion.charge
    
    def pseudo_strength(self, stellar_parameters=None, ion_delta=1.5):
        solar_ab = self.ion.solar_ab
        if stellar_parameters is None:
            theta = 1.0
            metalicity = 0.0
        else:
            theta = stellar_parameters.theta
            metalicity = stellar_parameters.metalicity
        ion_ratio = np.power(10.0, ion_delta)
        ion_frac = ion_ratio/(ion_ratio + 1.0)
        #ion_correction = np.log10(ion_frac)
        #neutral_correction = np.log10(1.0-ion_frac)
        if self.ion.charge == 1:
            ion_correction = np.log10(ion_frac)
        else:
            ion_correction = np.log10(1.0-ion_frac)
        return solar_ab + metalicity + self.loggf - theta*self.ep + ion_correction + np.log10(self.wv) - 3.0
    
    @property
    def x(self):
        return self.pseudo_strength()


transgroup_assoc = sa.Table("transgroup_assoc", Base.metadata,
    Column("transition_id", Integer, ForeignKey("Transition._id")),
    Column("transition_group_id", Integer, ForeignKey("TransitionGroup._id")),
)

class TransitionGroup(ThimblesTable, Base):
    transitions = relationship("Transition", secondary=transgroup_assoc, order_by="Transition.loggf")
    _grouping_standard_id = Column(Integer, ForeignKey("TransitionGroupingStandard._id"))
    
    def __init__(self, transitions=None):
        if transitions is None:
            transitions = []
        self.transitions = transitions
    
    def __len__(self):
        return len(self.transitions)
    
    def __getitem__(self, index):
        return self.transitions[index]
    
    def __setitem__(self, index, value):
        self.transitions[index] = value
    
    def pop(self, index):
        self.transitions.pop(index)
    
    def append(self, value):
        self.transitions.append(value)
    
    def extend(self, in_list):
        self.transitions.extend(in_list)
    
    def aggregate(self, attr, reduce_func=np.mean, empty_val=np.nan):
        if len(self) == 0:
            return empty_val
        attrvals = [getattr(t, attr) for t in self.transitions]
        return reduce_func(attrvals)


class TransitionGroupingStandard(ThimblesTable, Base):
    name = Column(String)
    groups = relationship("TransitionGroup")
    
    def __init__(self, groups, name=None):
        self.name = name
        tgroups = []
        for group in groups:
            if not isinstance(group, TransitionGroup):
                group = TransitionGroup(group)
            tgroups.append(group)
        self.groups = tgroups
    
    def __len__(self):
        return len(self.groups)
    
    def __getitem__(self, index):
        return self.groups[index]

    def __setitem__(self, index, value):
        self.groups[index] = value


def as_transition_group(tgroup):
    if isinstance(tgroup, TransitionGroup):
        return tgroup
    else:
        return TransitionGroup(tgroup)


@task(
    result_name="grouping_standard",
    sub_kwargs=dict(
        standard_name=dict(option_style="raw_string")
    )
)
def segmented_grouping_standard(
        standard_name, 
        tdb, 
        transition_filters=None, 
        min_wv=None, 
        max_wv=None,
        wv_split=100.0, #TODO: split on log wavelength instead
        ep_split=1.0, 
        loggf_split=3.0, 
        x_split=1.0,
        split_charge=True,
        auto_commit=False,
    ):
    existing_standards = tdb.query(TransitionGroupingStandard)\
     .filter(TransitionGroupingStandard.name == standard_name).all()
    if len(existing_standards) > 0:
        raise ValueError("TransitionGroupingStandard with name {} already exists".format(standard_name))
    if transition_filters is None:
        transition_filters = []
    if not min_wv is None:
        transition_filters.insert(0, Transition.wv > min_wv)
    if not max_wv is None:
        transition_filters.insert(0, Transition.wv < max_wv)
    t_query = tdb.query(Transition)
    for t_filter in transition_filters:
        t_query.filter(t_filter)
    transitions = t_query.all()
    split_scales=[
        ("z", 0.1),
        ("wv", wv_split),
        ("ep", ep_split),
        ("loggf", loggf_split),
        ("x", x_split),
    ]
    if split_charge:
        split_scales.append(("charge", 0.1))
    attr_list, split_vec= list(zip(*split_scales))
    trans_df = tmb.utils.misc.attribute_df(transitions, attr_list)
    grouping_vec = tmb.utils.misc.running_box_segmentation(
        df=trans_df, 
        grouping_vecs=attr_list,
        split_threshold=split_vec,
        combine_breaks=True,
    )
    grouped_transitions = []
    for t_idx, t in enumerate(transitions):
        list_idx = grouping_vec[t_idx]
        if list_idx >= len(grouped_transitions):
            grouped_transitions.append([])
        grouped_transitions[list_idx].append(t)
    tstand = TransitionGroupingStandard(grouped_transitions, name=standard_name)
    if auto_commit:
        tdb.add(tstand)
        tdb.commit()
    return tstand

