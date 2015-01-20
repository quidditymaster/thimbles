
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
            if isinstance(ion, tuple):
                speciesnum, charge = ion
            else:
                speciesnum = int(ion)
                charge = int((ion-speciesnum)*10)
            ion = Ion(speciesnum, charge)
        self.ion = ion
        
        if damp is None:
            damp = Damping()
        elif isinstance(damp, dict):
            damp = Damping(**damp)
        self.damp = damp
    
    @property
    def z(self):
        return self.ion.species.z
    
    @property
    def charge(self):
        return self.ion.charge
    
    def pseudo_strength(self, stellar_parameters=None):
        solar_ab = self.ion.species.solar_ab
        if stellar_parameters is None:
            theta = 1.0
            metalicity = 0.0
        else:
            theta = stellar_parameters.theta
            metalicity = stellar_parameters.metalicity
        return solar_ab + metalicity + self.loggf - theta*self.ep

transgroup_assoc = sa.Table("transgroup_assoc", Base.metadata,
    Column("transition_id", Integer, ForeignKey("Transition._id")),
    Column("transition_group_id", Integer, ForeignKey("TransitionGroup._id")),
)

class TransitionGroup(ThimblesTable, Base):
    transitions = relationship("Transition", secondary=transgroup_assoc)
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


@task(result_name="grouping_standard")
def make_grouping_standard(
        standard_name, 
        tdb, 
        transition_filters=None, 
        min_wv=None, 
        max_wv=None,
        wv_split=10.0,
        ep_split=1.0, 
        loggf_split=1.0, 
        x_split=1.0, 
        commit=False):
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
        ("wv", wv_split),
        ("ep", ep_split),
        ("loggf", loggf_split),
    ]
    attr_list = split_scale_dict.keys()
    split_vec = split_scale_dict.values()
    trans_df = tmb.utils.misc.attribute_df(transitions, [])
    grouping_vec = tmb.utils.misc.running_box_segmentation(
        trans_df, ["z", ],
    )
    
    

@task(result_name="grouping_standard")
def edit_grouping_standard(grouping_standard, tdb):
    pass
    
