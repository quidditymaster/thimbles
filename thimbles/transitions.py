import numpy as np
import pandas as pd

import thimbles as tmb
from thimbles.tasks import task
from thimbles.sqlaimports import *
from thimbles.thimblesdb import Base, ThimblesTable
from thimbles.modeling import Parameter, Model
from thimbles.abundances import Ion
from sqlalchemy.orm.collections import collection
from sqlalchemy.orm.collections import attribute_mapped_collection
from sqlalchemy.ext.associationproxy import association_proxy

import latbin

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
                    raise ValueError("Ion specification {} not understood\n possible formats are\n (z, charge)\n (z, charge, isotope)".format(ion))
            else:
                speciesnum = int(ion)
                charge = int(round(10*(ion-speciesnum)))
                if speciesnum > 100:
                    iso_mult = 1e5
                else:
                    iso_mult = 1e3
                species_leftover = ion-speciesnum-0.1*charge
                isotope = int(round(iso_mult*species_leftover))
            ion = Ion(speciesnum, charge, isotope)
        self.ion = ion
        
        if damp is None:
            damp = Damping()
        elif isinstance(damp, dict):
            damp = Damping(**damp)
        elif isinstance(damp, Damping):
            pass
        else:
            raise ValueError("damping of type {} is not understood".format(type(damp)))
        self.damp = damp
    
    def __repr__(self):
        return "Transition :{wv:7.3f}, {ion}, {ep:4.2f} {loggf:4.2f}".format(
            wv = self.wv,
            ion = self.ion,
            ep=self.ep,
            loggf=self.loggf,
        )
    
    @property
    def z(self):
        return self.ion.z
    
    @property
    def charge(self):
        return self.ion.charge
    
    def pseudo_strength(self, teff=5000.0, metalicity=0.0, ion_delta=1.5):
        solar_ab = self.ion.solar_ab
        theta = 5040.0/teff
        ion_ratio = np.power(10.0, ion_delta)
        ion_frac = ion_ratio/(ion_ratio + 1.0)
        if self.ion.charge == 1:
            ion_correction = np.log10(ion_frac)
        else:
            ion_correction = np.log10(1.0-ion_frac)
        return solar_ab + metalicity + self.loggf - theta*self.ep + ion_correction + np.log10(self.wv) - 3.0
    
    @property
    def x(self):
        return self.pseudo_strength()


class TransitionIndex(Base, ThimblesTable):
    _transition_id = Column(Integer, ForeignKey("Transition._id"))
    transition = relationship("Transition", foreign_keys=_transition_id)
    index = Column(Integer)
    _indexer_p_id = Column(Integer, ForeignKey("TransitionIndexerParameter._id"))
    
    def __init__(self, transition, index):
        self.transition = transition
        self.index = index


class TransitionIndexer(object):
    
    def __init__(self):
        self.transitions = []
        self.trans_to_idx = {}
        self._trans_idx_list = []
    
    def extend_transitions(self, transitions):
        for t in transitions:
            t_i = len(self.transitions)
            t_index = TransitionIndex(t, t_i)
            self._add_transition_index(t_index)
    
    @collection.appender
    def _add_transition_index(self, t_index):
        self._trans_idx_list.append(t_index)
        t = t_index.transition
        i = t_index.index
        n_current = len(self.transitions)
        if n_current < i+1:
            self.transitions.extend([None for j in range(i+1-n_current)])
        self.transitions[i] = t
        self.trans_to_idx[t] = i
    
    @collection.remover
    def _remove_transition_index(self, t_index):
        t = t_index.transition
        i = t_index.index
        self._trans_idx_list.remove(t_index)
        self.transitions[i] = None
        self.trans_to_idx.pop(t)
    
    @collection.iterator
    def __iter__(self):
        for t_idx in self._trans_idx_list:
            yield t_idx
    
    def __getitem__(self, index):
        if isinstance(index, Transition):
            return self.trans_to_idx[index]
        else:
            return self.transitions[index]
    
    def __len__(self):
        return len(self.transitions)


class TransitionIndexerParameter(Parameter):
    _id = Column(Integer, ForeignKey("Parameter._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"TransitionIndexerParameter",
    }
    _value = relationship(
        "TransitionIndex", 
        collection_class=TransitionIndexer,
    )
    
    def __init__(self, transitions):
        self._value.extend_transitions(transitions)
    

class TransitionMappedFloat(Base, ThimblesTable):
    _transition_id = Column(Integer, ForeignKey("Transition._id"))
    transition = relationship("Transition", foreign_keys=_transition_id)
    _mapper_parameter_id = Column(Integer, ForeignKey("TransitionMappedParameter._id"))
    value = Column(Float)
    
    def __init__(self, transition, value):
        self.transition = transition
        self.value = value


class TransitionMappedParameter(Parameter):
    _id = Column(Integer, ForeignKey("Parameter._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"TransitionMappedParameter",
    }
    _transition_map = relationship(
        "TransitionMappedFloat",
        collection_class=attribute_mapped_collection("transition")
    )
    _value = association_proxy("_transition_map", "value")
    
    def __init__(self, mapped_values=None):
        if mapped_values is None:
            mapped_values = {}
        self._value = mapped_values


class TransitionMappedVectorizerModel(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"TransitionMappedVectorizerModel",
    }
    fill_value = Column(Float)
    
    def __init__(
            self, 
            output_p, 
            transition_mapped_p, 
            indexer_p, 
            fill_value=0.0
    ):
        self.output_p = output_p
        self.add_parameter("tdict", transition_mapped_p)
        self.add_parameter("indexer", indexer_p)
        self.fill_value = fill_value
    
    def __call__(self, vprep=None):
        vdict = self.get_vdict(vprep)
        tdict = vdict[self.inputs["tdict"]]
        indexer = vdict[self.inputs["indexer"]]
        ntrans = len(indexer)
        vec_vals = np.repeat(self.fill_value, ntrans)
        for t in tdict:
            t_idx = indexer.trans_to_idx.get(t)
            if not t_idx is None:
                vec_vals[t_idx] = tdict[t]
        return vec_vals


class ExemplarGrouping(Base, ThimblesTable):
    _transition_id = Column(Integer, ForeignKey("Transition._id"))
    transition = relationship("Transition", foreign_keys=_transition_id)
    _exemplar_id = Column(Integer, ForeignKey("Transition._id"))
    exemplar = relationship("Transition", foreign_keys=_exemplar_id)
    _grouping_id = Column(Integer, ForeignKey("ExemplarGroupingParameter._id"))
    
    def __init__(self, exemplar, transition):
        self.exemplar = exemplar
        self.transition = transition


class ExemplarMap(object):
    
    def __init__(self):
        self.groups = {}
        self._groupings = []
        
    def get(self, index):
        return self.groups.get(index)
    
    def exemplars(self):
        return self.groups.keys()
    
    def extend_groups(self, groups):
        for exemp in groups:
            for trans in groups[exemp]:
                cgroup = ExemplarGrouping(exemp, trans)
                self._add_grouping(cgroup)
    
    @collection.appender
    def _add_grouping(self, grouping):
        self._groupings.append(grouping)
        exemp = grouping.exemplar
        trans = grouping.transition
        group_l = self.groups.get(exemp)
        if group_l is None:
            group_l = []
        group_l.append(trans)
        self.groups[exemp] = group_l
    
    @collection.remover
    def _remove_grouping(self, grouping):
        self._groupings.remove(grouping)
        exemp = grouping.exemplar
        trans = grouping.transition
        self.groups[exemp].remove(trans)
    
    @collection.iterator
    def __iter__(self):
        for g in self._groupings:
            yield g

class ExemplarGroupingParameter(Parameter):
    _id = Column(Integer, ForeignKey("Parameter._id"), primary_key=True)
    __mapper_args__={
        "polymorphic_identity":"ExemplarGroupingParameter",
    }
    
    _value = relationship(
        "ExemplarGrouping",
        collection_class=ExemplarMap,
    )
    
    def __init__(self, groups):
        self._value.extend_groups(groups)

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
    
    @property
    def exemplar(self):
        ntrans = len(self.transitions)
        exemplar_idx = ntrans//2
        return self.transitions[exemplar_idx]
    
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
)
def segmented_grouping_standard(
        standard_name, 
        database, 
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
    existing_standards = database.query(TransitionGroupingStandard)\
     .filter(TransitionGroupingStandard.name == standard_name).all()
    if len(existing_standards) > 0:
        raise ValueError("TransitionGroupingStandard with name {} already exists".format(standard_name))
    if transition_filters is None:
        transition_filters = []
    if not min_wv is None:
        transition_filters.insert(0, Transition.wv > min_wv)
    if not max_wv is None:
        transition_filters.insert(0, Transition.wv < max_wv)
    t_query = database.query(Transition)
    for t_filter in transition_filters:
        t_query = t_query.filter(t_filter)
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
        database.add(tstand)
        database.commit()
    return tstand


def lines_by_species(linelist, match_isotopes=False):
    """turn a list of Transitions into a species keyed dictionary 
    of transitions.
    """
    if not match_isotopes:
        extract_species = lambda l: (l.ion.z, l.ion.charge)
    else:
        extract_species = lambda l: (l.ion.z, l.ion.charge, l.ion.isotope)
    
    species_dict = {}
    for line in linelist:
        species = extract_species(line)
        trans = species_dict.get(species, [])
        trans.append(line)
        species_dict[species] = trans
    return species_dict


def prefer_existing(trans, db, matches):
    return matches[0]

def accept_new(trans, db):
    return trans

@task()
def update_linelist(
        linelist,
        database=None,
        dlogwv=5e-6,
        dep=0.05,
        dloggf=1.0,
        match_isotopes=False,
        on_match=prefer_existing,
        on_matchless=accept_new,
        read_func=None, 
        read_kwargs=None,
        auto_commit=True,
):
    assert dwv > 0
    assert dep > 0
    assert dloggf > 0
    if read_func is None:
        read_func = tmb.io.linelist_io.read_linelist
    if read_kwargs is None:
        read_kwargs = {}
    if isinstance(linelist, str):
        linelist = read_func(linelist, **read_kwargs)
    
    line_wvs = [l.wv for l in linelist]
    min_wv = np.min(line_wvs)
    max_wv = np.max(line_wvs)
    
    #species keyed dictionary of line lists
    lbs = lines_by_species(linelist, match_isotopes=match_isotopes)
    species_keys = sorted(lbs.keys())
    for species in species_keys:
        if len(species) == 3:
            z, charge, isotope = species
        elif len(species) == 2:
            z, charge = species
        else:
            raise ValueError("unexpected species specification {}".format(species))
        ion_query = database.query(Ion)\
                .filter(Ion.z == z)\
                .filter(Ion.charge == charge)
        if match_isotopes:
            ion_query = ion_query.filter(Ion.isotope == isotope)
        else:
            isotope = 0
        cur_ion = ion_query.first()
        if cur_ion is None:
            cur_ion = Ion(z, charge, isotope)
        
        l_query = database.query(Transition)\
            .join(Ion)\
            .filter(Ion.z == z)\
            .filter(Ion.charge == charge)\
            .filter(Transition.wv < max_wv + dwv)\
            .filter(Transition.wv > min_wv - dwv)
        if match_isotopes:
            l_query = l_query.filter(Ion.isotope == isotope)
        
        db_lines = l_query.all()
        new_lines = lbs[species]
        
        #convert lists into data frames
        df_attrs = "wv ep loggf".split()
        db_df = tmb.utils.misc.attribute_df(db_lines, attrs=df_attrs)
        new_df = tmb.utils.misc.attribute_df(new_lines, attrs=df_attrs)
        #rescale by tolerances
        match_scale = pd.Series({"wv":dwv, "ep":dep, "loggf":dloggf})
        db_df /= match_scale
        new_df /= match_scale
        #import pdb; pdb.set_trace()
        matched_new, matched_db, dist = latbin.matching.match(new_df, db_df, tolerance=1.0)
        
        match_idx = 0
        for new_idx in range(len(new_lines)):
            cdb_match = []
            while (match_idx < len(matched_new)) and matched_new[match_idx] == new_idx:
                dbl_idx = matched_db[match_idx]
                cdb_match.append(db_lines[dbl_idx])
                match_idx += 1
            if len(cdb_match) > 0:
                ctrans = on_match(new_lines[new_idx], database, cdb_match)
            else:
                ctrans = on_matchless(new_lines[new_idx], database)
            ctrans.ion = cur_ion
            database.add(ctrans)
        if auto_commit:
            database.commit()

