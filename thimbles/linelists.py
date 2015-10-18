import numpy as np
from copy import copy
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import pandas as pd


def transition_list_to_group_vec(tlist, indexer):
    group_vecs = []
    row_idxs = np.array([indexer[t] for t in tgroup])
    col_idxs = np.zeros(len(row_idxs))
    dat = np.ones(len(row_idxs))
    ij_mat = np.array([row_idxs, col_idxs])
    vshape = (len(indexer.transitions), 1)
    gvec = sicpy.sparse.csc_matrix((dat, ij_mat), shape=vshape)
    return gvec


class TransitionGrouping(object):
    _fitness = None
    
    def __init__(
            self, 
            grouping_mat,
            fitness_function,
            tansition_mask,
    ):
        self.grouping_mat = grouping_mat
        #self.transition_mask = np.asarray(grouping_mat.sum(axis=1)).reshape((-1,)) > 0
        self.fitness_function = fitness_function
        self.transition_mask = transition_mask
    
    def make_gvec_column(self, row_indices):
        n_ind = len(row_indices)
        dat = np.repeat(1.0/np.sqrt(n_ind), n_ind)
        ij_mat = (row_indices, np.zeros(n_ind))
        vec_shape = (len(self.transition_mask), 1)
        return scipy.sparse.csc_matrix((dat, ij_mat), shape=vec_shape)
    
    def repair_gmat(self, gmat):
        gmat = gmat.tocsc()
        col_sum = np.asarray(gmat.sum(axis=1)).reshape((-1,))
        
        where_multiple = np.where(col_sum > 0)
        import pdb; pdb.set_trace()
        index_counts = dict(zip(where_multiple, col_sum(where_multiple)))
        missing_indexes = list(set(np.where((col_sum == 0)*self.transition_mask)[0]))
        inject_col_count = {}
        for inj_idx in np.random.randint(len(realized_vecs), size=(len(missing_indexes),)):
            cur_count = inject_col_count.get(inj_idx, 0)
            inject_col_count[inj_idx] = cur_count + 1
        
        cleaned_vecs = []
        cur_miss_idx = 0
        for rvec_idx in range(gmat.shape[1]):
            indices = []
            rvec = gmat[:, rvec_idx]
            for idx in rvec.indices:
                if idx in doubled_indexes:
                    doubled_indexes.remove(idx)
                else:
                    indices.append(idx)
            while rvec_idx in inject_col_count:
                indices.append(missing_indexes[cur_miss_idx])
                cur_miss_idx += 1
                inject_col_count[rvec_idx] = inject_col_count[rvec_idx] - 1
                if inject_col_count[rvec_idx] <= 0:
                    inject_col_count.pop(rvec_idx)
            if len(indices) > 0:
                new_vec = self.make_gvec_column(indices)
                cleaned_vecs.append(new_vec)
    
    def mate(self, other):
        prob_vecs = []
        n_match = min(self.grouping_mat.shape[1], other.grouping_mat.shape[1])
        longer = self.grouping_mat
        longer_fit = self.fit_vec
        shorter = other.grouping_mat
        shorter_fit = other.fit_vec
        if self.grouping_mat.shape[1] < other.grouping_mat.shape[1]:
            longer = other.grouping_mat
            longer_fit = other.fit_vec
            shorter = self.grouping_mat
            shorter_fit = self.fit_vec
        gprod = shorter.transpose()*longer
        gprod = gprod.tocsr()
        taken_col_idxs = set()
        for i in range(n_match):
            matching_preference = np.argsort(-gprod[i].data)
            for j in matching_preference:
                if not j in taken_col_idxs:
                    taken_col_idxs.add(j)
                    i_fit = shorter_fit[i]
                    j_fit = longer_fit[j]
                    i_weight = i_fit/(j_fit + i_fit)
                    i_weight = min(max(0.25, i_weight), 0.75)
                    j_weight = 1.0-i_weight
                    new_pvec = i_weight*shorter[:, i] + j_weight*longer[:, j]
                    prob_vecs.append(new_pvec)
                    break
        for i in set(np.arange(longer.shape[1]))-taken_col_idxs:
            prob_vecs.append(longer[:, i])
        
        #realize the prob vecs
        realized_vecs = []
        for pvec in prob_vecs:
            data_mask = pvec.data > np.random.random(len(pvec.data))
            n_accept = np.sum(data_mask)
            realization = scipy.sparse.csc_matrix((np.ones(n_accept), (pvec.indices[data_mask], np.zeros((n_accept,)))), shape=pvec.shape)
            realized_vecs.append(realization)
        
        #test for double grouped and missing transitions
        full_mat = scipy.sparse.hstack(realized_vecs)
        
        new_mat = scipy.sparse.hstack(cleaned_vecs)
        return TransitionGrouping(new_mat, fitness_function=self.fitness_function)
    
    def mutate(self):
        n_vecs = self.grouping_mat.shape[1]
        mutate_rand = np.random.random()
        if mutate_rand < 0.5:
            #remove one grouping vector and add it randomly to another one
            mutated_vecs = [self.grouping_mat[:, i] for i in range(n_vecs)]
            if n_vecs > 1:
                
                remove_idx = np.random.randint(n_vecs)
                add_idx = np.random.randint(n_vecs-1)
                if remove_idx != add_idx:
                    removed_vec = mutated_vecs.pop(remove_idx)
                    new_vec = mutated_vecs[add_idx] + removed_vec
                    new_vec.data /= np.sum(new_vec.data)
                    mutated_vecs[add_idx] = new_vec
        elif mutate_rand <= 1.0:
            #donate some fraction of all the transitions to a new grouping.
            donated_indexes = []
            mutated_vecs = []
            n_mix = 2.0
            donate_prob = n_mix/n_vecs
            donate_frac = 1.0/n_mix
            gvecs = [self.grouping_mat[:, i] for i in range(n_vecs)]
            for gvec in gvecs:
                if np.random.random() <= donate_prob:
                    keep_mask = np.random.random(len(gvec.data)) > donate_frac
                    n_keep = np.sum(keep_mask)
                    if n_keep > 0:
                        vec = self.make_gvec_column(gvec.indices[keep_mask])
                        mutated_vecs.append(vec)
                    donated_indexes.extend(gvec.indices[np.logical_not(keep_mask)])
                else:
                    mutated_vecs.append(gvec)
            n_donate = len(donated_indexes)
            if n_donate > 0:
                donated_vec = self.make_gvec_column(donated_indexes)
                mutated_vecs.append(donated_vec)
        new_gmat = scipy.sparse.hstack(mutated_vecs)
        return TransitionGrouping(new_gmat, fitness_function=self.fitness_function)
    
    def _check_exec_fitness(self):
        if self._fitness is None:
            self._fitness, self._fitness_dict = self.fitness_function(self)
    
    @property
    def fitness(self):
        self._check_exec_fitness()
        return self._fitness
    
    @property
    def fit_vec(self):
        self._check_exec_fitness()
        return self._fitness_dict["vec"]


def make_fitness_function(
        wsim, 
        eps, 
        pseudostrengths, 
        central_ep=None, 
        max_eff_ep_offset=1.0, 
):
    if central_ep is None:
        central_ep = np.mean(eps)
    wsim_sum_diag = scipy.sparse.dia_matrix((np.asarray(wsim.sum(axis=0)).reshape((-1,)), 0), shape=wsim.shape)
    
    #import pdb; pdb.set_trace()
    def grouping_fitness(grouping):
        gmat = grouping.grouping_mat
        n_groups = grouping.grouping_mat.shape[1]
        grouping_vecs = [grouping.grouping_mat[:, i] for i in range(n_groups)]
        
        gwg_diag_mats = [gvec.transpose()*wsim*gvec for gvec in grouping_vecs]
        gwg_diags = []
        for gwg_d in gwg_diag_mats:
            if len(gwg_d.data) == 1:
                gwg_diags.append(gwg_d.data[0])
            elif len(gwg_d.data) == 0:
                gwg_diags.append(0)
            else:
                raise ValueError("expected 1x1 matrix")
        gwg_diags = np.array(gwg_diags)
        gwg_all = gmat.transpose()*wsim_sum_diag*gmat
        degen_off_diag = np.asarray(gwg_all.sum(axis=0)).reshape((-1,)) - gwg_diags
        
        ep_sigmas = []
        ep_offsets = []
        pst_sigmas = []
        n_per = []
        for i in range(n_groups):
            gvec = grouping_vecs[i]
            n_per.append(len(gvec.data))
            grouped_eps = [eps[i] for i in gvec.indices]
            cur_ep_std = np.std(grouped_eps)
            cur_ep_offset = np.abs(np.mean(grouped_eps) - central_ep)
            ep_offsets.append(cur_ep_offset)
            ep_sigmas.append(cur_ep_std)
            
            grouped_pst = [pseudostrengths[i] for i in gvec.indices]
            pst_sigmas.append(np.std(grouped_pst))
        
        ep_sigmas = np.array(ep_sigmas)
        ep_offsets = np.array(ep_offsets)
        pst_sigmas = np.array(pst_sigmas)
        n_per = np.array(n_per)
        
        fit_vec = 3.0*gwg_diags 
        fit_vec += -1.5*degen_off_diag
        fit_vec += -1*pst_sigmas
        fit_vec += -1*ep_sigmas
        fit_vec += 1.5*(n_per > 1) + 0.25*np.clip(n_per, 0, 30)
        per_group_cost = 2.5
        fit_vec -= per_group_cost
        
        fitness = np.sum(fit_vec)
        if fitness < 1:
            fitness = np.exp(fitness-1)
        fdict = dict(
            vec=np.clip(fit_vec, -5, 10)+5.0,
            n_per=n_per,
            ep_sigmas=ep_sigmas,
            ep_offsets=ep_offsets,
            pst_sigmas=pst_sigmas,
            gwg_d=gwg_diags,
            gwg_od=degen_off_diag,
        )
        return fitness, fdict
    return grouping_fitness

def evolve_transition_groups(
        spectra, 
        shared_parameter_space,
        population_size=400,
        n_generations=10,
        mating_fraction=0.4,
        mutate_frequency=0.8,
        diagnostic_plot_dir=None,
):
    assert population_size >= 2
    transition_indexer = shared_parameter_space["transition_indexer"].value
    transitions = transition_indexer.transitions
    
    ntrans = len(transitions)
    total_wsim = None
    n_spec = len(spectra)
    for spec in spectra:
        source = spec.source
        feature_matrix = source["feature_matrix"].value
        sampling_matrix = spec["sampling_matrix"].value
        sf = sampling_matrix*feature_matrix
        cur_wsim = sf.transpose()*sf
        if total_wsim is None:
            total_wsim = cur_wsim
        else:
            total_wsim = total_wsim + cur_wsim
    total_wsim = (1.0/n_spec)*total_wsim
    
    pseudostrengths = [t.pseudo_strength() for t in transitions]
    eps = [t.ep for t in transitions]
    fitness_func = make_fitness_function(
        total_wsim, 
        pseudostrengths=pseudostrengths, 
        eps=eps, 
    )
    
    species_vals = np.array([t.ion.z + 0.1*t.ion.charge for t in transitions])
    unique_species, sp_group_id = np.unique(species_vals, return_inverse=True)
    
    sub_pops = []
    for sp_idx in range(len(unique_species)):
        print("working on species {}".format(unique_species[sp_idx]))
        trans_mask = sp_group_id == sp_idx
        dplot_path=None
        if not diagnostic_plot_dir is None:
            cz = int(unique_species[sp_idx])
            ccharge = int(round(10*(unique_species[sp_idx] % 1)))
            #symb = tmb.periodic_table.atomic_sy
            dplot_path = os.path.join(diagnostic_plot_dir, "{}_fitness_history.png".format(unique_species[sp_idx]))
        sub_groupings = evolve_sub_groups(
            shared_parameter_space,
            transition_mask=trans_mask,
            fitness_function=fitness_func,
            population_size=population_size,
            n_generations=n_generations,
            mating_fraction=mating_fraction,
            mutate_frequency=0.8,
            starting_population=None,
            diagnostic_plot_path=dplot_path,
        )
        sub_pops.append(sub_groupings)
    return sub_pops

def evolve_sub_groups(
        shared_parameter_space,
        transition_mask,
        fitness_function,
        population_size=400,
        n_generations=5,
        mating_fraction=0.4,
        mutate_frequency=0.8,
        starting_population=None,
        diagnostic_plot_path=None,
):
    n_species_trans = np.sum(transition_mask)
    n_transitions = len(transition_mask)
    valid_idxs = np.where(transition_mask)[0]
    population = []
    if not starting_population is None:
        population.extend(starting_population)
    print("making starting population")
    for i in range(population_size):
        min_n_g = int(np.power(n_species_trans, 1.0/3.0))
        max_n_g = max(min_n_g, n_species_trans//2)
        n_groups = np.random.randint(min_n_g, max_n_g+1)
        grouped_indexes = [[] for i in range(n_groups)]
        for i in range(n_species_trans):
            cur_idx = valid_idxs[i]
            group_to_add_to = np.random.randint(n_groups)
            grouped_indexes[group_to_add_to].append(cur_idx)
        
        gvecs = []
        vec_shape = (n_transitions, 1)
        for gr_idxs in grouped_indexes:
            c_ntrans = len(gr_idxs)
            if c_ntrans > 0:
                ij_mat = (gr_idxs, np.zeros(c_ntrans))
                dat = np.repeat(1.0/len(gr_idxs), len(gr_idxs))
                vec = scipy.sparse.csc_matrix((dat, ij_mat), shape=vec_shape)
                gvecs.append(vec)
        population.append(TransitionGrouping(scipy.sparse.hstack(gvecs), fitness_function=fitness_function))
    
    historical_fitness = []
    hist_fdicts = []
    n_keep = max(2, int(mating_fraction*population_size))
    for iter_idx in range(n_generations):
        print("mating and mutating population")
        print("generation {}".format(iter_idx))
        population = sorted(population, key=lambda x: x.fitness)[-n_keep:]
        fitnesses = [p.fitness for p in population]
        historical_fitness.append(fitnesses)
        cumsum_fit = np.cumsum(fitnesses)
        x_vals = np.zeros(n_keep+1)
        x_vals[1:] = cumsum_fit/cumsum_fit[-1]
        fitness_indexer = interp1d(x_vals, np.arange(n_keep+1))
        
        while len(population) < population_size:
            idx1 = int(fitness_indexer(np.random.random()))
            idx2 = int(fitness_indexer(np.random.random()))
            loop_count = 0
            while idx2 == idx1:
                idx2 = int(fitness_indexer(np.random.random()))
                loop_count += 1
                if loop_count > 20:
                    print("warning could not find a mate")
                    break
            offspring = population[idx1].mate(population[idx2])
            if np.random.random() < mutate_frequency:
                offspring = offspring.mutate()
            population.append(offspring)
    population = sorted(population, key=lambda x: x.fitness)[-n_keep:]
    fitnesses = [p.fitness for p in population]
    historical_fitness.append(fitnesses)
    print("diagnostic plot path", diagnostic_plot_path)
    if not diagnostic_plot_path is None:
        fig, axes = plt.subplots(3, sharex=True)
        for gen_idx, past_fit in enumerate(historical_fitness):
            axes[0].plot(past_fit, color="k", alpha=0.4)
        axes[0].plot(past_fit, color="r", lw=2.0)
        annotate_loc = (2, np.min(past_fit))
        axes[0].annotate("max fitness {:7.3f}".format(np.max(fitnesses)), annotate_loc)
        fig.savefig(diagnostic_plot_path)
        plt.close()
    return population


def gmat_to_exemplar_dict(gmat, indexer):
    exemp_dict = {}
    for gvec in [gmat[:, i] for i in range(gmat.shape[1])]:
        gvec = gvec.tocsc()
        group_trans = [indexer.transitions[tidx] for tidx in gvec.indices]
        eps = np.array([t.ep for t in group_trans])
        psts = np.array([t.pseudo_strength() for t in group_trans])
        
        mean_eps = np.mean(eps)
        mean_pst = np.mean(psts)
        min_dev_idx = np.argmin((eps-mean_eps)**2 + (psts-mean_pst)**2)
        exemplar = group_trans[min_dev_idx]
        exemp_dict[exemplar] = group_trans
    return exemp_dict


def get_mixed_exemplar_dict(transition_groups_list, indexer):
    edict = {}
    for tg in transition_groups_list:
        edict.update(gmat_to_exemplar_dict(tg, indexer))
    return edict


def verify_columns(line_data, on_missing="inject"):
    """make sure that the passed in line data has all the 
    columns we expect for a linelist.
    
    expected columns:
    wv        : transition wavelength
    species   : species identifier e.g. 26.1 for Fe II  607 for CN etc etc
    ep        : transition lower level excitation potential
    loggf     : transition likelihood log(gf)
    Z         : number of protons of species
    ion       : ionization stage of species
    ew        : an associated equivalent width
    rad_damp  : radiative damping
    stark_damp: stark damping
    waals_damp: vanderwaals damping
    moog_damp : the MOOG line list damping parameter
    D0        : the dissociation energy for molecular lines
    line_id   : a unique integer for each transition
    """
    nan_cols =[ "wv", "species", "ep", "loggf", "ew", "rad_damp",
                "stark_damp", "waals_damp", "moog_damp", "D0"]
    for col_name in nan_cols:
        try:
            line_data[col_name]
        except KeyError as e:
            if on_missing == "inject":
                line_data[col_name] = np.repeat(np.nan, len(line_data))
            elif on_missing == "raise":
                raise e
            else:
                raise Exception("on_missing option not recognized must be either 'raise' or 'inject'")
    
    #special handling for Z and ion cols derive from species column
    try:
        line_data["Z"]
        line_data["ion"]
    except KeyError as e:
        if on_missing == "inject":
            species = line_data["species"]
            Z = np.array(species, dtype=int)
            ion = np.array((species-Z)*10, dtype=int)
            line_data["Z"] =  Z
            line_data["ion"] = ion
        elif on_missing == "raise":
            raise e
        else:
            raise Exception("on_missing option not recognized must be either 'raise' or 'inject'")
    
    #put the line_id column in if not there
    try:
        line_data["line_id"]
    except KeyError as e:
        if on_missing == "inject":
            line_data["line_id"] = np.arange(len(line_data))
        elif on_missing == "raise":
            raise e
        else:
            raise Exception("on_missing option not recognized must be either 'raise' or 'inject'")

