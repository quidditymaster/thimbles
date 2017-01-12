"""interogate the model network specification and database and
instantiate the corresponding model network.
"""
import matplotlib as mpl
mpl.use("qt4Agg")
import thimbles as tmb
import json
from tmb_model_specification import model_network

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("db_path")
parser.add_argument("--transition-linelist", required=True)
parser.add_argument("--exemplar-map", required=True)
parser.add_argument("--commit", action="store_true")


if __name__ == "__main__":
    args = parser.parse_args()
    db_path = args.db_path
    
    db = tmb.ThimblesDB(db_path)
    
        #global_params = db.query(tmb.analysis.SharedParameterSpace).filter(tmb.analysis.SharedParameterSpace.name == 'global').one()
    
    if True:
        #initialize the global parameter space
        global_params = tmb.analysis.SharedParameterSpace("global")
        tmb.contexts.model_spines.add_global("global", global_params)
        
        print("loading linelist {}".format(args.transition_linelist))
        linelist = tmb.io.read_linelist(args.transition_linelist)
        db.add_all(linelist)
    
        ex_map_file = open(args.exemplar_map)
        exemplar_mapping_dict = json.load(ex_map_file)
        #convert the keys from strings to integers
        exemplar_mapping_dict = {int(k):exemplar_mapping_dict[k] for k in exemplar_mapping_dict}
        ex_map_file.close()
        
        print("reconstructing exemplar mapping")
        exemplar_map = {linelist[ex_idx]:[linelist[xi] for xi in exemplar_mapping_dict[ex_idx]] for ex_idx in exemplar_mapping_dict}
        
        exemplars = sorted(exemplar_map.keys(), key=lambda x: (x.ion.z, x.ion.charge, x.wv))
        exemplar_indexer_p = tmb.transitions.TransitionIndexerParameter(transitions=exemplars)
        global_params.add_parameter(
            "exemplar_indexer",
            exemplar_indexer_p
        )
        exemplar_map_p = tmb.transitions.ExemplarGroupingParameter(groups=exemplar_map)
        global_params.add_parameter(
            "exemplar_map",
            exemplar_map_p
        )
        transition_indexer_p = tmb.transitions.TransitionIndexerParameter(transitions=linelist)
        global_params.add_parameter(
            "transition_indexer",
            transition_indexer_p
        )
        
        db.add(global_params)
        #db.commit()
    
    print("constructing model network")
    model_network.initialize_network(db)

    if args.commit:
        db.commit()
