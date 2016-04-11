
import thimbles as tmb
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("linelist")
parser.add_argument("--output", required=True)
parser.add_argument("--min-pseudostrength", type=float, default=-2.0)
parser.add_argument("--atomic-parameter-cuts", nargs="*")
parser.add_argument("--included-elements", nargs="*")
parser.add_argument("--max-teff", default=8000.0, type=float)


def param_filter(transition, param_bounds_dict):
    for param_name in param_bounds_dict:
        lb, ub = param_bounds_dict[param_name]
        pval = getattr(transition, param_name)
        if not lb is None:
            if pval < lb:
                return False
        if not ub is None:
            if pval > ub:
                return False
    return True


if __name__ == "__main__":
    args = parser.parse_args()
    
    #apply direct parameter cuts
    atp_bounds_dict = {}
    
    if len(args.atomic_parameter_cuts) % 3 > 0:
        raise ValueError("atomic-parameter-cuts must be specified in the form\n  --atomic-parameter-cuts property_name min max property_name2 min max \n\n for example \n\n --atomic-parameter-cuts ep 0 5.0  wv 4500 9000\n")
    
    atp_cuts = args.atomic_parameter_cuts
    for i in range(len(atp_cuts), 3):
        pname = atp_cuts[i]
        lb, ub = atp_cuts[i+1:i+3]
        if lb.lower() == "none":
            lb = None
        else:
            lb = float(lb)
        if ub.lower() == "none":
            ub = None
        else:
            ub = float(ub)
        
        atp_bounds_dict[pname] == [lb, ub]
    
    transitions = tmb.io.linelist_io.read_linelist(args.linelist)
    n_trans_orig = len(transitions)
    n_prev = n_trans_orig
    print("read in {} transitions".format(n_trans_orig))
    
    #cull the transitions via species
    if len(args.included_elements) > 0:
        z_set = set(list([int(z) for z in args.included_elements]))
        transitions = [trans for trans in transitions if trans.ion.z in z_set]
        print("discareded {} transitions by element".format(n_prev - len(transitions)))
        print("{} transitions remaining".format(len(transitions)))
        n_prev = len(transitions)
    
    #cull the transitions via atomic params
    transitions = [trans for trans in transitions if param_filter(trans, atp_bounds_dict)]
    
    print("discarded {} from atomic parameter cuts".format(n_prev-len(transitions)))
    print("{} transitions remaining".format(len(transitions)))
    n_prev = len(transitions)
    
    #cull the transitions via pseudo-strength
    transitions = [trans for trans in transitions if trans.pseudo_strength(teff=args.max_teff) >= args.min_pseudostrength]
    
    print("discarded {} transitions from pseudostregth".format(n_prev-len(transitions)))
    print("{} transitions remaining".format(len(transitions)))
    
    print("writing result to {}".format(args.output))
    tmb.io.write_linelist(args.output, transitions)
