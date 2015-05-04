import thimbles as tmb
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("db_path")
parser.add_argument("--line-list", "--ll")
parser.add_argument("--standard_name", default="default")
parser.add_argument("--vac-to-air", action="store_true")

if __name__ == "__main__":
    args = parser.parse_args()
    
    tdb = tmb.ThimblesDB(args.db_path)
    
    if not (args.line_list is None):
        ll = tmb.io.read_linelist(args.line_list)
        if args.vac_to_air:
            for transition in ll:
                transition.wv = tmb.utils.misc.vac_to_air_sdss(transition.wv)
        tdb.add_all(ll)
    
    tstand = tmb.transitions.segmented_grouping_standard(
        standard_name = args.standard_name,
        tdb=tdb,
    )
    tdb.add(tstand)
    
    tdb.commit()
    tdb.close()
