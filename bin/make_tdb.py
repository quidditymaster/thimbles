import thimbles as tmb
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("db_path")
parser.add_argument("--line-list", "--ll")

if __name__ == "__main__":
    args = parser.parse_args()
    
    tdb = tmb.ThimblesDB(args.db_path)
    
    if not (args.line_list is None):
        ll = tmb.io.read_linelist(args.line_list)
        tdb.add_all(ll)
    
    tdb.commit()
    tdb.close()
