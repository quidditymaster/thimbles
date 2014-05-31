import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from dbclasses import Base

current_dbs = {}

def get_db(path):
    abspath = os.path.abspath(path)
    new_db = current_dbs.get(abspath)
    if not new_db is None:
        return new_db
    else:
        return(ThimblesDB(abspath))

class ThimblesDB(object):
    """ a class to encapsulate a thimbles style data format as a combination
    of an SQL database an hdf5 file and pickle files. (SQL to handle irregular 
    collections and links between data, hdf5 to handle large regular
    numerical data arrays and pickle to handle everything else.)
    """
    
    def __init__(self, path):
        self.db_url = "sqlite:///{}".format(os.path.abspath(path))
        
        self.engine = create_engine(self.db_url)
        #engine = create_engine("sqlite:////home/tim/sandbox/sqlalchemy/test.db")
        
        Base.metadata.create_all(self.engine)
        
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
