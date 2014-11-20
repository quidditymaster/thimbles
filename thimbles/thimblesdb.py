import os

import numpy as np
import h5py

from sqlalchemy import ForeignKey
from sqlalchemy import Column, Date, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import relationship, backref

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from thimbles.options import Option, opts, OptionSpecificationError

Base = declarative_base()

class ThimblesDB(object):
    """ a class to encapsulate a thimbles style data format as a combination
    of an SQL database an hdf5 file and pickle files. (SQL to handle irregular 
    collections and links between data, hdf5 to handle large regular
    numerical data arrays and pickle to handle everything else.)
    
    instead of directly instantiating this class use the get_db function in
    this module
    """
    
    def __init__(self, path):
        self.path = os.path.abspath(path)
        if not os.path.isdir(self.path):
            raise Exception("{} is not a valid directory".format(self.path))
        
        #set up the database
        self.db_url = "sqlite:///{}tdb.db".format(self.path)
        self.engine = create_engine(self.db_url)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        #set up the hdf5 file
        hdf5_path = os.path.join(self.path, "tdb.h5")
        self.h5 = h5py.File(hdf5_path, "r+")
    
    def save(self):
        self.session.commit()
    
    def close(self):
        self.session.close()
        self.h5.close()

current_dbs = {}
Option("thimblesdb", option_style="parent_dict")
Option("path", parent="thimblesdb", option_style="raw_string",  envvar="THIMBLESDB.PATH", default=None)
current_db_path = os.environ.get("THIMBLESPROJECTDB", None)
if not current_db_path is None:
    current_db_path = os.path.abspath(current_db_path)

def set_db(path):
    global current_db_path
    current_db_path = os.path.abspath(path)

def get_db(path=None):
    if path is None:
        if current_db_path is None:
            raise Exception("No database path set")
        path = current_db_path
    abspath = os.path.abspath(path)
    new_db = current_dbs.get(abspath)
    if not new_db is None:
        return new_db
    else:
        return ThimblesDB(abspath)

class ThimblesTable(object):
    
    @declared_attr
    def __tablename__(cls):
        return cls.__name__
    
    #db = get_db()
    _id = Column(Integer, primary_key=True)
    
    def load_nsqla(self, **kwargs):
        """load the columns which aren't stored in the SQL database
        """
        self.metadata
        if isinstance(db, basestring):
            db = get_db(db)
        elif not isinstance(db, ThimblesDB):
            raise ValueError("load requires either a valid db path or a ThimblesDB instance")
        if False:
            pass
            #TODO: search for instances in the database that match the given kwargs
        else:
            new_instance = cls(**kwargs)
            new_instance.db = db
    
    def unload(self):
        raise NotImplementedError()

class ArrayColumn(object):
    """a column type to be used for large numerical data arrays.
    """
    
    def __init__(self):
        pass
    
