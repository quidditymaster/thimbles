from copy import deepcopy
import os

import numpy as np
import h5py

from thimbles.sqlaimports import *

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
    
    def __init__(self, path, create_dir=True, auto_append=".tdb"):
        if not path[-len(auto_append):] == auto_append:
            path = path + auto_append
        self.path = os.path.abspath(path)
        if not os.path.exists(self.path):
            if create_dir:
                os.makedirs(self.path)
                import time; time.sleep(0.1)
            else:
                raise IOError("path does not exisit")
                #raise Exception("{} is not a valid directory".format(self.path))
        
        #set up the database
        self.db_url = "sqlite:///{}/tdb.db".format(self.path)
        self.engine = create_engine(self.db_url)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        #set up the hdf5 file
        hdf5_path = os.path.join(self.path, "tdb.h5")
        self.h5 = h5py.File(hdf5_path)
        self._managed_objs = set()
    
    def add(self, obj):
        self.session.add(obj)
        self.register_instance(obj)
    
    def add_all(self, obj_list):
        self.session.add_all(obj_list)
        for obj in obj_list:
            self.register_instance(obj)
    
    def register_instance(self, obj):
        if isinstance(obj, ThimblesTable):
            self._managed_objs.add(obj)
    
    def query(self, *args, **kwargs):
        return self.session.query(*args, **kwargs)
    
    def all(self, query):
        res = query.all()
        for obj in res:
            self.register_instance(obj)
        return res
    
    def first(self, query):
        res = query.first()
        self.register_instance(res)
    
    def save(self):
        self.session.commit()
        #for obj in self._managed_objs:
        #    obj.save(self)
    
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
    _id = Column(Integer, primary_key=True)    
    
    @declared_attr
    def __tablename__(cls):
        return cls.__name__
    
    def save(self, db):
        """load the columns which aren't stored in the SQL database
        """
        pass #TODO: save out the managed non-sqlalchemy data
    
    def unload(self):
        raise NotImplementedError()

