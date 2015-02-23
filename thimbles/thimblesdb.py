from copy import deepcopy
import os
from thimbles.tasks import task
from thimbles.sqlaimports import *
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy import create_engine
from thimbles.options import Option, opts, OptionSpecificationError
from sqlalchemy.orm import sessionmaker

Session = sessionmaker(expire_on_commit=False)
Base = declarative_base()

class ThimblesTable(object):
    _id = Column(Integer, primary_key=True)    
    
    @declared_attr
    def __tablename__(cls):
        return cls.__name__
    
    @property
    def session(self):
        return self._sa_instance_state.session

Option("database", option_style="parent_dict")
Option("dialect", option_style="raw_string", default="sqlite", parent="database")
Option("echo_sql", option_style="flag", parent="database")

class ThimblesDB(object):
    """a wrapper for a database containing our data and our fit-models and parameters
    """
    
    def __init__(self, path, dialect=None):
        if dialect is None:
            dialect = opts["database.dialect"]
        self.path = os.path.abspath(path)
        self.db_url = "{dialect}:///{path}".format(dialect=dialect, path=self.path)
        self.engine = create_engine(self.db_url, echo=opts["database.echo_sql"])
        Base.metadata.create_all(self.engine)
        self.session = Session(bind=self.engine)
    
    def add(self, obj):
        self.session.add(obj)
    
    def delete(self, obj):
        self.session.delete(obj)
    
    def add_all(self, obj_list):
        self.session.add_all(obj_list)
    
    def query(self, *args, **kwargs):
        return self.session.query(*args, **kwargs)
    
    def commit(self):
        self.session.commit()
        self.session.close()
    
    def close(self):
        self.session.close()
    
    def incorporate(self, data, template, **kwargs):
        global template_registry
        if isinstance(template, basestring):
            template = template_registry[template]
        temp_instance = template(data, self, **kwargs)
        self.add(temp_instance)


@task(result_name="tdb",
    sub_kwargs={"fname":dict(
        option_style="raw_string",
        editor_style="file")},
)
def load_tdb(fname):
    print "running load_tdb"
    return ThimblesDB(fname)

@task(result_name="injection_success")
def add_all(data, tdb):
    if not isinstance(data, list):
        data = [data]
    tdb.add_all(data)
