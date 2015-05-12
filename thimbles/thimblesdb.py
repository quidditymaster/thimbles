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
Option("dialect", default="sqlite", parent="database")
Option("echo_sql", parent="database", default=False)

class ThimblesDB(object):
    """a wrapper for a database containing our data and our fit-models and parameters

    path: string
      the location of the database
      if the empty string is passed as the path and the dialect is
      sqlite then an in memory sqlite database is generated.
    dialect: string
      the SQL backend database dialect
    echo_sql: bool
      if True then SQL queries will be echoed to the terminal.
    """
    
    def __init__(self, path="", dialect=None, echo_sql=None):
        if dialect is None:
            dialect = opts["database.dialect"]
        self.path = os.path.abspath(path)
        if path == "":
            if dialect =="sqlite":
                self.db_url = "sqlite://"
        else:
            self.db_url = "{dialect}:///{path}".format(dialect=dialect, path=self.path)
        if echo_sql is None:
            echo_sql = opts["database.echo_sql"]
        self.engine = create_engine(self.db_url, echo=echo_sql)
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
    
    def close(self):
        self.session.close()

    def cleanup(self):
        self.commit()
        self.close()

class HasName(object):
    name = Column(String)

@task(result_name="tdb",
    sub_kwargs={"fname":dict(
        editor_style="file")},
)
def load_tdb(fname):
    print("running load_tdb")
    return ThimblesDB(fname)


@task(result_name="operation_error")
def add_all(data, tdb):
    try:
        if not isinstance(data, list):
            data = [data]
        tdb.add_all(data)
    except Exception as e:
        return e
    return None


def find_or_create(
        instance_class,
        auto_add=True,
        database=None,
        **attrs
):
    """generates a simple query for the associated database session
    and attempt to find an existing table entry that has attributes that
    match the passed keyword arguments. If no such instance is found then the passed class is instantiated as instance_class(**attrs) and
    then returned.
    
    """
    try:
        query = database.query(instance_class)
        for attr in attrs:
            col = getattr(instance_class, attr)
            query = query.filter(col == attrs[attr])
        instance = query.one()
    except sa.orm.exc.NoResultFound:
        instance = instance_class(**attrs)
        if auto_add:
            database.add(instance)
    return instance
