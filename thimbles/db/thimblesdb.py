
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from . import _db_url

#if not len(_db_path):
#    raise ValueError("Must import `thimbles.db` and execute `timbles.db.set_db_path`")

engine = create_engine(_db_url)
#engine = create_engine("sqlite:////home/tim/sandbox/sqlalchemy/test.db")

Session = sessionmaker(bind=engine)
session = Session()
