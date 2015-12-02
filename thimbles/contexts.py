
from copy import copy
import thimbles.workingdataspace as wds

class ContextualizationEngine(object):
    cregistry = None
    _query_db = None
    _query = None
    
    def __init__(
            self,
            table_spec,
            extractors=None,
            join_spec=None,
            filter_factory=None,
            tag_filter_factory=None,
            post_query_filter=None,
            offset=None,
            limit=100,
    ):
        self.table_spec = table_spec
        self.filter_factory = filter_factory
        self.post_query_filter=post_query_filter
        self.join_spec = join_spec
        self.offset = offset
        self.limit = limit
        if extractors is None:
            extractors = {}
        self.extractors = extractors
    
    def add_named_context(self, name, extractor):
        self.extractors[name] = extractor
    
    def pre_query(self, db=None):
        if db is None:
            db = wds.db
        if self._query_db is db:
            if not self._query is None:
                return self._query
        q = db.query(self.table_spec)
        if not self.join_spec is None:
            q = q.join(self.join_spec)
        if not self.filter_factory is None:
            q = self.filter_factory(q)
        if not self.offset is None:
            q=q.offset(self.offset)
        if not self.limit is None:
            q=q.limit(self.limit)
        return q
    
    def find(self, tag=None, db=None):
        q = self.pre_query(db=db)
        if not (tag is None):
            q = self.tag_filter_factory(q, tag)
        #actually emit the query
        results = q.all()
        #do a post query python cleanup 
        if not self.post_query_filter is None:
            results = self.post_query_filter(results)
        return results
    
    def contextualize(self, result):
        if not self.cregistry is None:
            cdict = copy(self.cregistry.global_contexts)
        else:
            cdict = {}
        for cname in self.extractors:
            cdict[cname] = self.extractors[cname](instance)
        return cdict


class ContextualizationRegistry(object):
    
    def __init__(self):
        self.global_contexts = {}
        self.spines = {}
    
    def add_global(self, name, instance):
        self.global_contexts[name] = instance
    
    def register(self, spine_name, context_engine):
        self.spines[spine_name] = context_engine
        context_engine.cregistry = self


contextualizers = ContextualizationRegistry()
