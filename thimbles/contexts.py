
from copy import copy
import thimbles.workingdataspace as wds
from thimbles.sources import Source
import thimbles as tmb
from thimbles.star import Star
from thimbles.spectrographs import Aperture, Chip, Order
from thimbles.observations import Exposure
from thimbles.spectrum import Spectrum

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
        q = db.query(*self.table_spec)
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

    def set_context_registry(self, reg):
        self.cregistry = reg


class ContextualizationRegistry(object):
    
    def __init__(self):
        self.global_contexts = {}
        self.spines = {}
        self.modeling_templates = {}
    
    def add_global(self, name, instance):
        self.global_contexts[name] = instance
    
    def __getitem__(self, index):
        return self.spines[index]
    
    def __setitem__(self, index, value):
        value.set_context_registry(self)
        self.spines[index] = value


contextualizers = ContextualizationRegistry()
global_ce = ContextualizationEngine(
    table_spec=[tmb.analysis.SharedParameterSpace],
    filter_factory=lambda q: q.filter(tmb.analysis.SharedParameterSpace.name == "global"),
)
contextualizers["global"] = global_ce

star_contextualizer = ContextualizationEngine(
    table_spec = [Star],
    tag_filter_factory = lambda query, name : query.filter(Star.name == name),
    extractors={
        "star": lambda x: x,
    },
)
contextualizers["stars"] = star_contextualizer

SourceGrouping = tmb.sources.SourceGrouping
grouping_contextualizer = ContextualizationEngine(
    table_spec=[SourceGrouping],
    tag_filter_factory=lambda query, name: query.filter(SourceGrouping.name == name),
    extractors = {
        "group": lambda x: x
    },
)
contextualizers["source_groupings"] = grouping_contextualizer

aperture_contextualizer = ContextualizationEngine(
    table_spec = Aperture,
    tag_filter_factory = lambda query, name : query.filter(tmb.spectrographs.Aperture.name == name),
    extractors={
        "aperture": lambda x:x,
    }
)
contextualizers["apertures"] = aperture_contextualizer

order_contextualizer = ContextualizationEngine(
    table_spec=[tmb.spectrographs.Order],
    tag_filter_factory = lambda query, number : query.filter(Order.number == number),
    extractors={
        "order": lambda x: x
    }
)
contextualizers["orders"] = order_contextualizer

chip_contextualizer = ContextualizationEngine(
    table_spec = [Order],
    tag_filter_factory = lambda query, name : query.filter(Chip.name == name),
    extractors = {
        "chip": lambda x: x,
    }
)
contextualizers["chips"] = chip_contextualizer

exposure_contextualizer = ContextualizationEngine(
    table_spec = [Exposure],
    tag_filter_factory = lambda query, name : query.filter(Exposure.name == name),
    extractors = {
        "exposure": lambda x: x,
    }
)
contextualizers["exposures"] = exposure_contextualizer


spectrum_contextualizer = ContextualizationEngine(
    table_spec = [Spectrum],
    extractors = {
        "spectrum" : lambda x: x,
        "source" : lambda x: x.source,
        "exposure" : lambda x: x.exposure,
        "aperture" : lambda x: x.aperture,
        "chip": lambda x: x.chip
    }
)
contextualizers["spectra"] = spectrum_contextualizer


source_spectra_pairs = ContextualizationEngine(
    table_spec = [Spectrum, Source],
    extractors = {
        "spectrum" : lambda x: x[0],
        "source" : lambda x: x[1],
        "exposure" : lambda x: x[0].exposure,
        "aperture" : lambda x: x[0].aperture,
        "chip": lambda x: x[0].chip
    }
)
contextualizers["source spectrum pairs"] = source_spectra_pairs

