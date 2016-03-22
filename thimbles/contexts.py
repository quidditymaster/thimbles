
from copy import copy
import thimbles.workingdataspace as wds
from thimbles.sources import Source
import thimbles as tmb
from thimbles.star import Star
from thimbles.spectrographs import Aperture, Chip, Order
from thimbles.observations import Exposure
from thimbles.spectrum import Spectrum

class PyEvalQuery(object):
    """
A convenience class for generating queries to databases from 
python strings.
"""
    _global_ns = wds.__dict__
    
    def __init__(self, query_expression):
        self.query_expression = query_expression
    
    def __call__(self, db):
        return eval(self.query_expression, self._global_ns, {"db":db})


class PyEvalFilter(object):
    _global_ns = wds.__dict__
    
    def __init__(self, filter_expression):
        self.filter_epression = filter_expression
    
    def __call__(self, results):
        filter_func = eval(self.filter_expression, self._global_ns)
        return filter_func(results)


class BlockPartitionTagFilter(object):
    
    def __init__(self, block_size):
        self.block_size = block_size
    
    def __call__(self, query, tag):
        offset = tag*self.block_size
        limit = (tag+1)*self.block_size
        return q.offset(offset).limit(limit)


class ContextualizationEngine(object):
    cregistry = None
    _query_global_ns = wds.__dict__
    
    def __init__(
            self,
            query_factory,
            extractors=None,
            tag_filter_factory=None,
            python_filter=None,
    ):
        self.query_factory = query_factory
        self.python_filter = python_filter
        self.tag_filter_factory = tag_filter_factory
        if extractors is None:
            extractors = {}
        self.extractors = extractors
    
    def find(self, tag=None, db=None):
        q = self.query_factory(db=db)
        if (not self.tag_filter_factory is None) and (not (tag is None)):
            q = self.tag_filter_factory(q, tag)
        #actually emit the query
        results = q.all()
        
        #do a post sql query python cleanup 
        if not self.python_filter is None:
            results = self.python_filter(results)
        return results
    
    def add_named_context(self, name, extractor):
        self.extractors[name] = extractor
    
    def contextualize(self, result):
        if not self.cregistry is None:
            cdict = copy(self.cregistry.global_contexts)
        else:
            cdict = {}
        for cname in self.extractors:
            cdict[cname] = self.extractors[cname](result)
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


model_spines = ContextualizationRegistry()
global_spine = ContextualizationEngine(
    PyEvalQuery("db.query(SharedParameterSpace).filter(SharedParameterSpace.name == 'global')"),
)
model_spines["global"] = global_spine


star_spine = ContextualizationEngine(
    PyEvalQuery("db.query(Star)"),
    tag_filter_factory = lambda query, name : query.filter(Star.name == name),
    extractors={
        "star": lambda x: x,
    },
)
model_spines["stars"] = star_spine

SourceGrouping = tmb.sources.SourceGrouping
grouping_spine = ContextualizationEngine(
    PyEvalQuery("db.query(SourceGrouping)"),
    tag_filter_factory=lambda query, name: query.filter(SourceGrouping.name == name),
    extractors = {
        "group": lambda x: x
    },
)
model_spines["source_groupings"] = grouping_spine

aperture_spine = ContextualizationEngine(
    PyEvalQuery("db.query(Aperture)"),
    tag_filter_factory = lambda query, name : query.filter(tmb.spectrographs.Aperture.name == name),
    extractors={
        "aperture": lambda x:x,
    }
)
model_spines["apertures"] = aperture_spine

order_spine = ContextualizationEngine(
    PyEvalQuery("db.query(Order)"),
    tag_filter_factory = lambda query, number : query.filter(Order.number == number),
    extractors={
        "order": lambda x: x
    }
)
model_spines["orders"] = order_spine

chip_spine = ContextualizationEngine(
    PyEvalQuery("db.query(Chip)"),
    tag_filter_factory = lambda query, name : query.filter(Chip.name == name),
    extractors = {
        "chip": lambda x: x,
    }
)
model_spines["chips"] = chip_spine

exposure_spine = ContextualizationEngine(
    PyEvalQuery("db.query(Exposure)"),
    tag_filter_factory = lambda query, name : query.filter(Exposure.name == name),
    extractors = {
        "exposure": lambda x: x,
    }
)
model_spines["exposures"] = exposure_spine


spectrum_spine = ContextualizationEngine(
    PyEvalQuery("db.query(Spectrum)"),
    extractors = {
        "spectrum" : lambda x: x,
        "source" : lambda x: x.source,
        "exposure" : lambda x: x.exposure,
        "aperture" : lambda x: x.aperture,
        "chip": lambda x: x.chip
    }
)
model_spines["spectra"] = spectrum_spine


source_spectra_pairs = ContextualizationEngine(
    PyEvalQuery("db.query(Spectrum, Source).filter(Spectrum._source_id == Source._id)"),
    extractors = {
        "spectrum" : lambda x: x[0],
        "source" : lambda x: x[1],
        "exposure" : lambda x: x[0].exposure,
        "aperture" : lambda x: x[0].aperture,
        "chip": lambda x: x[0].chip
    }
)
model_spines["source spectrum pairs"] = source_spectra_pairs

