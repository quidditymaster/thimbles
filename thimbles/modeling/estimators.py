from .derivatives import deriv

from thimbles.sqlaimports import *
from thimbles.modeling.distributions import NormalDistribution


class Estimator(object):
    
    def __init__(
            self,
            informants,
            informed,
            reference_params=None,
            override=None,
            set_values=True,
    ):
        self.informants = informants
        self.informed = informed
        if reference_params is None:
            reference_params = {}
        self.reference_params = reference_params
        if override is None:
            override = {}
        self.override = override
    
    def iterate(self):
        raise NotImplementedError("use a subclass")


class GeneralizedLinearEstimator(Estimator):
    
    def __init__(
            self,
            informants,
            informed,
            reference_params=None,
            override_params=None,
            set_values=True,
            decoupling_factories=None,
            reweighting_factories=None,
            keep_diagnostics = False,
    ):
        Estimator.__init__(
            self,
            informants=informants,
            informed=informed,
            reference_params=reference_params,
            override=override,
            set_values=set_values
        )
        if decoupling_factories is None:
            decoupling_factories = [None for i in range(len(informants))]
        self.decoupling_factories = decoupling_factories
        if reweighting_factories is None:
            decoupling_factories = [None for i in range(len(informants))]
        self.reweighting_factories = reweighting_factories
    
    def iterate(self):
        upstream_parameters = []
        for dist in self.informed:
            for param in dist.context.parameters:
                upstream_parameters.append(param)
        
        downstream_parameters = []
        for dist in self.informants:
            for param in dist.context.parameters:
                downstream_parameters.append(param)
        
        #find the target values and model values
        target_vecs = [dist.mean for dist in downstream_dists]
        model_vecs = [param.value for param in downstream_parameters]
        
        #stitch the vectors together and find the residual vec
        stitched_target = np.hstack(target_vecs)
        stitched_model = np.hstack(model_vecs)
        
        residuals = stitched_target - stitched_model
        
        #cut the residuals up by distribution
        lb, ub = 0, 0
        cut_residuals = []
        for tvec in target_vecs:
            ub += len(tvec)
            cut_residuals.append(residuals[lb:ub])
            lb = ub
        
        target_ivars = [dist.ivar for dist in upstream_dists]
        for dist_idx in range(len(target_ivars)):
            rw_fact = self.reweighting_factories[ivar]
            if not (rw_fact is None):
                target_ivars[dist_idx] = rw_fact(cut_residuals[dist_idx], target_ivars[dist_idx])
        
        #build the derivative matrix
        dmat = tmb.modeling.derivatives.deriv(
            downstream_params,
            upstream_params,
            combine_matrices=True,
            override=override,
        )
        
