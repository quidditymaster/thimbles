# Author Tim Anderton

"""A module for representing and fitting piecewise polynomial functions 
with and without regularity constraints.
"""
import numpy as np
lna = np.linalg
poly1d = np.poly1d
from scipy.interpolate import interp1d
from scipy.interpolate import splev, splrep

class Polynomial:
    "represents polynomials P(y) in a centered scaled coordinate y = (x-c)/s"
    
    def __init__(self, coefficients, center=0.0, scale=1.0):
        self.poly = poly1d(coefficients)
        self.center = center
        assert scale != 0
        self.scale = scale
    
    def __call__(self, xdat):
        return self.poly((xdat-self.center)/self.scale)
    
    def deriv(self):
        """returns the polynomial which is the derivative of this one
        """
        new_coeffs = self.poly.deriv().c/self.scale #divide by scale because of chain rule
        return Polynomial(new_coeffs, self.center, self.scale)
    
    def integ(self, integration_constant=0):
        """ returns a polynomial that is the integral of this one
        
        integration_constant: float
        the constant term of the returned polynomial after integration.
        """
        new_coeffs = self.poly.integ(k=integration_constant/self.scale)*self.scale
        return Polynomial(new_coeffs, self.center, self.scale)

class PolynomialBasis:
    "a class representing a collection of polynomials"
    
    def __init__(self, poly_coefficients, center = 0.0, scale = 1.0):
        """coefficients: a (n_basis,  poly_order+1) shaped array 
        containing the polynomial coefficients"""
        self.coefficients = poly_coefficients
        self.coefficients_transpose = self.coefficients.transpose()
        self.n_basis, order_plus_one = poly_coefficients.shape
        self.order = order_plus_one - 1
        self.center = center
        self.scale = scale
        self.basis_polys = []
        for poly_idx in range(self.n_basis):
            new_poly = Polynomial(self.coefficients[poly_idx], self.center, self.scale)
            self.basis_polys.append(new_poly)
    
    def evaluate_to_polynomial(self, basis_coefficients):
        "returns a polynomial as a weighted sum of the basis polynomials"
        output_poly_coeffs = np.dot(self.coefficients, basis_coefficients)
        outpol = Polynomial(output_poly_coeffs, self.center, self.scale)
        return outpol
    
    def realize_basis(self, xvals):
        """returns a (self.n_basis, len(xvals)) shaped array 
        containing the polynomials evaluated at the positions in xvals"""
        xvec = np.array(xvals)
        out_basis = np.zeros((self.n_basis, len(xvec)))
        for basis_idx in range(self.n_basis):
            out_basis[basis_idx] = self.basis_polys[basis_idx](xvec)
        return out_basis
    
def powers_from_max_order(max_orders):
    powers = []
    poly_order = np.array(max_orders) + 1
    nd_iterator = np.ndindex(*poly_order)
    for power_tuple in nd_iterator:
        ptup = np.array(power_tuple)
        if any(poly_order < ptup):
            break
        tot_ord = np.sum(ptup)
        for var_idx in range(len(ptup)):
            if ptup[var_idx] > 0:
                if tot_ord > ptup[var_idx]:
                    break
        powers.append(ptup)
    powers = np.array(powers)
    return powers

class MultiVariatePolynomial:
    
    def __init__(self, coefficients, powers, center=None, scale=None):
        """
        coefficients: the coefficients of the monomial terms 
            (n_terms, [out_dim])
        powers: the powers of each monomial in the polynomial
            (n_terms, n_variables)
            e.g. specify the terms x*y and y*z**2 by rows in powers 
            x*y would have a power row [1, 1, 0], that is first variable
            x enters with power 1, second variable enters with power 1
            and third variable enters with power 0.
            y*z**2 would be [0, 1, 2] first variable enters with power 0
            second variable enters with power 1 and third variable enters
            with power 2.
        center: the effective origin of this polynomial
        scale: an effective rescaling of the x axis.
        """
        #import pdb; pdb.set_trace()
        self.coeffs = np.asarray(coefficients)
        self.powers = np.asarray(powers, dtype = int)
        self.max_powers = np.max(self.powers, axis = 0)
        self.n_coeffs, self.n_dims = self.powers.shape
        if  center is None:
            center = np.zeros(self.n_dims)
        self.center = np.asarray(center)
        if scale is None:
            self.scale = np.ones(self.n_dims, dtype = float)
        else:
            self.scale = np.asarray(scale)
    
    def __add__(self, B):
        assert all(self.center == B.center)
        assert all(self.scale  == B.scale)
        new_coeffs = []
        new_powers = []
        powers_dict = {}
        for coeff_idx in range(self.n_coeffs):
            new_coeffs.append(self.coeffs[coeff_idx])
            new_powers.append(self.powers[coeff_idx])
            powers_dict[tuple(self.powers[coeff_idx])] = coeff_idx
        for coeff_idx in range(B.n_coeffs):
            cpow = tuple(B.powers[coeff_idx])
            out_idx = powers_dict.get(cpow)
            if out_idx != None:
                new_coeffs[out_idx] += B.coeffs[coeff_idx]
            else:
                new_coeffs.append(B.coeffs[coeff_idx])
                new_powers.append(B.powers[coeff_idx])
        return MultiVariatePolynomial(new_coeffs, new_powers, self.center, self.scale)
    
    def __mul__(self, B):
        assert all(self.center == B.center)
        assert all(self.scale  == B.scale)
        new_coeffs = []
        new_powers = []
        powers_dict = {}
        cur_out_idx = 0
        for coeff_idx_1 in range(self.n_coeffs):
            for coeff_idx_2 in range(B.n_coeffs):
                cpow = self.powers[coeff_idx_1] + B.powers[coeff_idx_2]
                tcpow = tuple(cpow)
                ccoeff = self.coeffs[coeff_idx_1]*B.coeffs[coeff_idx_2]
                out_idx = powers_dict.get(tcpow)
                if out_idx != None:
                    new_coeffs[out_idx] += ccoeff
                else:
                    powers_dict[tcpow] = cur_out_idx
                    new_coeffs.append(ccoeff)
                    new_powers.append(cpow)
                    cur_out_idx += 1
        return MultiVariatePolynomial(new_coeffs, new_powers, self.center, self.scale)
    
    def __call__(self, x):
        pofx = self.get_pofx(x)
        return np.dot(pofx, self.coeffs)
    
    def get_pofx(self, x):
        """return an array of the powers of x after shifting and scaling"""
        xnd = np.asarray(x)
        if xnd.ndim == 2:
            #the array is 2d treat the rows of x as different coordinates
            #and the columns of x as the different dimensions
            x2d = (xnd-self.center)/self.scale
        elif xnd.ndim == 1:
            #x is a 1d array assume it is a single multi dim coordinate
            x2d = ((xnd-self.center)/self.scale).reshape((1, -1))
        npts, n_cols = x2d.shape
        n_terms = len(self.powers)
        pofx = np.empty((npts, n_terms))
        for term_idx in range(n_terms):
            result_col = np.ones(npts)
            for var_idx in range(n_cols):
                cpow = self.powers[term_idx, var_idx]
                result_col *= np.power(x2d[:, var_idx], cpow)
            pofx[:, term_idx] = result_col
        return pofx

def multivariate_from_univariate(poly_coeffs, center, scale, axis):
    """creates a multvariate polynomial from a 1d polynomial
    poly_coeffs: the 1d polynomial coefficients highest order first
    center: the multi-dimensional center M(x) is M_shift((x-center)/scale)  
    scale: the multi-dimensional scale
    axis: the number of the dimension which the multivariate polynomial
    will be a function
    """
    n_coeffs = len(poly_coeffs)
    n_dims = len(center)
    powers = np.zeros((n_coeffs, n_dims), dtype = int)
    powers[:, axis] = np.arange(n_coeffs-1, -1, -1)
    return MultiVariatePolynomial(poly_coeffs, powers, center, scale)

class Binning:
    
    def __init__(self, bins):
        self.bins = bins
        self.lb = bins[0]
        self.ub = bins[-1]
        self.n_bounds = len(self.bins)
        self.last_bin = bins[0], bins[1]
        self.last_bin_idx = 0
    
    def get_bin_index(self, xvec):
        xv = np.asarray(xvec)
        input_shape = xv.shape
        xv = np.atleast_1d(xv)
        out_idxs = np.zeros(len(xv.flat), dtype = int)
        for x_idx in range(len(xv.flat)):
            if np.isnan(xv[x_idx]):
                out_idxs[x_idx] = -3
                break
            #check if the last solution still works
            if self.last_bin[0] <= xv[x_idx] <= self.last_bin[1]:
                out_idxs[x_idx] = self.last_bin_idx
                continue
            elif self.lb > xv[x_idx]:
                out_idxs[x_idx] = -1
                continue
            elif self.ub < xv[x_idx]:
                out_idxs[x_idx] = -2
                continue
            lbi, ubi = 0, self.n_bounds-1
            #import pdb; pdb.set_trace()
            while True:
                mididx = (lbi+ubi)/2
                midbound = self.bins[mididx]
                if midbound <= xv[x_idx]:
                    lbi = mididx
                else:
                    ubi = mididx
                if self.bins[lbi] <= xv[x_idx] <= self.bins[lbi+1]:
                    self.last_bin = self.bins[lbi], self.bins[lbi+1]
                    self.last_bin_idx = lbi
                    break
            out_idxs[x_idx] = lbi
        out_idxs = out_idxs.reshape(input_shape)
        return out_idxs

class PiecewisePolynomial:
    
    def __init__(self, coefficients, control_points, centers = None, 
                 scales = None, bounds = (float("-inf"), float("inf")), 
                 fill_value = np.nan):
        """represents a piecewise polynomial function which transitions from one polynomial
        to the next at the control points.
        coefficients should be an (m, n+1) array
        m is the number of polynomial pieces == len(control_points) + 1
        n is the order of the polynomial pieces
        
        The function takes on values which are determined by the polynomial coefficients with the highest order terms coming first and each polynomail being centered around either the corresponding value in the centers array if it is passed as an argument By default the center is chosen as the midpoint of its two bounding points. If one of the current bounding points is + or -infinity the other bounding point is taken as the "center" of that polynomial bin
        
        Example:
        coefficients = np.array([[3, 2], [1, 0], [-1, -1]]) control_points = [5, 6]
        and bounds = (-float('inf'), 8)
        
        because the centers are 
        
        would be evaluated at a point x < 5 as 
        y = 3*(x-5) + 2
        
        and at a point 5 < x < 6 
        
        y = 1*(x-4.5) + 0
        
        and at a point 6 < x < 8
        
        y = -1*(x-7) + -1
        
        points above the upper bound of 8 will return nan
        """
        self.coefficients = np.asarray(coefficients, dtype=float)
        self.bounds = bounds
        self.control_points = control_points
        n_polys, poly_order = self.coefficients.shape
        self.poly_order = poly_order
        self.ncp = len(control_points)  
        self.fill_value = fill_value
        boundary_points = np.zeros(self.ncp+2)
        boundary_points[0] = bounds[0]
        boundary_points[-1] = bounds[1]
        boundary_points[1:-1] = control_points
        self.binning = Binning(boundary_points)
        self.n_polys = n_polys
        if centers is None:
            self.centers = np.zeros(n_polys)
            #set the centers in such a way to allow for infinite bounds
            for center_idx in range(n_polys):
                lb = boundary_points[center_idx]
                ub = boundary_points[center_idx+1]
                if lb == float("-inf"):
                    lb = boundary_points[center_idx+1]
                if ub == float("inf"):
                    ub = boundary_points[center_idx]
                self.centers[center_idx] = 0.5*(lb+ub)
        else:
            self.centers = centers
        if scales is None:
            self.scales = np.ones(n_polys)
        else:
            self.scales = scales
        self.poly_list = []
        for poly_idx in range(n_polys):
            self.poly_list.append(Polynomial(coefficients[poly_idx], self.centers[poly_idx], self.scales[poly_idx]))
    
    def __call__(self, x):
        x_in = np.asarray(x)
        output = np.zeros(x_in.shape)
        poly_idxs = self.binning.get_bin_index(x_in)
        output[np.isnan(poly_idxs)] = self.fill_value
        for p_idx in range(self.n_polys):
            pmask = poly_idxs == p_idx
            output[pmask] = self.poly_list[p_idx](x_in[pmask])
        return output
    
    def deriv(self):
        """return a piecewise polynomial which is this ones derivative"""
        new_cfs = [pol.deriv().poly for pol in self.poly_list]
        out_ppol = PiecewisePolynomial(new_cfs, self.control_points, centers=self.centers, 
                 scales=self.scales, bounds=self.bounds, 
                 fill_value=self.fill_value)
        return out_ppol

    def integ(self, integration_constant=0):
        """integrate each polynomial and set their constants of integration
        to preserve continuity. 
        """
        #import pdb; pdb.set_trace()
        integ_polys = [pol.integ() for pol in self.poly_list]
        integ_polys[0].poly.coeffs[-1] = integration_constant
        for interface_idx in range(len(self.poly_list)-1):
            interface_pt = self.binning.bins[interface_idx+1]
            rval = integ_polys[interface_idx](interface_pt)
            lval = integ_polys[interface_idx+1](interface_pt)
            integ_polys[interface_idx+1].poly.coeffs[-1] = rval-lval
        new_cfs = np.array([intp.poly.coeffs for intp in integ_polys])
        out_ppol = PiecewisePolynomial(new_cfs, self.control_points, centers=self.centers, 
                 scales=self.scales, bounds=self.bounds, 
                 fill_value=self.fill_value)
        return out_ppol

class InvertiblePiecewiseQuadratic(PiecewisePolynomial):

    def __init__(self, coefficients, control_points, centers = None, 
                 scales = None, bounds = (float("-inf"), float("inf")), 
                 y_bounds=(float("-inf"), float("inf")), fill_value = np.nan):
        """if the provided coefficients represent a monotonic function then
        the inverse method of this class should provide an inverse
        """
        assert coefficients.shape[1] == 3 #must be a quadratic
        PiecewisePolynomial.__init__(self, coefficients, control_points, 
                 centers=centers, scales=scales, bounds=bounds, 
                 fill_value=fill_value)
        self.y_control_points = self(control_points)
        y_boundary_points = np.empty(len(control_points)+2)
        y_boundary_points[0] = y_bounds[0]
        y_boundary_points[-1] = y_bounds[1]
        y_boundary_points[1:-1] = self.y_control_points
        self.y_binning = Binning(y_boundary_points)
        #figure out whether we have a positive derivative
        #and therefore should use a + sign in the inverse formula
        #or a negative derivative and therefore use a minus sign
        self.branch_sign = np.sign(self.poly_list[0].deriv()(self.centers[0]))
    
    def inverse(self, yvals):
        output = np.zeros(yvals.shape)
        poly_idxs = self.y_binning.get_bin_index(yvals)
        output[poly_idxs < 0] = self.fill_value
        for p_idx in range(self.n_polys):
            pmask = poly_idxs == p_idx
            a, b, c = self.poly_list[p_idx].poly.coeffs
            x = (-b+self.branch_sign*np.sqrt(b**2-4.0*a*(c-yvals[pmask])))/(2.0*a)
            output[pmask] = x*self.scales[p_idx]+self.centers[p_idx]
        return output

class PiecewisePolynomialBasis(PolynomialBasis):
    
    def __init__(self, coefficients, control_points, centers = None, 
        scales = None, bounds = (float("-inf"), float("inf")), 
        fill_value = np.nan):
        """coefficients: a (n_basis, n_control_points+1, poly_order+1) shaped array 
        containing the polynomial coefficients"""
        self.coefficients = coefficients
        self.n_basis, n_cp_plus_one, order_plus_one = coefficients.shape
        assert n_cp_plus_one == len(control_points) + 1
        self.order = order_plus_one - 1
        self.n_cp = n_cp_plus_one - 1
        if centers == None:
            centers = np.zeros(self.n_cp + 1)
        self.centers = centers
        if scales == None:
            scales = np.ones(self.n_cp + 1)
        self.scales = scales
        self.control_points = control_points
        self.bounds = bounds
        self.fill_value = fill_value
        self.basis_piecewise_polys = []
        for basis_idx in range(self.n_basis):
            cur_coeffs = self.coefficients[basis_idx, ]
            new_ppol = PiecewisePolynomial(cur_coeffs, self.control_points, 
                                           centers=self.centers, scales=self.scales,
                                           bounds=self.bounds, 
                                           fill_value=self.fill_value)
            self.basis_piecewise_polys.append(new_ppol)
    
    def realize_piecewise_polynomial(self, basis_coefficients):
        "returns a polynomial as a weighted sum of the basis polynomials"
        output_poly_coeffs = np.dot(self.coefficients.reshape((self.n_basis, -1)), basis_coefficients)
        output_ppol = PiecewisePolynomial(output_poly_coeffs, self.control_points,
                                          centers=self.centers, scales=self.scales,
                                          bounds=self.bounds, 
                                          fill_value=self.fill_value)
        return output_ppol
    
    def realize_basis(self, xvals):
        """returns a (self.n_basis, len(xvals)) shaped array 
        containing the polynomials evaluated at the positions in xvals"""
        xvec = np.array(xvals)
        out_basis = np.zeros((self.n_basis, len(xvec)))
        for basis_idx in range(self.n_basis):
            out_basis[basis_idx] = self.basis_piecewise_polys[basis_idx](xvec)
        return out_basis

class MultiVariatePiecewisePolynomial():
    
    def __init__(self, coefficients, powers, control_points,
                 centers=None, scales=None, bounds=None,
                 fill_value = np.nan):
        """a piecewise polynomial with multiple input variables.
        
        inputs
        coefficients: a list of the polynomial coefficients
            shaped as (n_pieces, n_coefficients)
        powers: the particular powers associated with the coefficients.
        control_points: a list of lists of the control points along each 
            dimension. The control points are tensored together to determine
            the break points between adjacent multivariate polynomials.
        centers: optional centers to use for the polynomials
        scales: optional scales to use for the polynomials
        bounds: optional bounds
        fill_value: value to return for coordinates outside of the bounds.
        """
        self.coefficients = coefficients
        self.powers = powers
        if bounds == None:
            bounds = [(-float("inf"), float("inf"))]
        self.bounds = bounds
        self.control_points = control_points
        n_polys, poly_order = coefficients.shape
        self.poly_order = poly_order
        self.ncp_vec = [len(cpoints) for cpoints in control_points]
        self.fill_value = fill_value
        boundary_points = np.zeros(self.ncp+2)
        boundary_points[0] = bounds[0]
        boundary_points[-1] = bounds[1]
        boundary_points[1:-1] = control_points
        self.binning = Binning(boundary_points)
        self.n_polys = n_polys
        if centers == None:
            self.centers = np.zeros(n_polys)
            #set the centers in such a way to allow for infinite bounds
            for center_idx in range(n_polys):
                lb = boundary_points[center_idx]
                ub = boundary_points[center_idx+1]
                if lb == float("-inf"):
                    lb = boundary_points[center_idx+1]
                if ub == float("inf"):
                    ub = boundary_points[center_idx]
                self.centers[center_idx] = 0.5*(lb+ub)
        else:
            self.centers = centers
        if scales == None:
            self.scales = np.ones(n_polys)
        else:
            self.scales = scales
        self.poly_list = []
        for poly_idx in range(n_polys):
            self.poly_list.append(Polynomial(coefficients[poly_idx], self.centers[poly_idx], self.scales[poly_idx]))

def interp_by_index(output_indexes, yvals):
    index_interper = interp1d(np.arange(len(yvals)), yvals)
    return index_interper(output_indexes)

class RegularityConstrainedPiecewisePolynomialBasis:
    
    def __init__(self, poly_order, control_points, 
                 centers = None, scales = None, 
                 regularity_constraints = None, 
                 bounds = (float("-inf"), float("inf")),
                 b_spline_rotation=True):
        self.bounds = bounds
        self.control_points = control_points
        self.poly_order = poly_order
        self.ncp = len(control_points)
        if regularity_constraints == None:
            self.regularity_constraints = np.ones((poly_order, self.ncp), dtype = bool)
        else:
            self.regularity_constraints = regularity_constraints
        boundary_points = np.zeros(self.ncp+2)
        boundary_points[0] = bounds[0]
        boundary_points[-1] = bounds[1]
        boundary_points[1:-1] = control_points
        self.binning = Binning(boundary_points)
        n_polys = self.ncp+1
        self.n_polys = n_polys
        if centers == None:
            self.centers = np.zeros(n_polys)
            #set the centers in such a way to allow for infinite bounds
            for center_idx in range(n_polys):
                lb = boundary_points[center_idx]
                ub = boundary_points[center_idx+1]
                if lb == float("-inf"):
                    lb = boundary_points[center_idx+1]
                if ub == float("inf"):
                    ub = boundary_points[center_idx]
                self.centers[center_idx] = 0.5*(lb+ub)
        else:
            self.centers = centers
        if scales == None:
            scales = np.ones(n_polys)
        self.scales = scales
        poly_basis_list = [[] for _i in range(n_polys)]
        for poly_i in range(n_polys):
            #cdomain = (self.boundary_points[poly_i], self.boundary_points[poly_i+1])
            for comp_i in range(poly_order+1):
                comp_vec = np.zeros((poly_order+1))
                comp_vec[comp_i] = 1.0
                #poly_basis_list[poly_i].append(Legendre(comp_vec, domain = cdomain)) 
                poly_basis_list[poly_i].append(Polynomial(comp_vec, self.centers[poly_i], self.scales[poly_i]))
        #generate the constraint matrix
        #nrows = self.poly_order*self.ncp
        nrows = np.sum(self.regularity_constraints)
        constraint_matrix = np.zeros((nrows, (self.poly_order+1)*self.n_polys))
        constraint_number = 0
        nco, ncp = self.regularity_constraints.shape
        for control_i in range(ncp):
            c_control_point = self.control_points[control_i]
            l_basis = poly_basis_list[control_i] #left basis functions
            r_basis = poly_basis_list[control_i+1] #right basis functions
            for constraint_order in range(nco):
                if not self.regularity_constraints[constraint_order, control_i]:
                    continue
                fp_coeff_idx = control_i*(self.poly_order+1)
                sp_coeff_idx = (control_i+1)*(self.poly_order+1)
                #print "cp", control_i, "sp i", sp_coeff_idx
                for coefficient_i in range(self.poly_order+1):
                    lreg_coeff = l_basis[coefficient_i](c_control_point)
                    rreg_coeff = r_basis[coefficient_i](c_control_point)
                    constraint_matrix[constraint_number, fp_coeff_idx+coefficient_i] = lreg_coeff
                    constraint_matrix[constraint_number, sp_coeff_idx+coefficient_i] = -rreg_coeff
                #go up to the next order constraint by taking the derivative of our basis functions
                constraint_number += 1
                l_basis = [cpoly.deriv() for cpoly in l_basis]
                r_basis = [cpoly.deriv() for cpoly in r_basis]
        self.constraint_matrix = constraint_matrix
        um, sm, vm = lna.svd(self.constraint_matrix, full_matrices=True)
        self.n_basis = (self.poly_order+1)*self.n_polys-nrows
        self.basis_coefficients = np.zeros((self.n_basis, self.n_polys, self.poly_order+1))
        self.basis_polys = [[] for _bi in range(self.n_basis)]
        vclip = vm[-self.n_basis:]
        for basis_i in range(self.n_basis):
            for poly_i in range(self.n_polys):
                coeff_lb = (self.poly_order+1)*poly_i
                coeff_ub = coeff_lb + self.poly_order+1
                ccoeffs = vclip[basis_i, coeff_lb:coeff_ub]
                self.basis_coefficients[basis_i, poly_i] = ccoeffs
                self.basis_polys[basis_i].append(Polynomial(ccoeffs, self.centers[poly_i], self.scales[poly_i]))
        #import pdb; pdb.set_trace()        
        if b_spline_rotation:
            if len(self.control_points) >= 1:
                return
            ctrans = self.b_spline_rotation()
            vrot = np.dot(ctrans.transpose(), vclip)
            self.basis_polys = [[] for _bi in range(self.n_basis)]
            for basis_i in range(self.n_basis):
                for poly_i in range(self.n_polys):
                    coeff_lb = (self.poly_order+1)*poly_i
                    coeff_ub = coeff_lb + self.poly_order+1
                    ccoeffs = vrot[basis_i, coeff_lb:coeff_ub]
                    self.basis_coefficients[basis_i, poly_i] = ccoeffs
                    self.basis_polys[basis_i].append(Polynomial(ccoeffs, self.centers[poly_i], self.scales[poly_i]))
        #import pdb; pdb.set_trace()    
    
    def b_spline_rotation(self):
        #evaluate the b splines
        #import pdb; pdb.set_trace()
        bound_pts = np.zeros(len(self.centers) + 2)
        bound_pts[1:-1] = self.centers
        if len(self.centers) > 1:
            bound_pts[0] = self.centers[0] - np.abs(self.centers[1] - self.centers[0])
            bound_pts[-1] = self.centers[-1] + np.abs(self.centers[-2] - self.centers[-1])
        else:
            #no point in rotating the coefficients
            return np.diag(np.ones(self.n_basis), dtype=float)
        iterp_idexes = np.linspace(0, len(bound_pts)-1, self.n_basis)
        sprep_inp_pts = interp_by_index(iterp_idexes, bound_pts)
        tck = splrep(sprep_inp_pts, sprep_inp_pts, k=self.poly_order)
        knot_pts = bound_pts
        n_knots = len(knot_pts)
        xvals = np.zeros(n_knots + 2*(self.poly_order+1)*(n_knots-1))
        knot_diffs = knot_pts[1:]-knot_pts[:-1]
        xvals[-n_knots:] = knot_pts
        frac_alpha = 1.0/float(self.poly_order+2)
        mid_idx = (self.poly_order+1)*(n_knots-1)
        for p_idx in range(self.poly_order+1):
            xvals[p_idx*(n_knots-1):(p_idx+1)*(n_knots-1)] = knot_pts[1:]-0.5*((p_idx+1)*frac_alpha)*knot_diffs
            xvals[mid_idx+p_idx*(n_knots-1):mid_idx+(p_idx+1)*(n_knots-1)] = knot_pts[:-1]+0.5*((p_idx+1)*frac_alpha)*knot_diffs
        xvals = np.sort(xvals)
        bsp_tup = (tck[0], np.diag(np.ones(self.n_basis)), self.poly_order)
        b_splines = np.asarray(splev(xvals, bsp_tup))
        cur_basis = self.get_basis(xvals)
        coeff_trans = np.linalg.lstsq(cur_basis.transpose(), b_splines.transpose())[0]
        #import pdb; pdb.set_trace()
        return coeff_trans
    
    def get_basis(self, in_vec):
        xvec = np.array(in_vec)
        poly_idxs = self.binning.get_bin_index(xvec)
        out_basis = np.zeros((self.n_basis, len(xvec)))
        for basis_idx in range(self.n_basis):
            for poly_idx in range(self.n_polys):
                xmask = poly_idxs == poly_idx 
                cx = xvec[xmask]
                out_basis[basis_idx][xmask] = self.basis_polys[basis_idx][poly_idx](cx)
        return out_basis


def fit_piecewise_polynomial(x, y, order, control_points, bounds = (float("-inf"), float("inf")), regularity_constraints = None, centers = None, scales = "autoscale"):
    if scales == "autoscale":
        scales = np.ones(len(control_points)+1, dtype=float)*np.std(x)*(len(control_points)+1)
    pp_gen = RegularityConstrainedPiecewisePolynomialBasis(order, control_points=control_points, bounds = bounds, regularity_constraints = regularity_constraints, centers = centers, scales = scales)
    gbasis = pp_gen.get_basis(x)
    n_polys = len(control_points) + 1
    n_coeffs = order+1
    out_coeffs = np.zeros((n_polys, n_coeffs))
    fit_coeffs = np.linalg.lstsq(gbasis.transpose(), y)[0]
    for basis_idx in range(pp_gen.n_basis):
        c_coeffs = pp_gen.basis_coefficients[basis_idx].reshape((n_polys, n_coeffs))
        out_coeffs += c_coeffs*fit_coeffs[basis_idx]
    return PiecewisePolynomial(out_coeffs, control_points, centers=centers, scales=scales, bounds=bounds)
           
RCPPB = RegularityConstrainedPiecewisePolynomialBasis #a shortcut for the reiculously long name

if __name__ == "__main__":
    import matplotlib.pyplot as plt #@UnusedImport
    
    p1coeffs = np.array([1.0, 2.0, 3.0])
    p1 = Polynomial(p1coeffs, scale = 1.0, center=-1.0)
    p1d = p1.deriv()
    pol_der_pass = False
    if (p1d.poly[0] == 2.0) and (p1d.poly[1] == 2.0):
        print("PASSED simple poly derivative test")
    else:
        print("FAILED simple poly derivative test")
    
    test_x = np.linspace(-1, 1, 4000)
    #test_y = np.sin(2.1*np.pi*test_x)
    test_y = test_x * 2 - test_x**2 + 3.14
    
    ppol = fit_piecewise_polynomial(test_x, test_y, 2, np.array([-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75]))
    fit_y = ppol(test_x)
    print("starting 1d piecewise poly tests")
    if np.sum(np.abs(test_y-fit_y)) <= 1e-10:
        print("PASSED exact fit test")
    else:
        print("FAILED exact fit test")
    
    #piecewise polynomial integration and differentiation tests
    ppol_to_int = PiecewisePolynomial(coefficients=np.array([[2.0, 0.5], [0.15, 1.15]]),
                                      control_points=[1], 
                                      centers=[0.5, 1.5],
                                      scales=[2.0, 0.5])
    proper_integ = PiecewisePolynomial(coefficients=np.array([[0.5, 0.0, 0.0], [0.15, 0.7, -0.35]]),
                                      control_points=[1],
                                      centers=[0.0, 0.0])
    x = np.linspace(0, 2, 100)
    y = ppol_to_int(x)
    int_ppol = ppol_to_int.integ(0.125)
    yint_true = proper_integ(x)
    yint_test = int_ppol(x)
    if np.sum(np.abs(yint_test-yint_true)) < 1e-10:
        print("PASSED simple piecewise poly integration test")
    else:
        print("FAILED simple piecewise poly integration test")
    
    #test the invertible quadratic
    inv_qp = InvertiblePiecewiseQuadratic(proper_integ.coefficients, control_points=[1], centers=[0.0, 0.0])
    
    print("starting multi variate polynomial tests")
    A = MultiVariatePolynomial([2.3, 1], [[0, 0], [0, 1]], center = 0.0, scale = 1.0)
    B = MultiVariatePolynomial([1, 2], [[1, 1], [0, 2]], center = 0.0, scale = 1.0)
    
    evaluation_test_passed = True
    zerovec = np.zeros(2) 
    if (A(zerovec) != 2.3) and (B(zerovec) != 0):
        evaluation_test_passed = False
    ones_vec = np.ones(2)
    if A(ones_vec) != 3.3:
        evalutation_test_passed = False
    if B(ones_vec) != 3:
        evaluation_test_passed = False
    pi_e = np.array([np.pi, np.e])
    if B(pi_e) != np.pi*np.e + np.e**2*2:
        evaluation_test_passed = False
    if evaluation_test_passed:
        print("PASSED evaluation test")
    else:
        print("FAILED evaluation test")
    
    ##orthogonalization 
    #randx = np.random.random(30)-0.5
    #rcppb = RCPPB(3, [-0.5, 0.5])
    
