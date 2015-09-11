import numpy as np
from scipy.interpolate import interp1d
import scipy.sparse
from scipy.sparse.linalg import lsqr
from scipy.sparse import lil_matrix
import scipy.stats
import time
from thimbles.coordinatization import centers_to_edges, edges_to_centers
from thimbles.coordinatization import as_coordinatization
from thimbles.numba_support import double, jit
import matplotlib.pyplot as plt


def uniform_cdf(z):
    return np.clip(z+0.5, 0.0, 1.0)


def gaussian_cdf(z):
    return 0.5*(1.0+scipy.special.erf(z/np.sqrt(2)))


def box_convolved_cdf_factory(box_width=1.0):
    def box_convolved_gaussian_cdf(z):
        t = z/np.sqrt(2)
        bw = box_width/np.sqrt(2)
        tpbw = t+bw
        tmbw = t-bw
        d1 = tpbw*scipy.special.erf(tpbw) - tmbw*scipy.special.erf(tmbw)
        d2 = np.exp(-tpbw**2)-np.exp(-tmbw**2)
        return 0.25*(d1 + d2/np.sqrt(np.pi) + 2.0)
    return box_convolved_gaussian_cdf


def pixel_integrated_lsf(
        x_in, 
        x_out, 
        lsf, 
        lsf_cdf=None, 
        lsf_cut=4.0, 
        normalize="rows", 
):
    """generate a sparse matrix the columns of which correspond to the pixels of
    x_in and the rows to x_out. The entries of each column j represent the 
    definite integral of a line spread function centered at x_in[j] with width 
    lsf[j] and the limits of the integral determined by the output pixels as 
    0.5*(x_out[i*-1]+x[i*]) to 0.5*(x_out[i*+1]+x[i*]) where i* is the index 
    of the output pixel which most closely approximates the coordinate of x_out[j]
    
    parameters
    x_in: ndarray or Coordinatization
      input coordinates
    x_out: ndarray or Coordinatization
      output coordinates
    lsf: ndarray
      the lsf coordinate widths to use for the line spread function centered
      around each input pixel.
    lsf_cdf: function
      function that takes a numpy array and returns the value of the integral
      of an line spread profile with a width of 1. If none is specified 
      defaults to the integral of the standard normal distribution.
    lsf_cut: float
      at what multiple of the primary width to assume that the lsf
      contribution drops off to zero. 
    normalize: string
      after the fact normalization.
    """
    x_in = as_coordinatization(x_in)
    x_out = as_coordinatization(x_out)
    
    if lsf_cdf is None:
        lsf_cdf = scipy.stats.norm.cdf
    
    valid_norms = "rows columns none".split()
    if not normalize.lower() in valid_norms:
        raise ValueError("normalize must be one of {}".format(valid_norms))
    
    in_coords = x_in.coordinates
    out_coords = x_out.coordinates
    
    lsf_in = lsf
    #interpolate the input lsf onto the output lsf
    in_to_out = x_in.interpolant_sampling_matrix(out_coords)
    if not hasattr(lsf_in, "shape"):
        lsf_in = np.repeat(lsf_in, len(x_in))
    lsf_out = in_to_out*lsf_in
    #calculate needed bandwidth in rows for each output pixel
    max_lsf_delt = lsf_out*lsf_cut
    max_out_idxs = x_out.get_index(out_coords+max_lsf_delt)
    min_out_idxs = x_out.get_index(out_coords-max_lsf_delt)
    #adopt the maximum necessary bandwidth of any one row.
    row_bandwidth = int(np.max(np.abs(max_out_idxs-min_out_idxs)) + 1)
    #run from minus the bandwidth to bandwidth+1 because we will be replacing
    #the row coordinates with the row pixel edges
    row_offsets = np.arange(-row_bandwidth, row_bandwidth+2)
    
    #find the center row of our matrix band
    nearest_out_idx = x_out.get_index(in_coords, snap=True)
    out_pixel_edges = centers_to_edges(out_coords)
    edge_idxs = np.clip(nearest_out_idx.reshape((1, -1))+row_offsets.reshape((-1, 1)), 0, len(x_out))
    edge_coordinates = out_pixel_edges[edge_idxs]
    #lsf centers == in_coords
    delta_lsf = (edge_coordinates - in_coords)/lsf_in #lsf width normalized coord offsets
    #evaluate the integral at each delta
    indef_integs = lsf_cdf(delta_lsf)
    #convert the edge integral values to definite integrals accross the pixels
    definite_integrals = np.diff(indef_integs, axis=0)
    #plt.imshow(definite_integrals, interpolation="nearest")
    #plt.show()
    
    row_idxs = nearest_out_idx.reshape((1, -1)) + np.arange(-row_bandwidth, row_bandwidth+1).reshape((-1, 1))
    row_idxs = np.clip(row_idxs, 0, len(x_out)-1)
    col_idxs = np.arange(len(x_in)).reshape((1, -1))*np.ones((len(row_offsets)-1, 1))
    #flatten the coordinate and data arrays before passing in to coo_matrix
    row_idxs = row_idxs.reshape((-1,))
    col_idxs = col_idxs.reshape((-1,))
    definite_integrals = definite_integrals.reshape((-1,))
    
    transform = scipy.sparse.coo_matrix((definite_integrals, (row_idxs, col_idxs)), (len(x_out), len(x_in)))
    
    #normalize row or column sums to 1.
    if normalize == "rows":
        transform = transform.tocsr()
        row_sum = transform*np.ones(len(x_in))
        row_rescale = 1.0/np.where(row_sum > 0, row_sum, 1.0)
        normalizer = scipy.sparse.dia_matrix((row_rescale, 0), (len(x_out), len(x_out)))
        transform = normalizer*transform
    elif normalize == "columns":
        transform = transform.tocsc()
        col_sum = transform*np.ones(len(x_in))
        col_rescale = 1.0/np.where(col_sum > 0, col_sum, 1.0)
        normalizer = scipy.sparse.dia_matrix((col_rescale, 0), (len(x_in), len(x_in)))
        transform = transform*normalizer
    return transform


def resampling_matrix(
        x_in, 
        x_out, 
        lsf_in=0.0,
        lsf_out=1.0,
        lsf_cdf=None,
        lsf_cut=5.0,
        lsf_units="pixel",
        normalize="rows",
        min_rel_width=0.144
):
    """generate a resampling matrix R which when multiplied against a vector 
    y[x_in] will yield an estimate of y[x_out],  y[x_out] ~ R*y[x_in]. 
    Input vectors are assumed to already contain the effects of a line 
    spread function with width specified by lsf_in. 
    Therfore a differential lsf smoothing is applied with width 
    lsf_diff = sqrt(lsf_out**2 - lsf_in**2)
    This derives from the assumtpion that the line spread function has the 
    property that convolution of two line spread functions with widths a and b 
    results in a profile with the same shape and root mean square width 
    sqrt(a**2+b**2). This assumtion holds exactly for gaussian, lorentzian, 
    and voigt profiles and is a good approximtion for most profiles. 
    
    parameters
    
    x_in: ndarray or Coordinatization
      the coordinates at which input vectors will be sampled
    x_out: ndarray or Coordinatization
      the desired sampling coordinates of the output vectors
    lsf_in: ndarray or float
      lsf_width of input data (lsf <--> line spread function)
    lsf_out: ndarray or float
      the widths of the line spread function which is to be integrated.
    lsf_cdf: function
      the cumulative distribution function of the 
    lsf_cut: float
      the maximum number of principle differential lsf widths to calculate before
      letting the matrix entries drop to zero.
    lsf_units: string
      "pixel" lsf widths are specified in units of pixel widths
      "coordinate" lsf widths are specified in the same units as x_in and x_out
    min_rel_width: float
      The minimum differential lsf width to allow. This width is always specifed
      in terms of a fraction of the size of input pixels.
      Since only the differential smoothing between lsf_in and lsf_out is applied
      the effective lsf width to use when the difference between the output and 
      input smoothings is small (or negative) must be truncated at some small 
      but positive width. The default is 0.144 which is the rms width of a uniform
      function of half width 0.5.  
    """
    x_in = as_coordinatization(x_in)
    x_out = as_coordinatization(x_out)
    
    in_coords = x_in.coordinates
    out_coords = x_out.coordinates
    
    if lsf_cdf is None:
        lsf_cdf = scipy.stats.norm.cdf
    
    if not hasattr(lsf_in, "shape"):
        lsf_in = np.repeat(lsf_in, len(x_in))
    if not hasattr(lsf_out, "shape"):
        lsf_out = np.repeat(lsf_out, len(x_out))
    
    if not lsf_units in "pixel coordinate".split():
        raise ValueError("lsf_units option {} not recognized".format(lsf_units))
    
    dx_in = scipy.gradient(in_coords)
    if lsf_units == "pixel":
        lsf_in *= dx_in
        dx_out = scipy.gradient(out_coords)
        lsf_out *= dx_out
    
    out_to_in = x_out.interpolant_sampling_matrix(x_in.coordinates)
    iterped_lsf_out = out_to_in*lsf_out
    
    diff_lsf_sq = lsf_in**2 - iterped_lsf_out**2
    min_sq_widths = dx_in**2*min_rel_width**2
    lsf = np.sqrt(np.where(diff_lsf_sq > min_sq_widths, diff_lsf_sq, min_sq_widths))
    
    return pixel_integrated_lsf(x_in, x_out, lsf=lsf, lsf_cdf=lsf_cdf, lsf_cut=lsf_cut, normalize=normalize)

def get_transformed_covariances(transform_matrix, input_covariance, fill_variance = 0):
    #import pdb; pdb.set_trace()
    if len(input_covariance.shape) == 2:
        out_var = transform_matrix*input_covariance*transform_matrix.transpose()
    elif len(input_covariance.shape) == 1:
        ndat = len(input_covariance)
        ccov = scipy.sparse.dia_matrix((input_covariance, 0), (ndat, ndat))
        out_var = transform_matrix*ccov*transform_matrix.transpose()
    out_var = out_var.tolil()
    if fill_variance != 0:
        for i in range(transform_matrix.shape[0]):
            if out_var[i, i] == 0:
                out_var[i, i] = fill_variance
    out_var = out_var.tocsr()
    return out_var


def generate_wv_standard(min_wv, max_wv, npts, kind = "linear"):
    """if type == 'linear' wavelengths are equally spaced 
if type == 'log' the wavelengths will be equally spaced in log wavelengths which is equivalently a constant resolution """
    if kind == "log":
        log_wvs = np.linspace(np.log10(min_wv), np.log10(max_wv), npts)
        wvs = np.power(10.0, log_wvs)
    if kind == "linear":
        wvs = np.linspace(min_wv, max_wv, npts)
    return wvs

