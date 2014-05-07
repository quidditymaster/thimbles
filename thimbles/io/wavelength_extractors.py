import time
import warnings
import numpy as np
from .pixel_wavelength_functions import (NoSolution, LogLinear, Polynomial, 
                                        Linear , ChebyshevPolynomial,
                                        CubicSpline, LegendrePolynomial) 


# ########################################################################### #

__all__ = ['IncompatibleWavelengthSolution','from_w0','from_crval','from_crvl',
           'from_wcs','from_makee_wv','from_spectre']

# ########################################################################### #

class MulitpleWavelengthSolutions (Exception):
    """ Raised when multiple wavelength solutions were found """
    pass

class IncompatibleWavelengthSolution (Exception):
    """ Raised when the given wavelength solution can't be determined """
    pass

class NoWavelengthSolutionError (Exception):
    """ Raised when no wavelength solution could be found """
    pass

def pixels_to_phys_pixels (pixels,bzero=1,bscale=1):
    """
    convert points to wavelength in Angstroms via the wavelength solution
    
    Parameters
    ----------
    pixels : array
        contains the points used to convert element-by-element to wavelength
    
    bzero : float
        from the header file which gives the starting point for the physical pixels
    bscale : float
        from the header file which gives the scaling for the physical pixels
        pts = bzero + pts*bscale
    
    
    bzero = query_fits_header(header,'BZERO',noval=1) # for scaled integer data, here is the zero point
    bscale = query_fits_header(header,'BSCALE',noval=0) # for scaled integer data, here is the multiplier
    
    I'm pretty sure fits uses these when it reads in the data so I don't need to
   
    """
    if bzero != 0 or bscale !=1:
        raise ValueError(("I don't know exactly what to do with bzero!=1 "
                             "or bscale!=0 :<{}><{}>".format(bzero,bscale)))
    # should return [1,2,3,......,#pts]
    return  bzero + pixels*bscale    

pass
# ########################################################################### #

from_functions = {}

pass
# ===================== From W0 and WPC keywords

def from_pixels (header):
    """ Return a NoSolution wavelength solution for all possible orders
    
    If a NAXIS2 was given then there will be multiple orders. Each order
    is stacked against the previous order's last pixel.
        
    Parameters
    ----------
    header : `astropy.io.fits.header.Header`
     
    Returns
    -------
    wv_solns : list of `thimbles.NoSolution` objects             
        This list should be only one element long

    Raises
    ------
    IncompatibleWavelengthSolution : If missing keywords NAXIS1
     
    
        
    """
    try:
        naxis1 = header['NAXIS1']
        naxis2 = header.get('NAXIS2',1)
    except KeyError as e:
        raise IncompatibleWavelengthSolution(e.message)    
    wv_solns = []
    progressive = 1
    for _ in xrange(naxis2):
        # each order
        pixels = np.arange(naxis1)+progressive 
        wv_solns.append(NoSolution(pixels))        
        progressive = pixels[-1]+1
    return wv_solns
    
pass
# ===================== From W0 and WPC keywords

def from_w0 (header):
    """ Extracts coefficients from the W0 and WPC keywords
    
    Parameters
    ----------
    header : `astropy.io.fits.header.Header`
     
    Returns
    -------
    wv_solns : list of `thimbles.WavelengthSolution` objects             
        This list should be only one element long

    Raises
    ------
    IncompatibleWavelengthSolution : If missing keywords NAXIS1, W0 or WPC
    
    """
    # extract the keywords from the header
    try:
        naxis1 = header['NAXIS1']         
        w0 = header['W0'] 
        wpc = header['WPC']
    except KeyError as e:
        raise IncompatibleWavelengthSolution(e.message)
    warnings.warn("First time using W0 and WPC, check output of wavelength/flux")
    # create the pixels for naxis1
    pixels = np.arange(naxis1)+1
    # create and return
    c1 = wpc 
    c0 = w0     
    return [Linear(pixels,c1,c0)]    

from_functions['w0'] = from_w0

pass
# ===================== From CRVAL, CTYPE, etc keywords

def from_crval (header): 
    """
    Finds keywords CTYPE1, CRVAL1, CRPIX1, and CDELT1 and extracts the
    coefficients for the linear wavelength solution
    
    Parameters
    ----------
    header : `astropy.io.fits.header.Header`
     
    Returns
    -------
    wv_solns : list of `thimbles.WavelengthSolution` objects             
        This list should be only one element long

    Raises
    ------
    IncompatibleWavelengthSolution : If missing keywords NAXIS1 or CTYPE1 and if
        CTYPE1 is not 'LINEAR' or 'LOG-LINEAR'. Looks for CRVAL1, CRPIX1 and CDELT1
        and if not found it assumes 0, 1, and 1 respectivly 
                
    """ 
    # get necessary keywords
    try:
        naxis1 = header['NAXIS1'] 
        ctype1 = header["CTYPE1"]
        crval1 = header.get("CRVAL1",0)    
        crpix1 = header.get("CRPIX1",1)
        cdelt1 = header.get("CDELT1",1)
    except KeyError as e: 
        raise IncompatibleWavelengthSolution(e.message)
        
    start_pix = crpix1 + 1 # this is because I start the pixel counting at 1 later 
    pixels = np.arange(naxis1)+start_pix    
    c0 = crval1 
    c1 = crval1-start_pix*cdelt1
    # implement correct wavelength solution
    if ctype1.upper() == 'LINEAR':
        # TODO: other arguments?!
        return [Linear(pixels,c1,c0)]
    elif ctype1.upper() == "LOG-LINEAR":
        return [LogLinear(pixels,c1,c0)]
    else:
        raise IncompatibleWavelengthSolution("unknown value for keyword CTYPE={}".format(ctype1))

from_functions['crval'] = from_crval
pass
# ===================== From crvl, etc mutli-order keywords

def _order_from_crvl_ (header,order,pixels,linintrp):
    order = format(order,"02")
    crvl1_ = header.get('CRVL1_{}'.format(order)) # the starting wavelength
    cdlt1_ = header.get('CDLT1_{}'.format(order)) # the delta pixel change
    if crvl1_ is None or cdlt1_ is None:
        return 
    
    if linintrp == 'linear':
        c1 = cdlt1_
        c0 = crvl1_
        return Linear(pixels,c1,c0) 
    else:
        # NOTE : if you can fix this and make it better, do so 
        raise IncompatibleWavelengthSolution("Keyword LININTRP was missing or not 'linear', I don't know how to deal with this")    

def from_crvl (header):
    """ Looks at all the CRVL1_?? and CDLT1_?? keywords where ?? is all the possible orders
    (e.g. order 1 ==> CRVL1_01) 
    
    This method for storing wavelength solutions for many orders uses keywords
    CRVL1_?? and CDLT1_?? where the value for keyword NAXIS1 should tell you
    how many orders there are. The keyword LININTRP is used to specify what
    type of function. Only LININTRP implemented here is 'linear'.
    
    Parameters
    ----------
    header : `astropy.io.fits.header.Header`
     
    Returns
    -------
    wv_solns : list of `thimbles.WavelengthSolution` objects             
        This list should be only one element long        
    
    Raises
    ------
    IncompatibleWavelengthSolution : If missing keywords NAXIS1, NAXIS2
        or NAXIS1 of CRVL1_?? and CDLT1_?? keywords. Also if LININTRP is missing
        or non-linear
        
    """        
    # get keywords for this type of solution
    try:
        linintrp = header.get("LININTRP","").lower() # string with infor about the type of linear interpretation)
        naxis1 = header['NAXIS1']
        naxis2 = header['NAXIS2']
    except KeyError as e:
        raise IncompatibleWavelengthSolution(e.message)    
    # pixels from NAXIS1        
    pixels = np.arange(naxis1)+1
    
    # collect up wavelength solutions for each order
    # should be NAXIS2 long
    wv_solns = []
    for order in xrange(1,100):
        # extract solution for a specific order
        wv_soln = _order_from_crvl_(header, order, pixels, linintrp)
        # if no solution break
        if wv_soln is None:
            break         
        wv_solns.append(wv_soln)
        
    # check that length makes sense   
    n = len(wv_solns)
    if n == 0:
        # no values found
        raise IncompatibleWavelengthSolution("no orders found using CRVL1_? CDLT1_?")
    elif n != naxis2:
        # should have been length naxis2
        raise IncompatibleWavelengthSolution(("found wrong number of CRVL1_ "
                                              "(n={}) keywords for NAXIS2={}").format(n,naxis1))        
    return wv_solns

from_functions['crvl'] = from_crvl

pass
# ===================== From MAKEE WV keywords

def _order_from_makee_wv (header,order,base,pixels):    
    b1,b2 = base    
    b_0 = header.get("{0}{1:02}".format(b1,order))
    b_4 = header.get("{0}{1:02}".format(b2,order))    
    if b_0 is None or b_4 is None:
        return 
    
    func = lambda c: float(c or 0)       
    coeff = map(func,reversed(b_4.split()))
    coeff += map(func,reversed(b_0.split()))
    
    return Polynomial(pixels,coeff)

def from_makee_wv (header):
    """ Looks at all the WV_0_?? and WV)4_?? keywords where ?? is all the possible orders
    (e.g. order 1 ==> WV_0_01) 
    
    This method for storing wavelength solutions for many orders uses keywords
    WV_0_?? and WV_4_?? where the value for keyword NAXIS1 should tell you
    how many orders there are. 
    
    Parameters
    ----------
    header : `astropy.io.fits.header.Header`
     
    Returns
    -------
    wv_solns : list of `thimbles.WavelengthSolution` objects             
        This list should be only one element long        
    
    Raises
    ------
    IncompatibleWavelengthSolution : If missing keywords NAXIS1, NAXIS2
        or NAXIS1 of WV_0_?? and WV_4_?? keywords.
        
    """
    base = "WV_0_","WV_4_"
    # get necessary keywords
    try:
        naxis1 = header['NAXIS1']
        naxis2 = header['NAXIS2']
    except KeyError as e:
        raise IncompatibleWavelengthSolution(e.message)        
    # pixels from NAXIS1      
    pixels = np.arange(naxis1)+1
    
    # collect up wavelength solutions for each order
    # should be NAXIS2 long
    wv_solns = []
    for order in xrange(1,100):
        # extract solution for a specific order
        wv_soln = _order_from_makee_wv(header, order, base, pixels)
        # if no solution break
        if wv_soln is None:
            break         
        wv_solns.append(wv_soln)
        
    # check that length makes sense   
    n = len(wv_solns)
    if n == 0:
        # no values found
        raise IncompatibleWavelengthSolution("no orders found using WV_0_? WV_4_?")
    elif n != naxis2:
        # should have been length naxis2
        raise IncompatibleWavelengthSolution(("found wrong number of WV_0_?? "
                                              "(n={}) keywords for NAXIS2={}").format(n,naxis1))        
    return wv_solns

from_functions['makee_wv'] = from_makee_wv

pass
# ===================== From WCS

def _wcs_string_dispersion_format (pixels,spec_args_str):
    # - change the format of scientific notiation 1.0D+10 to 1.0E+10
    # - and make "+" signs implicit
    spec_args_str = spec_args_str.strip().replace('D','E').replace("+","")

    # split by spaces            
    spec_args = spec_args_str.split()

    # Create argument list
    args = []

    # [0] = aperture number
    args.append(int(spec_args[0]))
    
    # [1] = beam number
    args.append(int(spec_args[1]))
    
    # [2] = dispersion type, dcflag
    args.append(int(spec_args[2]))
    
    # [3] = c0, first physical pixel
    args.append(float(spec_args[3]))
    
    # [4] = c1, average disperasion interval
    args.append(float(spec_args[4]))
    
    # [5] = npts, number valid pixels !! could have problem if different orders have different lengths
    args.append(int(spec_args[5]))
    
    # [6] = rv,z, applies to all dispersions coordinates by multiplying 1/(1+z)
    args.append(float(spec_args[6]))
    
    # [7] = aplow, lower limit of aperture
    args.append(float(spec_args[7]))
    
    # [8] = aphigh, upper limit of aperture
    args.append(float(spec_args[8]))
        
    # OPTIONAL:    
    # function_i =  wt_i w0_i ftype_i [parameters] [coefficients]    
    n = len(spec_args)
    # [9]  = wieght wt_i 
    if n > 9:
        args.append(spec_args[9])
        
    # [10] = zeropoint offset w0_i
    if n > 10:
        args.append(spec_args[10])
        
    # [11] = type dispersion fxn, 1=cheby, 2=legrandre, 
    #         3=cubic spline3, 4=linear spline, 
    #         5=pixel coordinate array, 6=sampled coordinate array
    if n > 11:
        args.append(int(spec_args[11]))
               
    # [12+] = [parameters...]    
    # [12++] = [coefficients...]
    if n > 12:
        args += map(float,spec_args[12:])
    
    return _wcs_dispersion_format(pixels,*args) 

def _wcs_dispersion_format (pixels,*args):
    """
    
    REFERNCE: http://iraf.net/irafdocs/specwcs.php
    
    The dispersion functions are specified by attribute strings with the identifier specN where N is the physical image line. The attribute strings contain a series of numeric fields. The fields are indicated symbolically as follows. 
     
    specN = ap beam dtype w1 dw nw z aplow aphigh [functions_i]
    spec1 = "  1  1 0  4719.01928834982 0.020571536306138  4055 0.   1.0   1.0"
    
    Parameters
    ----------
    pixels : ndarray
    aperture_number : integer
    beam_number : interger
    dispersion_type : integer
        dcflag = -1 no disp, 0 linear, 1 log-linear, 2 nonlinear
    c0 : float
        first physical pixel
    c1 : float
        average dispersion interval
    npts : integer
        number of valid pixels
    z : float
        redshift
    aplow : float
        lower limit of aperture
    aphigh : float
        upper limit of aperture
    weight : integer
    zero_point : interger
    function_type : integer
        type dispersion fxn = 1 cheby, 2 legrandre, 3 cubic spline3, 
        4 linear spline, 5 pixel coordinate array, 6 sampled coordinate array        
    *parameters :
    *coefficients :
            
    NOTES
    -----        

    REFERNCE: http://iraf.net/irafdocs/specwcs.php
    The dispersion functions are specified by attribute strings with the 
    identifier specN where N is the physical image line. The attribute 
    strings contain a series of numeric fields. The fields are indicated 
    symbolically as follows. 
        specN = ap beam dtype w1 dw nw z aplow aphigh [functions_i]
    example :
        spec1 = "1  1 0  4719.01928 0.02057153  4055 0.   1.0   1.0"  
                      
        [0] = aperture number
        [1] = beam number
        [2] = dispersion type, 
        [3] = c0, first physical pixel
        [4] = c1, average disperasion interval
        [5] = npts, number valid pixels !! could have problem if different orders have different lengths
        [6] = rv,z, applies to all dispersions coordinates by multiplying 1/(1+z)
        [7] = aplow, lower limit of aperture
        [8] = aphigh, upper limit of aperture
        
        
        # ---------------- optional
        function_i =  wt_i w0_i ftype_i [parameters] [coefficients]
        [9]  = wieght wt_i 
        [10] = zeropoint offset w0_i
        [11] = type dispersion fxn, 1=cheby, 2=legrandre, 
                3=cubic spline3, 4=linear spline, 
                5=pixel coordinate array, 6=sampled coordinate array
        [12+] = [parameters...]
        [12++] = [coefficients...]

        
    
    """
    # ===================== read in arguments
    
    aperture_number = args[0]
    beam_number = args[1]
    dcflag = args[2]
    c0 = args[3]
    c1 = args[4]
    npts = args[5]
    z = args[6]
    aplow = args[7]
    aphigh = args[8]
        
    # ===================== use the variables to get wavelength solution
    
    pixels = np.arange(npts)+1
    
    # the redshift
    rv = 1.0/(1.0+z)

    # dispersion type flag        
    if dcflag == -1:
        return NoSolution(pixels,rv=rv)
    elif dcflag == 0:
        return Linear(pixels,c1,c0,rv=rv)
    elif dcflag == 1:        
        return LogLinear(pixels,c1,c0,rv=rv) 
    elif dcflag != 2:
        raise IncompatibleWavelengthSolution("Unknown value for WCS dcflag={}".format(dcflag))
    # dcflag = 2 # non-linear
    
    wt_i = args[9]
    zeropoint = args[10]
    func_type = args[11]
    par_coef = args[12:]

    if func_type == 1:
        # chebyshev polynomial
        msg = "WCS chebyshev specification is incorrect"
        if len(args) < 12:         
            raise IncompatibleWavelengthSolution(msg)        
        order = args[12]
        if len(args) < 15+order:
            raise IncompatibleWavelengthSolution(msg)
        xxmin = args[13]
        xxmax = args[14]
        coefficients = args[15:15+order]
        raise NotImplementedError("Chebyshev polynomial wavelength solution from WCS")
    elif func_type == 2:
        # legendre polyomial 
        raise NotImplementedError("Legendre polynomial wavelength solution from WCS")
    elif func_type == 3:
        # cubic spline
        raise NotImplementedError("Cubic spline wavelength solution from WCS")
    elif func_type == 4:
        # linear spline
        raise NotImplementedError("Linear spline wavelength solution from WCS")
    elif func_type == 5:
        # pixel coordinate array
        raise NotImplementedError("pixel coordinate array wavelength solution from WCS")
    elif func_type == 6:
        # sampled coordinate array
        raise NotImplementedError("sample coordinate array wavelength solution from WCS")
    else: 
        raise IncompatibleWavelengthSolution("Unknown value for WCS polynomial type={}".format(func_type))
  
def _multi_line_to_single (header,base="WAT2_"):
    """
    WAT2_001= 'wtype=multispec spec1 = "  1  1 0  4719.01928834982 0.02057153630613'
    WAT2_002= '8  4055 0.   1.0   1.0" spec2 = "  2  2 0  4782.77958543811 0.020850'
    WAT2_003= '979350395  4055 0.   2.0   2.0"                                     '    
     ...
       
    take these and return a single string
    
    ('wtype=multispec spec1 = "  1  1 0  4719.01928834982 0.02057153630613'
     '8  4055 0.   1.0   1.0" spec2 = "  2  2 0  4782.77958543811 0.020850'
     '979350395  4055 0.   2.0   2.0"')
              
    """
    # a variable to concatenate into
    wat_string = ""    
    # if the header is a dictionary not a `astropy.io.fits.header.Header`    
    if isinstance(header,dict):
        for i in xrange(1,101):
            keyword = base+format(i,"03")
            if header.has_key(keyword):
                wat_string += header[keyword]
            else:
                break 
        return wat_string
        
    # NOTE: I needed it to be the raw string because using header as a
    # dictionary caused problems. Mainly, the values of the dictionary cut some
    # white space from the end of the strings which squished together some 
    # important information
    hdrstr = header.tostring()    
    for i in xrange(1,101):
        keyword = base+format(i,"03")
        # check number of times this appears
        num = hdrstr.count(keyword)
        if num == 0:
            break 
        elif num > 1:
            raise KeyError("Found more than one keyword in {}".format(keyword))        
        i = hdrstr.find(keyword)
        # hdrstr[i+11:i+79] ==  re.search("WAT2_003.*=.*\'(.*)\'",hdrstr[i:i+80]).groups()[0]        
        wat_string += hdrstr[i+11:i+79]
    if not len(wat_string):
        raise IncompatibleWavelengthSolution("Missing any keywords in form {0}???".format(base)) 
    return wat_string    
       
def _wcs_arguments_to_dict (value,lower=False):    
    """ Takes a string line and converts to a dictionary based on '=' signs
    
    
    'key1 = "a b c " key2=world key3= " 1  10 2.3  "' => 
    
    dict(key1="a b c",key2="world",key3="1  10 2.3  ")
    
        
    """    
    equals = [-1]
    equals += [i for i in xrange(len(value)) if value[i] == "="]
    if len(equals) == 1:
        raise ValueError("problem parsing wcs")        
    if lower:
        value = value.lower()
    parsed = {}    
    for i in xrange(1,len(equals)-1):        
        # (=) key1 = a b c key2 = b
        #  0       7            20 
        j_prev = equals[i-1]
        j_cur = equals[i]
        j_next = equals[i+1]
        
        # key is the item prior to the equals 
        
        prev = value[j_prev+1:j_cur]
        key = prev.split()[-1]
        
        # the value is everything but the next key
        next_ = value[j_cur+1:j_next]
        next_key = next_.split()[-1]
        val = value[j_cur+1:j_next-len(next_key)-1]
    
        # store the result
        parsed[key] = val.strip().replace('"',"")
        
    # deal with the very last one          
    key = value[equals[-2]+1:equals[-1]].split()[-1]
    val = value[equals[-1]+1:]
    parsed[key] = val.strip().replace('"',"")
        
    return parsed

def _parse_wcs_keywords (header,base="WAT2_",lower=False):
    value = _multi_line_to_single(header, base)
    return _wcs_arguments_to_dict(value,lower)

def from_wcs (header):
    """ Extracts wavelength solution from the World Coordinate System standard 
      
    Parameters
    ----------
    header : `astropy.io.fits.header.Header`
     
    Returns
    -------
    wv_solns : list of `thimbles.WavelengthSolution` objects             
        This list should be only one element long        
    
    Raises
    ------
    IncompatibleWavelengthSolution : If missing expected keywords and values   
    NotImplementedError : If the wavelength solution is stored in an equation 
        type which hasn't been implemented yet
        
    """    
    try: 
        naxis1 = header['naxis1']
        naxis2 = header['naxis2']
    except KeyError as e:
        raise IncompatibleWavelengthSolution(e.message)
    
    # check the format of the wavelength solution    
    wat0 = _parse_wcs_keywords(header,'WAT0_',lower=True) # WAT0_001 
    if not wat0.has_key('system'):
        raise IncompatibleWavelengthSolution("Expected WAT0_001 to have keyword `system`")    
    
    if wat0['system'] not in ("physical","multispec"):
        raise IncompatibleWavelengthSolution("Expected WAT0_001 keyword system to be 'physical' or 'multispec'")
    
    wat1 = _parse_wcs_keywords(header,"WAT1_",lower=True) # WAT1_001
    # check that these keys match values
    require = [('wtype','multispec'),
                ('label','wavelength')]
    for key,value in require:
        if wat1[key] != value:        
            raise IncompatibleWavelengthSolution(("Expected WAT1_001 keyword "
                                                  "{}={} not {}".format(key,value,wat1[key])))
    wv_unit = wat1.get('unit','angstroms')
    
    # now extract all the information
    wat2 = _parse_wcs_keywords(header,"WAT2_",lower=True) # WAT2_???
    
    pixels = np.arange(naxis1)+1
    wv_solns = []
    for order in xrange(1,naxis2+1):
        key = "spec{}".format(order)
        spec_args_str = wat2.get(key)
        if spec_args_str is None:
            raise IncompatibleWavelengthSolution("expected wavelength solution for order {}".format(order))
        wv_solns.append(_wcs_string_dispersion_format(pixels, spec_args_str))
    
    return wv_solns

from_functions['wcs'] = from_wcs

pass
# ===================== From SPECTRE HISTORY keywords

def _parse_spectre_timetag (hist_line,sec=0):
    """ parse the SPECTRE time tag 
    
    Parameters
    ----------
    hist_line : string
        "HISTORY 23: 7:2008 D1,2,3: 5.50348825884E+03 4.46070136915E-02 0.00000000000E+00"
    
    Returns
    -------
    time : float
        the date 7/23/2008 + sec in seconds      
    """
    date_str = hist_line[:10]
    day,month,year = [int(s) for s in date_str.split(":")]
    timetag = time.mktime((year,month,day,0,0,sec,0,0,0))
    return timetag

def _parse_spectre_coefficients (hist_line):
    """ parse the coefficients from the spectre line
    
    Parameters
    ----------
    hist_line : string
        "HISTORY 23: 7:2008 D1,2,3: 5.50348825884E+03 4.46070136915E-02 0.00000000000E+00"
    
    Returns
    -------
    coefficients : list
        [5.50348825884e+03, 4.46070136915e-02, 0.0e+00]
        
    """
    coeffs = hist_line[18:36],hist_line[36:54],hist_line[54:]
    coeffs = [float(c.replace("D","e")) for c in coeffs]
    return coeffs

def _parse_spectre_history (histories):   
    """ Parses the SPECTRE HISTORY tags
    
    Parameters
    ----------
    histories : list of strings
    
    HISTORY 23: 7:2008 D1,2,3: 5.50348825884E+03 4.46070136915E-02 0.00000000000E+00
    HISTORY 23: 7:2008 D4,5,6: 0.00000000000E+00 0.00000000000E+00 0.00000000000E+00
    HISTORY 23: 7:2008 D7,8,9: 0.00000000000E+00 0.00000000000E+00 0.00000000000E+00
    HISTORY 24: 7:2008 D1,2,3: 5.49828526104E+03 4.46070136915E-02 0.00000000000E+00
    HISTORY 24: 7:2008 D4,5,6: 0.00000000000E+00 0.00000000000E+00 0.00000000000E+00
    HISTORY 24: 7:2008 D7,8,9: 0.00000000000E+00 0.00000000000E+00 0.00000000000E+00
    
    Returns
    -------
    spectre_history : dict
        The keys are the time stamps (a second is added if the tag appears later)
        the values are tuples function_type and coefficients
        
    """ 
    # get the line "D/d,/d,/d"
    get_spectre_d = lambda x: x[11:17]
    # object to store all of the history tags
    spectre_history = {}
    for i in xrange(len(histories)-2):
        hist_line = histories[i]    
        # the first tag you hit is "D1,2,3"
        ds1 = get_spectre_d(hist_line)
        if ds1 != "D1,2,3":
            continue     
        # get the next two lines as well
        hl1 = hist_line
        hl2 = histories[i+1]
        hl3 = histories[i+2]    
        # check tags
        ds2 = get_spectre_d(hl2)
        ds3 = get_spectre_d(hl3)
        if ds2 != "D4,5,6" or ds3 != "D7,8,9":
            warnings.warn("Expected next two history lines "
                          "to have D4,5,6 and D7,8,9")
            # TODO: should this be an error?
            continue        
        sec = len(histories)-i # work backwards in time
        # check time stamp
        tt1 = _parse_spectre_timetag(hl1,sec=sec) 
        tt2 = _parse_spectre_timetag(hl2,sec=sec)
        tt3 = _parse_spectre_timetag(hl3,sec=sec)
        if not (tt1==tt2 and tt2==tt3):
            # time for these tags must be the same
            continue     
        # get the coefficients off the line
        c_1 = _parse_spectre_coefficients(hl1) # c0,c1,c2        
        c_2 = _parse_spectre_coefficients(hl2) # c3,c4,c5
        c_3 = _parse_spectre_coefficients(hl3) # c6,c7,c8
        
        # combine the coefficients together
        coeff = np.asarray(c_1+c_2+c_3[:-1])
        disp_info = c_3[-1] 

        # figure out what dispersion type
        # check out SPECTRE/Wave.f for source
        if disp_info == 1: 
            coeff = coeff[:5]
            disp_type = 'chebyshev poly'
        elif disp_info == 2: 
            coeff = coeff[:2]
            disp_type = 'legendre poly'
        elif disp_info == 3:
            disp_type = 'spline3'
        elif np.count_nonzero(coeff) == 2:
            coeff = coeff[:2]
            disp_type = 'linear'
        elif coeff[0] == 0 or coeff[1] == 0:
            disp_type = 'no solution'
        else: 
            disp_type = 'poly' 
        # store the history tag
        spectre_history[tt1] = (disp_type,coeff) 
    return spectre_history
    
def from_spectre (header):
    """ Extract wavelengths from SPECTRE HISTORY keywords
    
    The tool SPECTRE stores the wavelength solution using the keywords HISTORY
    along with it's own formatting. 
    
    TODO: example
    
    Parameters 
    ----------
    header : dict or `astropy.io.fits.header.Header`
    
    Returns
    -------
    wavelength_soln_list : list of `thimbles.spectrum.WavelengthSolution`
        Callable to take pixels to wavelengths        
    
    Raises
    ------
    IncompatibleWavelengthSolution : no SPECTRE wavelength solution identified
    
    """
    # check for necessary keywords
    try: 
        naxis1 = header['NAXIS1']
        # get all history tags from the header
        histories = header['HISTORY']        
    except KeyError as e:
        raise IncompatibleWavelengthSolution(e.message)    
    # parse all the history tags
    spectre_history = _parse_spectre_history(histories)
    # is the parsed history appropriate?
    if len(spectre_history) == 0:
        raise IncompatibleWavelengthSolution("Couldn't parse any of the HISTORY lines as SPECTRE tags")
    pixels = np.arange(naxis1)+1
    # get the history tag with greatest time
    tt = np.max(spectre_history.keys())
    disp_type,coeffs = spectre_history[tt]
    coeffs = list(reversed(coeffs))  
    
    import pdb; pdb.set_trace()
    # take values and return them
    if disp_type == 'no solution':
        return [NoSolution(pixels)]  
    elif disp_type == 'chebyshev poly':
        return [ChebyshevPolynomial(pixels,coeffs)]
    elif disp_type == 'legendre poly':   
        return [LegendrePolynomial(pixels,coeffs)]
    elif disp_type == 'spline3':
        warnings.warn("spectre spline3 wavelength solution is unvetted, check output")        
        # from SPECTRE: s = (point-1.)/(real(npt)-1.)*c(8)
        # c(8) == disp_info[1] ==> true # may need to extract this   
        return [CubicSpline(pixels,coeffs)]
    elif disp_type == 'linear':
        c1 = coeffs[-2]
        c0 = coeffs[-1]
        return [Linear(pixels,c1,c0)]
    else:
        return [Polynomial(pixels,coeffs)]    

from_functions['spectre'] = from_spectre

pass
# ===================== 

def from_header (header,preference=None):
    """ Get the wavelength solution 
    
    This will check many different types of wavelength solution functions and
    give you the one which works for the given header. If you give a preference
    then it will use that specific function. If no solutions are found then it
    will return NoSolution wavelength solution objects.
        
    Parameters 
    ----------
    header : dict or `astropy.io.fits.header.Header`
    preference : string in {0}
        Will use the specific function
    
    Returns
    -------
    wavelength_soln_list : list of `thimbles.spectrum.WavelengthSolution`
        Callable to take pixels to wavelengths        
    
    Raises
    ------
    IncompatibleWavelengthSolution : If preference is given then it will raise
        if the preference is incompatible.
    MulitpleWavelengthSolutions : If more than one wavelenght solution is found
    
    """
    # specify a preference
    if preference is not None:
        func = from_functions.get(preference)
        if func is None:
            raise ValueError("preference needs to be in {}".format(from_functions))
        return func(header)
        
    # get all the wavelength solutions
    wv_solutions = []
    compatible_solutions = []
    for func_name,func in from_functions.iteritems():
        # try to get the solution
        try: 
            wv_solutions.append(func(header))
            compatible_solutions.append(func_name)
        except IncompatibleWavelengthSolution:
            pass
    # check how many solutions were found
    n = len(wv_solutions)
    if n == 0:
        return from_pixels(header)        
    elif n > 1:        
        raise MulitpleWavelengthSolutions("solutions found for methods {}".format(compatible_solutions))
    return wv_solutions[0]
