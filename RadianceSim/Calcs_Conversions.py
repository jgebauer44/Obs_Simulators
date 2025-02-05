import numpy as np


################################################################################
# This file contains the following funtions:
# planck()
# planck_microwave()
# invplanck()
# inv_planck_microwave()
# t2density()
# rh2w()
# rh2dpt()
# epres()
# esat()
# w2e()
# w2rh()
# w2rho()
# wsat()
# w2pwv()
# tq2wvdens()
# inv_hypsometric()
# t2theta()
# t2thetae()
# theta2t()
# tvirt()
#################################################################################

################################################################################
# This function computes Planck blackbody radiance 
################################################################################

def planck(wn,temp):
    
    #From the LBLRTM v6.01. These radiation constants, according to inline
    #LBLRTM documentation, are taken from NIST
    
    c1 = 1.191042722e-12
    c2 = 1.4387752          # units are [K cm]
    c1 = c1 * 1e7           # units are now [mW/m2/ster/cm-4]
    rad = c1 * wn * wn *wn / (np.exp(c2 * wn / temp)-1.0)
    
    return rad
    
################################################################################
# This is planck function designed for microwave wavelenghts
################################################################################

def planck_microwave(v,T):

    h = 6.626070040e-34       # J s
    c = 2.99792458e8          # m / s
    k = 1.38064852e-23        # J / K
    vv = v * 1e9      # Convert frequency from GHz to Hz
    lambda_si = c / vv
    I = (2.*h*vv/(lambda_si**2))*(1./(np.exp(h*vv/(k*T))-1))
    
    return I
    
################################################################################
# This function computes the inverse Planck's function (it gives blackbody
# temperature)
################################################################################
    
def invplanck(wn,rad):
    
    c1 = 1.191042722e-12
    c2 = 1.4387752          # units are [K cm
    c1 = c1 * 1e7           # units are now [mW/m2/ster/cm-4]
    temp = c2 * wn / (np.log(1.0 + (c1 * ((wn)**3.0) / rad)))
    
    return temp
    
################################################################################
# This is the inverse planck function designed for microwave wavelengths
################################################################################

def inv_planck_microwave(v, I):
    h = 6.626070040e-34       # J s
    c = 2.99792458e8          # m / s
    k = 1.38064852e-23        # J / K
    vv = v * 1e9      # Convert frequency from GHz to Hz
    lambda_si = c / vv
    T = (h*vv/k)*1/np.log((2*h*vv/(I*(lambda_si**2.)))+1.)
    
    return T
    
################################################################################
# This rountine calculates the density (molecules/m^3) from the profiles
# of temperature (degC) and pressure (mb). Logic by Rich Ferrare, NASA/LaRC.
# It also provides the density of the (dry) air in kg/m3 as a keyword option
################################################################################

def t2density(t, p, dens = False):
    
    density = (p * 100.0) / (1.381e-23 * (t + 273.15))
    foo = np.where(p < 0)[0]
    if len(foo) > 0:
        density[foo] = -999.
    
    Rv = 287.05              # Gas constant for dry air [J / (kg k)]
    dens_other = (p * 100.) / (Rv * (t+273.15) )
    
    if dens:
        return density, dens_other
    else:
        return density
    
################################################################################
# This function computes the mixing ratio of water from the relative humidity
################################################################################

def rh2w(temp, rh, pres):
    e = epres(temp, rh)        # Get the vapor pressure
    e = e /100.                # Convert the vapor pressure to mb (same as pres)
    w = 0.622 * e / (pres - e) # this ratio is in g/g
    w = w * 1000               # convert to g/kg
    
    return w
    
################################################################################
# This funciton computes the dew point given temperature and relative humidity
################################################################################

def rh2dpt(temp,rh):
    es0 = 611.0
    gascon = 461.5
    trplpt = 273.16
    tzero = 273.15
    
    yo = np.where(np.array(temp) == 0)[0]
    if len(yo) > 0:
        return np.zeros(len(temp))

    latent = 2.5e6 - 2.386e3*temp
    dpt = np.copy(temp)
    
    for i in range(2):
        latdew = 2.5e6 - 2.386e3*dpt
        dpt = 1.0 / ((latent/latdew) * (1.0 / (temp + tzero) - gascon/latent * np.log(rh)) + 1.0 / trplpt * (1.0 - (latent/latdew))) - tzero
    
    return dpt
    
    
################################################################################
# This function computes the vapor pressure give the temperature and relative
# humidity
################################################################################

def epres(temp, rh):
    ep = rh * esat(temp,0)
    return ep
    
################################################################################
# This function computes the saturation vapor pressure over liquid water or ice
################################################################################

def esat(temp,ice):
    es0 = 611.0
    gascon = 461.5
    trplpt = 273.16
    tzero = 273.15
    
    # Compute saturation vapor pressure (es, in mb) over water or ice at temperature
    # temp (in Kelvin using te Goff-Gratch formulation (List, 1963)
    #print(type(temp))
    if ((type(temp) != np.ndarray) & (type(temp) != list) & (type(temp) != np.ma.MaskedArray)):
        temp = np.asarray([temp])

    if type(temp) == list:
        temp = np.asarray(temp)

    tk = temp + tzero
    es = np.zeros(len(temp))
  
    if ice == 0:
        wdx = np.arange(len(temp))
        nw = len(temp)
        nice = 0
    else:
        icedx = np.where(tk <= 273.16)[0]
        wdx = np.where(tk > 267.16)[0]
        nw = len(wdx)
        nice = len(icedx)
    
    if nw > 0:
        y = 373.16/tk[wdx]
        es[wdx] = (-7.90298 * (y - 1.0) + 5.02808 * np.log10(y) -
            1.3816e-7 * (10**(11.344 * (1.0 - (1.0/y))) - 1.0) +
            8.1328e-3 * (10**(-3.49149 * (y - 1.0)) - 1.0) + np.log10(1013.246))
            
    if nice > 0:
        # for ice
        y = 273.16/tk[icedx]
        es[icedx] = (-9.09718 * (y - 1.0) - 3.56654 * np.log10(y) +
                    0.876793 * (1.0 - (1.0/y)) + np.log10(6.1071))
    
    es = 10.0**es
    
    # convert from millibar (mb) to Pa
    es = es * 100
    return es
    
################################################################################
# This function converts water vapor mixing ratio to the partial pressure
# of water vapor
################################################################################

def w2e(w,p):
    ww = w/1000.
    e = p * ww / (0.622 + ww)
    return e
    
################################################################################
# This function calculates relative humidity given the mixing ratio, the
# barametric pressure, and temperature.
################################################################################

def w2rh(w, p, t, ice=0):
    
    e = w2e(w, p)
    es = esat(t, ice) / 100.0          # To convert it to the correct units (mb)
    rh = e / es
    
    return rh
    
################################################################################
# This function computes water vapor density [g/m3], given the mixing ratio [g/kg]
# pressure [mb], and temperature [degC]
################################################################################

def w2rho(w, t, p):

    rho = w * p * 0.3477 / (t + 273.16)

    return rho

################################################################################
# This function computes the saturation water vapor mixing ratio given the 
# temperature and pressure. Note that for pressures less than 50 mb, wsat
# will be set to zero.
################################################################################

def wsat(temp, pres):
    
    eps = 0.621970585
    
    t = temp
    p = pres
    
    es = esat(t,0)
    es = es / 100.0         # convert from Pa to hPa (mb)
    w = 1000.0 * eps * es/ (p-es)
    foo = np.where(p < 50)[0]
    
    if len(foo) > 0:
        w[foo] = 0
    
    return w
    
################################################################################
# This routine calculates total precipitable water from mixing ratio and 
# pressure profiles
################################################################################

def w2pwv(w, p, pwv_err_flag = False, w_err = None, err_thres = 25, max_ht_index = None,
        pwv_max_ht_flag = False, min_w = -5, min_w_err = 0, pwv_profile_flag = False):
        
    yo = np.where(np.array(w) == 0)[0]
    if len(yo) > 0:
        return 0.
 
    if w_err is None:
        err = w * 0
    else:
        err = w_err
    
    if max_ht_index is None:
        max_ht_index = len(p)
    
    w1 = np.copy(w)
    p1 = np.copy(p)
    err1 = err
    index = np.arange(len(p))
    error = np.abs(100.0 * err / w)
    foo = np.where(((np.isfinite(error)) & (err >= min_w_err) & (error > err_thres) & (p < 750)) | (index > max_ht_index))[0]
    if len(foo) > 0:
        w1[foo[0]:len(w1)] = -999.
        p1[foo[0]:len(w1)] = -999.
        err1[foo[0]:len(w1)] = -999.
        pwv_max_ht_index = index[foo[0]]
    else:
        pwv_max_ht_index = max_ht_index
        
    # Don't allow any bad error values (which are probably missing values
    # due to the merging of different RL channels) whack us out
    
    if len(foo) > 0:
        first_bad_point = foo[0]
    else:
        first_bad_point = len(err1)
    
    foo = np.where((err1 < min_w_err) & (index <= first_bad_point))[0]
    if len(foo) > 0:
        err1[foo] = min_w_err
    
    # Look for any values where the mixing ratio is bad or missing and delete
    # these. By deleting the same levels in the error profile, we will help
    # ensure that the error is representative
    foo = np.where(w1 < min_w)[0]
    if len(foo) > 0:
        w1[foo] = -999
        err1[foo] = -999
        p1[foo] = -999
    
    # Equation used
    # PWV = 1/g * integral(q dp)
    mp = p1 / 10.0           # Convert pressure from mb (hPa) to kPa
    r = w1 / 1000.0          # Convert from g/kg to g/g
    re = err1 / 1000.0       # Convert the error from g/kg to g/g
    q = r / (1.0 + r)        # Convert mixing ratio to specific humidity
    qe = re / (1.0 + re)     # Convert the error into specific humidity
    
    pwv_profile = np.ones(len(w1)) * -999
    
    foo = np.where(w1 > -800)[0]
    bar = np.where((err1 > -800) & (w1 > -800))[0]
    if len(foo) == 0:
        pwv = -999.
        pwv_err = -999.
    else:
        
        # Calculate the PWV
        pwv = 0
        for i in range(1, len(foo)):
            pwv = pwv + ((q[foo[i]] + q[foo[i-1]])/2.0) * (mp[foo[i-1]] - mp[foo[i]])
            pwv_profile[foo[i]] = pwv
        
        # Calculate the PWV error
        if len(bar) == 0:
            pwv_err = -999.
        else:
            if bar[0] == 0:
                pwv_err = 0
            else:
                
                # If the first valid error value is not the first point
                # in the error profile, extend the first valid value
                # down to the surface by assuming the same error at the
                # surface as there is at the first valid point
                
                pwv_err = 2*(qe[bar[0]]**2) * (((mp[0] - mp[bar[0]])/2.0)**2)
            
            # Compute the rest of the error
            for i in range(len(bar)):
                pwv_err = pwv_err + (qe[bar[i]]**2 + qe[bar[i-1]]**2) * (((mp[bar[i-1]] - mp[bar[i]])/2.0)**2)
    
    # Account for the gravitational constant and convert
    # to the right units
    
    pwv = pwv/9.8          # Correct for G
    pwv = pwv * 100.0      # Convert from m to cm
    
    foo = np.where(pwv_profile <= 0)[0]
    pwv_profile = pwv_profile / 9.8           # Correct for G
    pwv_profile = pwv_profile / 100.0         # Convert from m to cm
    if len(foo) > 0:
        pwv_profile[foo] = -999
    
    pwv_err = pwv_err / (9.8**2)         # Correct for G
    pwv_err = pwv_err * (100.0)**2       # Convert from m to cm
    
    pwv_err = np.sqrt(pwv_err)
    
    if ((pwv_profile_flag) & (pwv_err_flag) & (pwv_max_ht_flag)):
        return pwv, pwv_err, pwv_max_ht_index, pwv_profile
    elif ((pwv_err_flag) & (pwv_max_ht_flag)):
        return pwv, pwv_err, pwv_max_ht_flag
    elif ((pwv_profile_flag) & (pwv_max_ht_flag)):
        return pwv, pwv_max_ht_flag, pwv_profile
    elif ((pwv_profile_flag) & (pwv_err_flag)):
        return pwv, pwv_err, pwv_profile
    elif pwv_profile_flag:
        return pwv, pwv_profile
    elif pwv_err_flag:
        return pwv, pwv_err
    else:
        return pwv
    
################################################################################
# This function computes water vapor density [mol/cm3] from temperature [C],
# water vapor mixing ratio [g/kg], and pressure [mb]
################################################################################

def tq2wvdens(t, q, p):
    
    # Compute the dry air density from the temperature and pressure
    atdens = t2density(t,p)             # mol / m3
    
    # Use the dry air density [mol / m3] to convert from WVMR [g/k]
    # to WV density [mol / cm3]
    
    # The 0.622 is the ratio of the weight of water to dry air
    # The factor of 1000 is to convert from g/g to g/kg
    # The conversion factor 1e6 is to convert mol/m3 to mol/cm3
    
    wvdens = (q / (0.622*1000)) *(atdens / 1.0e6)
    
    return wvdens
    
################################################################################
# This function is the inverse of the hypsometric function; i.e., it takes
# two heights and a reference pressure and returns the other pressure value.
################################################################################

def inv_hypsometric(z, t, p0):
    
    # Quick QC
    
    foo = np.where(t <= 0)[0]
    if len(foo) > 0:
        print('ERROR in inv_hypsometric: Temperature is assumed to be Kelvin -- abort')
        return -999
    
    a = 29.2911          # Gas constant in dry air divided by the acceleration due to gravity
    
    temp = np.copy(t)
    p = np.zeros(len(z))
    p[0] = p0*100.0            # Convert from mb (hPa) to Pa
    zz = z * 1000.             # Convert from Pa to hPa (mb)
    
    for i in range(1,len(zz)):
        p[i] = p[i-1] / np.exp(2*(zz[i]-zz[i-1]) / (a * (temp[i]+temp[i-1])))
    
    p = p/100.
    return p
    
################################################################################
# This function computes potential temperature from an input temperature
# water vapor mixing ratio profile. This code assumes that the first
# level is surface level (i.e., has highest pressure)
################################################################################

def t2theta(t, w, p, p0 = 1000.):
    yo = np.where(np.array(p) == 0)[0]
    if len(yo) > 0:
        theta = np.zeros(len(t))
    else:
        theta = (t+273.16) * (p0/p)**0.286
        theta = theta * (1 + 0.61 * (w/1000.))
    return theta
################################################################################
# This function computes equivalent potential temperature from an input temperature
# water vapor mixing ratio profile. The code assumes that the first level is the
# surface level (i.e., has highest pressure)
################################################################################

def t2thetae(t, w, p, p0 = 1000.):
    theta = t2theta(t,w*0, p)                 # Get the potential temperature [K]
    
    L = 2.5e6              # Latent heat of vaporization [J kg-1]
    Cp = 1005.             # Specific heat capacity [J Kg-1]
    
    thetae = theta * (1 + (L*w/1000.) / (Cp*(t+273.16)))
    return thetae

################################################################################
# This function computes ambient temperature from the input potential temperature
# and water vapor mixing ratio profile. The code assumes that the first level is
# the surface level (i.e., has hightest pressure)
################################################################################

def theta2t(theta, w, p, p0 = 1000.):
    th = theta/ ( 1 + 0.61 * (w/1000.))
    t = th / ( (p0/p)**0.286 ) - 273.16
    
    return t

################################################################################
# This function computes the virtual temperature given the ambient temperature,
# the relative humidty, and the ambient pressure
#     Temp in [C]
#     RH in unitless [between 0 and 1]
#     pres in Pascals
# Output Tv is in [C]
################################################################################

def tvirt(temp, rh, pres):
    rvap = 461.5
    tzero = 273.15
    vt = (temp + tzero) / (1.0 - 0.378 * epres(temp,rh)/pres) - tzero
    return vt

def q2w(q):
    qq = q/1000
    w = qq/(1-qq)
    return w*1000

