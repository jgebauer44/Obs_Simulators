import sys
import numpy as np
import scipy.io
import scipy.interpolate
from datetime import datetime
from netCDF4 import Dataset

import Calcs_Conversions

################################################################################
# This file contains the following functions:
# matchtimes()
# smooth()
# predict_co2_concentration()
# find_wnum_idx()
# add_sys_error()
# trace_gas_profile()
# inflate_prior_covariance()
# compute_pblh()
# compute_sbi()
# compute_lcl()
# get_a2_pblh()
# covariance_matrix_decorrelate_level()
# get_aeri_bb_emis()
# aeri_recal()
# find_cloud()
# get_scat_properties()
# od2numdens()
# numdens2od()
# get_ir_cld_ods()
# cloud_reflectivity()
# radxfer()
# apodize_norton_beer()
# apodize_kaiser_bessel()
# apodizer()
# convolve_to_aeri()
# write_arm_sonde_file()
# radxfer_microwave()
# compute_cgamma()
# lcurve()
# compute_vres_from_akern()
# aeri_zerofill()           -- TODO -- need to update to be consistent
# fix_aeri_vlaser_mod()
# compute_extra_layers()
# aeri_ffovc()              -- TODO -- add this bad boy
# change_aeri_ffovc()       -- TODO -- add this bad boy
################################################################################


################################################################################
# This function takes two arrays of julian times and compares the two to find where
# array two is equal to array one within a certain tolerance.
################################################################################

def matchtimes(time1,time2,delta):
    flag = np.zeros(len(time2)).astype('int')
    indx = (np.zeros(len(time2))-1).astype('int')
    
    for i in range(len(flag)):
        foo = np.where(np.nanmin(np.abs(time1-time2[i])) == np.abs(time1-time2[i]))[0]
        if len(foo) == 0:
            print('Big problem in matchtime(). How did this happen???')
            return flag, 0
        if np.abs(time1[foo[0]] - time2[i]) <= delta:
            flag[i] = 1
            indx[i] = foo[0]
    
    # By using the min distance above, we have eliminated one-to-many scenario.
    # We now need to eliminate the many-to-one possiblility
    
    vals = np.unique(indx)
    foo = np.where(vals >= 0)[0]
    for i in range(len(foo)):
        bar = np.where(indx == vals[foo[i]])[0]
        if len(bar) > 1:
            duh = np.where(np.nanmin(np.abs(time1[vals[foo[i]]] - time2[bar])) == np.abs(time1[vals[foo[i]]] - time2[bar]))
            if len(duh) == 0:
                print('Big problem in matchtime(). This should not have happened')
                return flag, 0
            flag[bar] = 0         #reset to zero
            flag[bar[duh[0]]] = 1 #and set only one
    
    return flag, 1

################################################################################
# This function is a boxcar smoother with mirrored edges. Npts needs to be an
# odd number
################################################################################

def smooth(data,npts):
    
    if npts % 2 == 0:
        print('Error in boxcar smoother. Npts needs to be an odd number. Not sure how this happened')
        return data, 0
    
    extended = np.insert(data,0,np.flipud(data[0:int(npts/2)]))
    extended = np.append(extended,np.flipud(extended[len(extended)-(int(npts/2)):len(extended)]))
    
    cumsum = np.cumsum(np.insert(extended,0,0))
    
    smoothed = (cumsum[npts:]-cumsum[:-npts])/float(npts)
    
    return smoothed, 1

################################################################################   
# This is a simple model that DDT uses to predict CO2 concentration
################################################################################

def predict_co2_concentration(yy,mm,dd):
    # Coefficients from "plot_predict_co2.pro" on nollij, which is
    # based upon Mauna Loa data from 1958-2010
    
    offset = [ 0.03, 0.67, 1.43, 2.56, 2.97, 2.31, 0.71, -1.40, -3.10, -3.21, -2.07, -0.90]
    coef2 = [0.0121792,-46.8970,45446.4]
    nyears = 100
    year0 = 1960
    x = np.arange(nyears*12)/12. + year0 + (0.5/12)
    y0 = np.array(offset).T
    for i in range(1,nyears):
        y0 = np.append(y0,offset)
    y = y0 + np.polyval(coef2, x)
    # Now get the current year + year_fraction
    jt = datetime(yy,mm,dd).timetuple().tm_yday
    yr = yy + jt/365.
    if (yr < np.nanmin(x)) | (yr > np.nanmax(x)):
        print('Error: the desired year in predict_co2_concentration is outside bounds')
        return -999.
    else:
        return np.interp(yr,x,y)

################################################################################
# This function finds the index of the spectral bands
################################################################################

def find_wnum_idx(wnum, bands):
    flg = np.zeros(len(wnum))
    sz = bands.shape
    if len(sz) == 1:
        if sz[0] != 2:
            print('Problem with band definition in find_wnum_idx')
            return -1, -1
        foo = np.where((bands[0] <= wnum) & (wnum <= bands[1]))[0]
    elif len(sz) == 2:
        if sz[0] == 2:
            for j in range(sz[1]):
                foo = np.where((bands[0,j] <= wnum) & (wnum <= bands[1,j]))[0]
                if len(foo) > 0:
                    flg[foo] = 1
    else:
        'Problem with band definition in find_wnum_idx'
        return -1, -1
    foo = np.where(flg == 1)[0]
    
    return foo, len(foo)
    
################################################################################
# This function computes a covariance matrix with off-diagonal elements; i.e.,
# this covariance includes both random error and systematic error. This
# function was written to simulate calibration error (in PWV) in the RLID WV data
################################################################################

def add_sys_error(x,sig_x,syserr):
    n = 25*len(x)
    prof = np.zeros((len(x),n))
    for i in range(n):
        calfactor = np.random.normal()*(syserr/100.) + 1
        noise = np.random.normal(size=len(x))*sig_x
        prof[:,i] = x*calfactor + noise
    
    cov = np.cov(prof)
    
    return cov
         
################################################################################
# This function computes the trace gas profile based upon the three input
# coefficients, and the assumed function form
################################################################################

def trace_gas_prof(types, z, coef):
    if types == 0:
        # The constant profile, given by the first element of the input vector
        tg_prof = np.ones(len(z))*coef[0]
    elif types == 1:
        # The exponential profile used by Kobra Khosravain.
        #       Coef[0] is the value in the free troposhere [ppm]
        #       Coef[1] is the value at the surface relative to Coef[0] [ppm]
        #       Coef[2] is the shape parameter. It should be between (approx) -20 to -1. [unitless]
        tg_prof = coef[0] + coef[1]*np.exp( coef[2]*z)
    elif types == 2:
        # The stair step profile
        #       Coef[0] is the value in the free troposhere [ppm]
        #       Coef[1] is the value in boundary layer relative to Coef[0] [ppm]
        #       Coef[2] is the top of the boundary layer [km AGL]
        tg_prof = np.ones(len(z))*coef[0]
        foo = np.where(z <= coef[2])[0]
        if len(foo) > 0:
            tg_prof[foo] += coef[1]
    else:
        # This option has not been defined. Abort !!
        print('Error in trace_gas_prof: Undefined type specified.')
        return -999.
    return tg_prof

################################################################################    
#  This function is used to inflate the variance in the temperature and/or humidity
#  profile in the near-surface portion of the prior covariance matrix.  It can 
#  also decrease the covariance between temperature and humidity over the entire matrix
################################################################################

def inflate_prior_covariance(Sa, z, prior_t_ival, prior_t_iht, prior_q_ival,
                prior_q_iht, prior_tq_cov_val, prior_chimney_ht, verbose):
                
    status = 0           # Default is bad
    
    # First, apply some QC to the input inflation factors
    if prior_t_ival < 1:
        print('Error: Prior temperature inflation value must be >= 1 -- aborting')
        return -999, status
    if prior_q_ival < 1:
        print('Error: Prior wate vapor mixing ratio inflation value must be >= 1 -- aborting')
        return -999, status
    if ((prior_t_iht <= 0) | (prior_t_iht > np.nanmax(z))):
        print('Error: Prior temperature inflation height must be a valid height (0 to max(z)) -- aborting')
        return -999, status
    if ((prior_q_iht <= 0) | (prior_t_iht > np.nanmax(z))):
        print('Error: Prior water vapor mixing ratio inflation height must be a valid height (0 to max(z)) -- aborting')
        return -999, status
    if ((prior_tq_cov_val < 0) | (prior_tq_cov_val > 1)):
        print('Error: Prior T/q covariance scalar must be in the interval [0,1] -- aborting')
        return -999, status
        
    if ((prior_t_ival > 1) & (verbose > 1)):
        print('Inflating the prior temperature profile near the surface by a factor of ' + prior_t_ival)
    if ((prior_q_ival > 1) & (verbose > 1)):
        print('Inflating the prior WVMR profile near surface by a factor of ' + prior_q_ival)
    if ((prior_tq_cov_val < 1) & (verbose > 1)):
        print('Decreasing the covariance between T and q in the prior by a factor of ' + prior_tq_cov_val)
    
    # Extract ou the variance values
    var =np.copy(np.diag(Sa))
    
    # Compute the correlation matrix, and scale the T/q covariance values
    
    C = np.copy(Sa)
    newC = np.copy(C)
    for i in range(len(Sa[0,:])):
        for j in range(len(Sa[:,0])):
            C[j,i] = Sa[j,i]/(np.sqrt(var[i])*np.sqrt(var[j]))
            newC[j,i] = np.squeeze(np.copy(C[j,i]))
            if (((j < len(z)) & (i >=len(z))) | ((i < len(z)) & (j >= len(z)))):
                newC[j,i] = newC[j,i] * prior_tq_cov_val
    
    # Scale the temperature variances
    sval = np.array([prior_t_ival,1,1])
    sht = np.array([0,prior_t_iht,np.max(z)])
    var[0:len(z)] = var[0:len(z)] * np.interp(z,sht,sval)
    
    # Scale the WVMR variances
    sval = np.array([prior_q_ival,1,1])
    sht = np.array([0,prior_q_iht,np.max(z)])
    var[len(z):2*len(z)] = var[len(z):2*len(z)] * np.interp(z,sht,sval)
    
    # For all heights below the chimney height, set all covariance to zero. In
    # other words, we don't want there to exist any correlations with heights
    # below the top of the chimney and any other level. Note that I need to do the 
    # same action for the covariance (T,T), covariance(Q,Q) and covariance(T,Q),
    # but do not make the diagonal elements zero in (T,T) or (Q,Q) as then the 
    # matrix is not invertable
    
    foo = np.where(z < prior_chimney_ht)[0]
    if ((len(foo) > 0) & (verbose >= 1)):
        print('Modifying Sa to account for a chimney effect')
    k = len(z)
    for i in range(len(foo)):
        for j in range(len(z)):
            if (i != j):
                
                # This handles the (T,T) (note that diagonal not affected)
                newC[i,j] = 0.
                newC[j,i] = 0.
                
                # This handles the (Q,Q) (note that diagonal not affected)
                newC[i+k,j+k] = 0.
                newC[j+k,i+k] = 0.
            
                # This handles the (T,Q) and (Q,T)
            newC[i,j+k] = 0.
            newC[i+k,j] = 0.
            newC[j,i+k] = 0.
            newC[j+k,i] = 0.
    
    # Rebuild the covariance matrix again
    newS = np.copy(newC)
    for i in range(len(Sa[0,:])):
        for j in range(len(Sa[:,0])):
            newS[j,i] = newC[j,i] * (np.sqrt(var[i])*np.sqrt(var[j]))
            
    status = 1
    return newS, status
    
################################################################################
# This function computes the height of the convective mixed layer (PBL) using
# a parcel theory approach (i.e., lift a parcel from the minht up and look for 
# when its potential temperature is above the surface value). I will slightly
# nudge the theta(minht) value to account for retrieval noise (this is commonly done
# even for radiosonde profiles)
################################################################################

def compute_pblh(ht, t, p, sigt, nudge=0.5, minht=0.300, maxht=6.0):
    st = np.zeros_like(sigt)
    st[0] = sigt[0]
    theta = Calcs_Conversions.t2theta(t+st, np.zeros(len(t)), p)
    sval  = theta[0]
    sval += nudge
    foo = np.where((sval < theta) & (ht >= minht))[0]
    if len(foo) == 0:
        pblh = minht
    elif foo[0] == 0:
        pblh = ht[0]
    else:
        tmpz = ht[foo[0]-1:foo[0]+1]
        tmpt = theta[foo[0]-1:foo[0]+1]

        if np.abs(tmpt[0]-tmpt[1]) < 0.001:
            pblh = ht[foo[0]]
        else:
            pblh = np.interp(sval, tmpt, tmpz)

    if pblh < minht:
        pblh = minht

    if pblh > maxht:
        pblh = maxht

    return pblh

################################################################################
# This function computes the depth of the surface based inversion (sbih, km AGL)
# intensity (sbim, degC)
################################################################################

def compute_sbi(z, t, start_height = 0.025):
    
    for ii in range(1,len(z)):
        if z[ii] > start_height:
            break
    
    if ii >= len(z)-1:
        return {'sbih':-999., 'sbim':-999.}
    
    sbi = ii-1
    for i in range(ii,len(t)):
        if ((t[i] >= t[i-1]) & (sbi == i-1)):
            sbi = i
        else:
            if sbi > i-1:
                break
    
    if sbi > ii:
        sbih = z[sbi]
        sbim = t[sbi] - t[ii]
        return {'sbih':sbih, 'sbim':sbim}
    else:
        return {'sbih':-999., 'sbim':-999.}
        
################################################################################
# This function computes the lifting condensation level (LCL). The first three
# imputes are the surface temperature [C], water vapor [g/kg], and pressure [mb],
# while the next two are profiles of pressure [mb] and height [arbitrary]. The
# LCL value returned is the height of the LCL in the same units as the height
# field.
################################################################################

def compute_lcl(tsfc, wsfc, psfc, p, z):
    
    pp = psfc - np.arange(1000)
    foo = np.where(pp > 1)[0]
    pp = pp[foo]
    kappa = 0.286
    tt = (((pp / psfc)**kappa) *(tsfc+273.16)) - 273.16
    ws = np.zeros(len(pp))
    for i in range(len(pp)):
        ws[i] = Calcs_Conversions.wsat(tt[i], pp[i])
    foo = np.where(wsfc >= ws)[0]
    if len(foo) <= 0:
        return -999.
    fct = scipy.interpolate.interp1d(p,z)
    zlcl = fct(pp[foo[0]])
    return zlcl
    
################################################################################
# This function, suggested by Kobra Khosravian, is used to get the value of the "a2"
# parameter from the boundary layer height.
################################################################################

def get_a2_pblh(pblh):
    scalar = 0.1             # We want to get the 20% value
    coef = np.log(scalar)/pblh
    return coef
    
################################################################################
# This function decorrelates the levels above the PBL from the levels below.
################################################################################
             
def covariance_matrix_decorrelate_level(Sa,z,pblh,prior_pblh_decorrelate_factor):
    
    # Compute the correlation matrix from the covariance matrix. Note I am
    # only working with the T/q part of the covariance matrix; the cloud and
    # trace gas parts will be ignored (and thus will remain the same
    
    Ca = np.copy(Sa)
    for i in range(2*len(z)):
        for j in range(2*len(z)):
            Ca[i,j] = Sa[i,j]/(np.sqrt(Sa[i,i])*np.sqrt(Sa[j,j]))
    
    # Find the decorrelation height
    lev = np.where(z >= pblh)[0][0]
    
    # Now multiply the correlation values at this level by this factor
    # (which is constrained to be between 1 and 100)
    factor = np.max([prior_pblh_decorrelate_factor, 1])
    factor = np.min([factor, 1e2])
    idx = np.arange(len(z)*2)
    f0 = np.ones(len(z)*2)*factor
    f1 = np.copy(f0)
    f1[lev] = 1
    f1[len(z)+lev] = np.sqrt(factor)
    Ca[idx,lev] = Ca[idx,lev]/f1
    Ca[lev,idx] = Ca[lev,idx]/f1
    f1 = np.copy(f0)
    f1[lev] = np.sqrt(factor)
    f1[len(z)+lev] = 1
    Ca[idx,len(z)+lev] = Ca[idx,len(z)+lev]/f1
    Ca[len(z)+lev,idx] = Ca[len(z)+lev,idx]/f1
    
    # Now convert the correlation matrix back into a covariance matrix
    newSa = np.copy(Sa)
    for i in range(2*len(z)):
        for j in range(2*len(z)):
            newSa[i,j] = Ca[i,j]*(np.sqrt(Sa[i,i])*np.sqrt(Sa[j,j]))
    
    return newSa

################################################################################
# Look at the VCEIL sample to determine if there is a cloud or not in the
# FOV. If a cloud is found at a nearby time, then capture its median value.
# First look in the innere window for a cloud, and if there is one, use that
# height (most accurate). But if no cloud is found in the inner window, then
# look in the outer window to find the approximate height where clouds could
# be located (a bit less accurate). If both fail, then use the default CBH
################################################################################

def find_cloud(aerich1, vceil, window_in, window_out, default_cbh):
    
    twin = window_in    # Full-width of time window for AERI observations [min]
    twout = window_out  # Full-width of time window for AERI observations [max]
       
        # The cloud flag. 1-> Inner window, 2-> Outer window, 3-> defaultCBH
    vcbh = np.zeros(len(aerich1['secs']))
    vflag = np.zeros(len(aerich1['secs']))
    
    for i in range(len(aerich1['secs'])):
        
        # Get any ceil cloud base height data that exist in the inner window
        
        foo = np.where((aerich1['secs'][i]-(twin/2.)*60. <= vceil['secs']) &
                       (vceil['secs'] <= aerich1['secs'][i]+(twin/2.)*60.) &
                       (vceil['cbh'] >= 0))[0]
        
        if len(foo) != 0:
            vcbh[i] = np.nanmedian(vceil['cbh'][foo])
            vflag[i] = 1
        else:
            # Get any ceil cloud base height data that exist in the outer window
            
            foo = np.where((aerich1['secs'][i]-(twout/2.)*60. <= vceil['secs']) &
                       (vceil['secs'] <= aerich1['secs'][i]+(twout/2.)*60.) &
                       (vceil['cbh'] >= 0))[0]
            
            if len(foo) != 0:
                vcbh[i] = np.nanmedian(vceil['cbh'][foo])
                vflag[i] = 2
            else:
                vcbh[i] = default_cbh
                vflag[i] = 3
    
    return ({'success':1, 'secs':aerich1['secs'], 'ymd':aerich1['ymd'], 'hour':aerich1['hour'],
            'wnum':aerich1['wnum'], 'rad':aerich1['rad'], 'cbh':vcbh, 'cbhflag':vflag,
            'atmos_pres':aerich1['atmos_pres'], 'fv':aerich1['fv'], 'fa':aerich1['fa'],
            'missingDataFlag':aerich1['missingDataFlag'], 'hatchopen':aerich1['hatchopen']})

################################################################################
# This routine returns the emissivity spectrum for the AERI blackbodies,
# interpolated onto the correct wavenumber grid.
#
# Original code from Dave Tobin, UW-Madison
# See notebook SSEC/CIMSS pg 55, or email from Tobin dated Aug 2, 2002
# Yao Te's calculations are from a Monte Carlo simulation he performed at SSEC
# during the fall of 2002. The results were passed to Dave Turner by BobK on
# 12/17/02 in an email
################################################################################

def get_aeri_bb_emis(wn,cavity_factor = 12.79, option=0):
    
    if ((cavity_factor == 12.79) & (option == 0)):
        print('WARNING: Cavity factor set to default value of 12.79')
    
    #The emissivity spectrum, from DaveT's file
    v = [400.00000, 500.00000, 550.00000, 600.00000, 650.00000, 700.00000,
         740.00000, 765.00000, 800.00000, 850.00000, 900.00000, 950.00000,
         1000.0000, 1060.0000, 1100.0000, 1150.0000, 1200.0000, 1300.0000,
         1400.0000, 1510.0000, 1550.0000, 1650.0000, 1700.0000, 1732.0000,
         1746.0000, 1800.0000, 1850.0000, 1900.0000, 2030.0000, 2100.0000,
         2200.0000, 2300.0000, 2400.0000, 2500.0000, 2600.0000, 2700.0000,
         2800.0000, 2900.0000, 3000.0000, 3100.0000]
    e = [0.96230000, 0.96280000, 0.96200000, 0.95940000, 0.95600000, 0.95560000,
         0.95410000, 0.95480000, 0.95410000, 0.95560000, 0.95600000, 0.95560000,
         0.95030000, 0.94130000, 0.94950000, 0.96860000, 0.97320000, 0.97600000,
         0.97660000, 0.97700000, 0.97700000, 0.97700000, 0.97670000, 0.97510000,
         0.97440000, 0.96690000, 0.96440000, 0.96360000, 0.96400000, 0.96400000,
         0.96460000, 0.96540000, 0.96460000, 0.96460000, 0.96540000, 0.96540000,
         0.96610000, 0.96690000, 0.96760000, 0.96760000]
         
    #Interpolate this to our desired wavenumbers
    emis = np.interp(wn,v,e)
    
    #Using the curves provided by HankR on 5 Oct 2002
    #These were derived from Monte Carlo simulations of the BBs
    
    #Emissivity of the "wall" (i.e., paint) for each of the others
    paint_emis = [0.94,0.95,0.96,0.97,0.98]
    
    #The emissivity numbers below are based upon some
    #research into various models of the AERI BBs. This research was done
    #primarily in the late 1990's - early 2000's
    
    #Hemispherical FOV emissivity
    hemis_emis = [0.995032, 0.996407, 0.997149, 0.997867, 0.998587]
    
    #Normal FOV
    normal_emis = [0.998911, 0.999107, 0.999298, 0.999482, 0.999661]
    
    #FOV is 46 mrad full-angle about normal
    restricted_emis = [0.998895, 0.999097, 0.990291, 0.999475, 0.999658]
    
    #FOV is 46 mrad, full-angle, but reflectance is diffuse
    yaote_emis00 = [0.9993954, 0.9995041, 0.9996146, 0.9997120, 0.9998141]
    
    #FOV is 46 mrad, full-angle, but reflectance is specular
    yaote_emis99 = [0.9997791, 0.9998307, 0.9998798, 0.9999208, 0.9999522]
    
    #Apply the cavity factor and convert to blackbody emissivity
    #Note that the method chosen depends on the option selected
    
    if option == -1:
        print('Returning emissivity spectrum of the paint (no cavity factor applied)')
    elif option == 0:
        #The original way to use the wavelength-independent cavity factor
        rcf = 1.0/cavity_factor
        bb_emis = emis/(emis + rcf * (1 - emis))
    elif option == 9:
        #This spectrally-dependent cavity factor, along with the coefficients in
        #in "a", are the latest results from the Monte Carlo models performed by
        #UW-Madison as of Feb 2006.
        a = np.array([0.06,-3.9,93.])
        cavity_factor = np.polyval(a, 10000./wn)
        rcf = 1.0/cavity_factor
        bb_emis = emis/(emis + rcf * (1 - emis))
    else:
        if option == 1:
            fov_emis = hemis_emis
        elif option == 2:
            fov_emis = normal_emis
        elif option == 3:
            fov_emis = restricted_emis
        elif option == 4:
            fov_emis = yaote_emis00
        elif option == 5:
            fov_emis = yaote_emis99
        else:
            print('Error: Undefined value for option in get_aeri_bb_emis()')
            return 0
        
        bb_emis = emis * 0.
        for i in range(len(bb_emis)):
            bb_emis[i] = np.interp(emis[i],paint_emis,fov_emis)
    
    return bb_emis

################################################################################
# This routine recalibrates AERI spectrum based upon the temperature and
# emissivity of the blackbodies. It can also be used to estimate the uncertainty
# in the AERI radiance based upon uncertainties in the measured BB temperatures/
# emissivities. Note that it is designed to only process one radiance spectrum
# at a time.
################################################################################

def aeri_recal(wn,rad1,th1,th2,ta1,ta2,tr1,tr2,eh1,eh2,ea1,ea2):
    
    #Start with simple QC
    n = len(wn)
    if ((n != len(rad1)) | (n != len(eh1)) | (n != len(eh2)) | (n != len(ea1)) | (n != len(ea2))):
        print('Error: the wavenumber, radiance, and emissivity arrays must be the same length')
        return 0
        
    if not ((np.isscalar(th1)) | (np.isscalar(th2)) | (np.isscalar(ta1)) | (np.isscalar(ta2)) | (np.isscalar(tr1)) | (np.isscalar(tr2))):
        print('Error: the various temperatures must all be scalars')
        return 0
    
    #Compute the radiances for the various temperatures
    
    b_th1 = Calcs_Conversions.planck(wn, th1)
    b_th2 = Calcs_Conversions.planck(wn, th2)
    b_ta1 = Calcs_Conversions.planck(wn, ta1)
    b_ta2 = Calcs_Conversions.planck(wn, ta2)
    b_tr1 = Calcs_Conversions.planck(wn, tr1)
    b_tr2 = Calcs_Conversions.planck(wn, tr2)
    
    #Compute pseudo radiances
    Bh1 = eh1 * b_th1 + (1-eh1) * b_tr1
    Ba1 = ea1 * b_ta1 + (1-ea1) * b_tr1
    Bh2 = eh2 * b_th2 + (1-eh2) * b_tr2
    Ba2 = ea2 * b_ta2 + (1-ea2) * b_tr2
    
    #and recalibrate
    
    rad2 = ((rad1 - Ba1) / (Bh1 - Ba1)) * (Bh2 - Ba2) + Ba2
    
    return rad2

################################################################################
# This routine extracts out the single scattering properties from the
# matrices that were previously read in. It then extracts out the slices
# associated with the wavenumbers just above and below the desired wavenumber,
# and then interpolates these to the desired wavenumber. Finally it
# interpolates the properties to the effective radius and returns.
# 
# The is some error checking in this routine, but if an error is found 
# the code just stops there. Probably the best answer if the code ever stops
# in one of these traps is to improve the scattering properties database(s) to
# include the wavelength/size desired
################################################################################

def get_scat_properties(field, wnum, reff, database):
    
    if field == 'ext_xsec':
        index = 3
    elif field == 'sca_xsec':
        index = 4
    elif field == 'abs_xsec':
        index = 5
    elif field == 'omega':
        index = 6
    elif field == 'w0':
        index = 6
    elif field == 'asym':
        index = 7
    elif field == 'g':
        index = 7
    elif field == 'Qe':
        index = 8
    elif field == 'Qa':
        index = 9
    elif field == 'Qs':
        index = 10
    elif field == 'vol':
        index = 11
    elif field == 'proj_area':
        index = 12
    elif field == 'pf':
        index = np.arange(database['nphase']) + 13
    else:
        print('Undefined field used in get_scat_properties()')
        return -999.
    
    scattdb = np.copy(database['data'])
    
    # Check to see if r or wnum is an array, but make sure that they are not
    # both arrays
    if ((type(wnum) == np.ndarray) & (type(reff) == np.ndarray)):
        print('Error: both Reff and wnum are arrays -- only one of these is allowed to be')
        return -999.
        
    elif type(reff) == np.ndarray:
        iidx = 2        # Interpolation index
        sidx = 1        # Selection index
        sval = wnum
        scom = 'wavenumber'
        icom = 'effective radius'
        desx = np.copy(reff)
        
    else:
        iidx = 1       # Interpolation index
        sidx = 2       # Selection index
        sval = reff
        scom = 'effective_radius'
        icom = 'wavenumber'
        if type(wnum) == np.ndarray:
            desx = np.copy(wnum)
        else:
            desx = wnum
    
    # Now find the surrounding data
    sdata = scattdb[sidx,:]
    sdata = np.sort(sdata)
    sdata_uniq = np.unique(sdata)
    bar = np.where(sdata_uniq >= sval)[0]
    if len(bar) == 0:
        print('Error: the entered ' + scom + ' is above the max value in the SSP database (' + str(np.max(sdata_uniq)) +  ')')
        return -999.
    
    if bar[0] == 0:
        print('Error: the entired ' + scom + ' is below the min value in the SSP database (' + str(np.min(sdata_uniq)) + ')')
        return -999.
    
    foo1 = np.where(scattdb[sidx,:] == sdata_uniq[bar[0]-1])[0]
    foo2 = np.where(scattdb[sidx,:] == sdata_uniq[bar[0]])[0]
    
    if len(foo1) != len(foo2):
        print('This should NEVER happen and these should be the same')
        return -999.
        
    wt = (sval-sdata_uniq[bar[0]-1])/(sdata_uniq[bar[0]]-sdata_uniq[bar[0]-1])
    
    ivalx = np.copy(scattdb[iidx,foo1])
    ivaly1 = np.copy(scattdb[index,foo1])
    ivaly2 = np.copy(scattdb[index,foo2])
    ivaly = ivaly1 * (1-wt) + wt * ivaly2
    
    if np.min(ivalx) > np.min(desx):
        print('Error: the entired ' + icom + ' is below the min value in the SSP database (' + str(np.min(ivalx)) + ')')
        return -999.
    
    if np.max(ivalx) < np.max(desx):
        print('Error: the entered ' + icom + ' is below the max value in the SSP database (' + str(np.max(ivalx)) + ')')
        return -999.
    
    desy = np.interp(desx,ivalx,ivaly)
    return desy

################################################################################
# This routine is used to convert cloud particle number density into
# optical depth.  One must be careful of the units used here.  The extinction
# cross section coefficient is assumed to have come from the single scattering
# databases, and hence have units [um^2].  All other units are in meters to
# some power.  Note that if the user wishes to use equivalent volume spheres,
# or equivalent area spheres, or equivalent volume/area spheres, the adjustment
# of the effective radius (which was used to get the extinction coefficient)
# and the number density needs to be done outside of this function.
################################################################################

def od2numdens(ext_coef, thickness, od):
    
    # Convert from [um^2] to [m^2]
    extcoefxsec = ext_coef/(1e6**2)
    
    # Get the volume extincition coef [m-1] from the optical depth
    extcoefvol = od/thickness
    
    # Back out the number density from the two extinction coefficients
    # Units are [m^-1] / [m^2] = [m^-3], which is correct
        
    numdens = extcoefvol / extcoefxsec
    
    return numdens
    
################################################################################
# This routine is used to convert cloud optical depth to cloud particle number
# density.  One must be careful of the units used here.  The extinction
# cross section coefficient is assumed to have come from the single scattering
# databases, and hence have units [um^2].  All other units are in meters to
# some power.  Note that if the user wishes to use equivalent volume spheres,
# or equivalent area spheres, or equivalent volume/area spheres, the adjustment
# of the effective radius (which was used to get the extinction coefficient)
# and the number density needs to be done outside of this function.
################################################################################

def numdens2od(ext_coef, thickness, numdens):
    
    # Convert from [um^2] to [m^2]
    extcoefxsec = ext_coef/(1e6**2)
    
    # Convert extinction cross section [m^2]
    # to volume extinction coefficient [m-1]
    extcoefvol = extcoefxsec * numdens
    
    # And assuming a constant extinction coef with thickness, compute optical depth
    od = extcoefvol * thickness
    
    return od
        
################################################################################
# This routine gets the spectral IR absorption optical depths. It uses the scattering
# properties in a SSP database (read into python with a previous call). The logic is the
# same as what is encoded in LBLDIS.
################################################################################

def get_ir_cld_ods(ssp, intau, wnum, reff, cldthick):
    
    # Quick QC
    if ((reff < np.nanmin(ssp['data'][2,:])) | (reff > np.nanmax(ssp['data'][2,:]))):
        print('Error: the desired Reff is outside the Reff range in the SSP ' + ssp['dbname'])
        return -999.
    
    # Get the extinction efficiency and extinction coefficient at the
    # reference wavenumber for the input (geometric-limit) optical depth
    # and cloud thickness...
    wnum_reff = 900.
    Qext = get_scat_properties('Qe',wnum_reff,reff,ssp)
    kext = get_scat_properties('ext_xsec',wnum_reff,reff,ssp)
    
    # Compute the number density for this cloud using the reference wavenumber
    numdens = od2numdens(kext,cldthick*1000.,intau*Qext/2.)
    
    # Get the absoprtion coefficient, absorption coefficient spectrum, and
    # then compute the spectral cloud absorption optical depth
    
    kabs = get_scat_properties('abs_xsec',wnum,reff,ssp)
    outtau = numdens2od(kabs, cldthick*1000.,numdens)
    
    return outtau
    
################################################################################
# This routine returns the IR spectral cloud reflectivity (an estimate of i,
# using the form shown in DDT's dissertation) as a function of cloud optical depth
################################################################################

def cloud_reflectivity(dwnum, tau, ice = False):
    
    if type(tau) is np.ndarray:
        print('Warning: the routine cloud_reflectivity needs tau as a signleton - returning zeros')
        return np.zeros(len(dwnum))
    
    # This spectrum is from DDT's dissertation, with water at Reff=7.5 and
    # optical depth 4.0
    
    wnum = np.array( [400.0, 478.500, 496.750, 530.700, 560.250, 772.850, 788.300, 810.950,
  	819.850, 831.450, 845.450, 862.050, 874.850, 893.850, 901.800, 934.650,
	962.100, 991.500, 1080.70, 1095.45, 1114.95, 1128.50, 1145.10, 1159.30] )
    wref40 = np.array( [0.0169606, 0.0147522, 0.0142854, 0.0130697, 0.0120277, 0.00572757,
  	0.00532051, 0.00481670, 0.00467556, 0.00448597, 0.00424655, 0.00434656,
	0.00451425, 0.00476012, 0.00492731, 0.00691303, 0.00935897, 0.0128049,
	0.0222350, 0.0237693, 0.0252363, 0.0261802, 0.0273126, 0.0281716] )
    iref40 = np.array( [0.0575502, 0.0325507, 0.0224449, 0.0170424, 0.0132832, 0.00555305,
  	0.00546641, 0.00514759, 0.00497298, 0.00474025, 0.00445012, 0.00397647, 
	0.00357329, 0.00301045, 0.00276519, 0.00185897, 0.00231630, 0.00420333, 
	0.00673380, 0.00687247, 0.00708907, 0.00725474, 0.00743006, 0.00753798] )
    
    if ice:
        wref40 = np.copy(iref40)
    
    # DDT then did some fooling around, and determined this extrapolation
    # techniue to go to at least 1400 cm-1. The better way would be to
    # recompute the cloud reflectivity spectrum in the first place to cover
    # the entire spectral range I desire.
    
    awnum = np.append(wnum,[1200,1250,1300,1350,1400])
    foo = np.where(wnum > 1050)[0]
    coef = np.polyfit(wnum[foo],wref40[foo],2)
    aref40 = np.append(wref40,[0,0,0,0,0])
    bar = np.where(awnum > 1050)[0]
    aref40[bar] = np.polyval(coef,awnum[bar])
    bar = np.where((aref40 == np.max(aref40)) & (awnum > 1100))[0]
    foo = np.where(awnum > awnum[bar[0]])[0]
    if len(foo) > 0:
        aref40[foo] = aref40[bar[0]]
    
    # This bit of logic scales the reflectivity spectrum to account for the
    # change in optical depth of the cloud
    emis = 1.-np.exp(-tau)
    reflect = aref40 * (emis**0.8)
    dreflect = np.interp(dwnum,awnum,reflect)
    
    return dreflect
    
################################################################################
# Clear-sky downwelling radiance using optical depths previously calculated.
# It uses a "linear in tau" approach. Note that the profiles *MUST* extend from
# the surface (index 0) to the TOA...
################################################################################

def radxfer(wnum, t, od, sfc_t = None, sfc_e = None, upwelling = False):
    
    # Some basic checks to ensure that the arrays are dimesioned correctly
    if type(wnum) != np.ndarray:
        opd = np.copy(od)
        a = opd.shape
        if len(a) != 1:
            print('Error: Optical depth array dimension does not match wnum dimension')
            return -1
    else:
        opd = np.copy(od)
        a = opd.shape
        if a[0] != len(wnum):
            opd = opd.T
            a = opd.shape
            if a[0] != len(wnum):
                print('Error: Optical depth array dimension does not match wnum dimension')
                return -3
    
    if a[0] != len(t)-1:
        if len(a) == 2:
            opd = opd.T
            a = opd.shape
        if a[0] == len(t)-1:
            looking_good = 1
        else:
            print('Error: Optical depth array dimension does not match temperature dimension')
            return -2
    foo = np.where(t < 0)[0]
    if len(foo) > 0:
        print('Error: temperature array has negative values; units should be Kelvin')
        return -4
    
    if upwelling:
        if sfc_e is None:
            if wnum != np.ndarray:
                sfce = 1.0
            else:   
                sfce = np.ones(len(wnum))
        else:
            sfce = np.squeeze(np.copy(sfc_e))
        foo = np.where((sfce < 0) | (sfce > 1))
        if len(foo) > 0:
            print('Error: surface emissivity is outside [0,1]')
            return -5
            
        # Make sure that the surface emissivity is the right size
        if type(wnum) != np.ndarray:
            if sfce == np.ndarray:
                print('Error: surface emissivity does not have the same dimension size as wnum')
                return -6
        else:
            if len(sfce) != len(wnum):
                print('Error: surface emissivity does not have the same dimension size as wnum')
            return -6
        
        # If no surface temperatue is given then use temp of lowest level
        if sfc_t is None:
            sfct = t[0]
        else:
            sfct = np.squeeze(np.copy(sfc_t))
        
        # set the indices for the loop
        k_start = 0
        k_end = len(opd[:,0])
        k_step = 1
    
    else:
        # No surface emission since this is downwelling
        if type(wnum) != np.ndarray:
            sfce = 0.0
        else:   
            sfce = np.zeros(len(wnum))
        sfct = 0. # Temperature of deep space...
        
        # Set the indices for the loop
        k_start = len(opd[:,0])-1
        k_end = -1
        k_step = -1
    
    # Get the radiance for a blackbody emitting at sfct. Note that
    # this isn't correct for downwelling radiation, but in that case
    # the emissivity is zero so it doesn't matter
    
    if upwelling:
        sfcb = Calcs_Conversions.planck(wnum,sfct)
    else:
        sfcb = 0.0
    
    # Now we are set to do the radiative transfer
    # Will use the linear in tau approach
    foo = np.where(opd < 1.0e-6)
    opd[foo] = 1.0e-6       # Add a small fudge factor to avoid singularity
    opd = opd.T
    rad = sfcb * sfce
    
    for k in range(k_start, k_end, k_step):
        # Get the transmission of the layer
        trans = np.exp(-1*opd[:,k])
        
        # Compute the blackbody radiance at the boundary closet to the observer
        if upwelling:
            b_boundary = Calcs_Conversions.planck(wnum,t[k+1])
        else:
            b_boundary = Calcs_Conversions.planck(wnum,t[k])
        
        # Compute the blackbody radiance at the middle of the layer (average temp)
        b_avg = Calcs_Conversions.planck(wnum, (t[k]+t[k+1])/2.)
        
        # Compute the radiance using Eq 13 in Clough et al 1992
        rad = rad*trans + (1.-trans)*(b_boundary+2*(b_avg-b_boundary)*(1./opd[:,k] - trans/(1.-trans)))
    return rad

################################################################################
# Norton-Beer apodization function
################################################################################

def apodize_norton_beer(n,md):
    n = int(n)
    beer = np.zeros(n)
    beer[0] = 1.
    for i in range(1,int(n/2)):
        if i <= md:
            beer[i] = (1-((i-1)/np.float(md))**2)**2
        else:
            beer[i] = 0.
    
    if n%2 == 0:
        beer[int(n/2):n] = np.fliplr([beer[0:int(n/2)]])[0]
    else:
        beer[int(n/2):n] = np.fliplr([beer[0:int(n/2)+1]])[0]

    return beer
    
################################################################################
# Kaiser-Bessel  apodization function
################################################################################

def apodize_kaiser_bessel(n,md,k=6):
    
    n = int(n)
    k = int(k)
    if k < 1:
        print('ERROR in apodize kaiser_bessel: Coefficient must be an integer > 6')
        sys.exit()
    
    apod = np.zeros(n)
    d = np.arange(n/2)
    d = d/np.float(np.max(d))
    d = d * (n/2)
    x = k * np.sqrt(1 - (d/np.float(md))**2)
    r = np.ones(n/2)
    f = 1
    for j in range(1,9):
        f = f * j
        r = r + ((x/2)**(2*j)) / (f**2)
    
    s = 1.
    f = 1
    for j in range(1,9):
        f = f * j
        s = s + ((k/2.)**(2*j)) / (f**2)
    
    foo = np.where(np.abs(d) <= md)[0]
    c = 1 * r[foo] / s
    apod[0:int(n/2)] = c
    
    if n%2 == 0:
        apod[int(n/2):n] = np.fliplr([apod[0:int(n/2)]])[0]
    else:
        apod[int(n/2):n] = np.fliplr([apod[0:int(n/2)+1]])[0]
    
    return apod
    
################################################################################
# Function used to apodize the input spectrum, using one of a few apodization
# functions
################################################################################

def apodizer(spectrum, aflag):
    n = len(spectrum)
    imd = n/2
    
    if aflag == 0:
        apod = apodize_norton_beer(n, imd)
    elif aflag == 1:
        apod = apodize_kaiser_bessel(n, imd)
    else:
        print('ERROR in apodizer: Undetermined apodization funcion - abort')
        sys.exit()
    
    new_spectrum = np.fft.fft(np.fft.ifft(spectrum)*apod)
    
    return new_spectrum

################################################################################
# This routine takes the input spectrum, and convolves it with the AERI filter
# function to get data at the AERI resolution.  Actually, instead of performing
# a convolution, it does this by multiplying the interferogram by the boxcar
# that is associated with the AERI's Max OPD.  To do this, the input spectrum
# is tapered and zeropadded to create a smooth spectrum, which is then interpolated
# to a multiple of the AERI's spectral resolution, which is then transformed back
# into interferogram space for chopping. 
################################################################################

def convolve_to_aeri(wnum, radiance):
    
    # These will be needed later
    minv = np.min(wnum)
    maxv = np.max(wnum)
    
    x = np.copy(wnum)
    y = np.copy(radiance)
    
    # Quick QC
    if len(y) != len(x):
        print('ERROR: wnum and radiance should have the same number of elements!')
        return 0
    
    # The AERI's delta nu
    aeri_dv = 15799./(2**15)
    # And the AERI's maximum optical path delay
    aeri_opd = 1./(2*aeri_dv)
    
    # Apply tapers to the end of the radiance spectrum
    # This will also result in the spectrum having 2^n + 1 points
    
    # Find the mean wavenumber delta. I want to use the mean, rather than
    # the difference of the first two elements
    
    delx = np.mean(x[1:len(x)] - x[0:len(x)-1])
    
    # First step: taper the head (long wavelength side) of the spectrum
    npts = int((x[0] - 0)/delx  - 1)
    xx = np.arange(npts+1) * delx
    yy = np.zeros(npts+1)
    x = np.append(xx,x)
    y = np.append(yy,y)
    
    # Second step: insert taper at tail to get the proper number of
    # points in the spectrum (2^n + 1 points)
    for i in range(100+1):
        if 2**i >= len(x):
            break
    npts = 2**(i+1) - len(x) +1
    xx = np.arange(npts) * delx + delx + x[len(x)-1]
    yy = np.zeros(len(xx))
    x = np.append(x,xx)
    y = np.append(y,yy)
    
    # Determin the size of the rolloff to apply; it should be 
    # no more than 100 cm-1, but may need to be smaller...
    rolloffsize = np.min([np.max(x)-maxv, minv, 100.])
    tapersize = 20.            # The amount of space in the spectrum to taper [cm-1]
    
    # Find the spectral regions that require a roll-off
    v_rolloff1 = np.where((minv-rolloffsize <= x) & (x <= minv))[0]
    v_rolloff2 = np.where((maxv <= x) & (x <= maxv + rolloffsize))[0]
    
    # Apply the roll-off and then make it smooth for a
    # small wavenumber region around the roll-off
    oldy = np.copy(y)               # Keep a copy for debugging, but not needed anymore
    bar = (np.cos((np.arange(len(v_rolloff1))/(len(v_rolloff1)-1.)) * np.pi - np.pi) + 1)/2.
    feh = np.where((minv <= x) & (x <= minv+tapersize))[0]
    y[v_rolloff1] = bar * np.mean(y[feh])
    weight = np.arange(len(feh))/(len(feh)-1.)
    y[feh] = y[feh]*weight + (1-weight)*np.mean(y[feh])
    
    bar = (np.cos((np.arange(len(v_rolloff2))/(len(v_rolloff2)-1.)) * np.pi) + 1)/2.
    feh = np.where((maxv-tapersize <= x) & (x <= maxv))[0]
    y[v_rolloff2] = bar * np.mean(y[feh])
    weight = np.arange(len(feh))/(len(feh)-1.)
    y[feh] = y[feh]*(1-weight) + weight*np.mean(y[feh])

    # If the wavenumber resolution is "coarse", then we need to zeropad
    # the spectrum to allow us to interpolate to a multiple of the AERI
    # resolution...
    
    # This threshold is a bit arbitrary...
    if delx > 0.01:
        
        # Now fold the spectrum to get a new spectrum with 2^(n+1) points
        n = len(y)
        yfold = np.append(y, np.fliplr([y[1:n-1]])[0])
        # Compute the interferogram
        n = len(yfold)
        inter = np.fft.ifft(yfold)*len(yfold)
        yyi = np.real(np.roll(inter, int(-1*n/2)))

        # Now we want to zeropad the spectrum to have 2^14 points
        # so we need to figure out how many zeros to put at the ends of
        # the interferogram. And we need to keep track of the factor that
        # we are expanding the interferogram by so we can multiply the
        # spectrum by it in a later step to put the energy back into it.
        
        for i in range(101):
            if 2**i >= len(x):
                break
        if i < 18:
            npts = 2**18 - 2**i
            factor = 2**18/(2**i)
            yyi_pad = np.append(np.zeros(int(npts/2)), yyi)
            yyi_pad = np.append(yyi_pad, np.zeros(int(npts/2)))
        else:
            factor = 1
            yyi_pad = np.copy(yyi)
        
        # Now compute the spectrum from this zeropadded spectrum
        n_pad = len(yyi_pad)
        yyi_pad_shift = np.roll(yyi_pad, int(n_pad/2))
        new_spec = np.fft.fft(yyi_pad_shift)/len(yyi_pad_shift)
        new_dv = delx/np.float(factor)
        
        # Cut the spectrum down (i.e., throw away the folded part)
        # and put the energy back in
        
        new_x = np.arange(len(new_spec)/2) * new_dv
        new_y = factor * np.real(new_spec[0:int(len(new_spec)/2)])
        new_delx = np.mean(new_x[1:len(new_x)] - new_x[0:len(new_x)-1])
    
    else:
        # Just reassign the vectors, as the spectral resolution is high enough
        
        new_x = np.copy(x)
        new_y = np.copy(y)
        new_delx = delx
    
    
    # Now we need to interpolate this spectrum to a multiple of the AERI's
    # wavenumber grid (power of 2, actually).
    if (aeri_dv / new_delx) > 256:
        sfac = 256.
    elif (aeri_dv / new_delx) > 128:
        sfac = 128.
    elif (aeri_dv / new_delx) > 64:
        sfac = 64.
    elif (aeri_dv / new_delx) > 32:
        sfac = 32.
    elif (aeri_dv / new_delx) > 16:
        sfac = 16.
    else:
        # If this happens, warn the user and still use sfac=16, but
        # this will not be an optimal interpolation
        print('Warning in convolve_to_aeri: Unanticipated problem in computing new_x')
        sfac = 16.
    
    new_aeri_dv = aeri_dv / sfac
    max_v = np.max(new_x)
    new_aeri_wnum = np.arange(int(max_v / new_aeri_dv)) * new_aeri_dv
    func = scipy.interpolate.interp1d(new_x,new_y, fill_value = 'extrapolate')
    new_aeri_spec = func(new_aeri_wnum)
    
    # In our desire to have a spectrum with 2^n + 1 points, I may need
    # to throw away a few points (but these are probably in the taper)
    for i in range(101):
        if 2**i >= len(new_aeri_wnum):
            break
    npts = 2**(i-1)
    new_aeri_wnum = new_aeri_wnum[0:npts+1]
    new_aeri_spec = new_aeri_spec[0:npts+1]
    
    # Now fold this spectrum, and compute its interferogrm
    n_aeri = len(new_aeri_spec)
    new_aeri_spec_fold = np.append(new_aeri_spec, np.fliplr([new_aeri_spec[1:n_aeri-1]])[0])
    n_fold = len(new_aeri_spec_fold)
    new_aeri_inter = np.fft.ifft(new_aeri_spec_fold) * len(new_aeri_spec_fold)
    new_aeri_inter = np.real(new_aeri_inter)
    new_aeri_inter = np.roll(new_aeri_inter, -1*int(n_fold/2))
    new_aeri_opd = 1./(2*new_aeri_dv)
    new_aeri_xx = ((np.arange(n_fold)/np.float(n_fold)) * 2  - 1) * new_aeri_opd
    
    # Now chop this at the desired optical path delay
    foo = np.where((-1*aeri_opd <= new_aeri_xx) & (new_aeri_xx < aeri_opd))[0]
    aeri_chop_inter = np.copy(new_aeri_inter[foo])
    
    # And transform back into the spectral domain
    n_chop = len(aeri_chop_inter)
    aeri_chop_inter_shift = np.roll(aeri_chop_inter, int(n_chop/2))
    final_aeri_spec = np.fft.fft(aeri_chop_inter_shift)/len(aeri_chop_inter_shift)
    final_aeri_spec = np.real(final_aeri_spec)
    
    # Compute the scale factor that will account for the energy redistribution
    factor = len(final_aeri_spec) / np.float(len(new_aeri_inter))
    final_aeri_spec = factor * final_aeri_spec
    
    # And compute the wavenumber grid for this data
    final_aeri_dv = 1./(2*aeri_opd)
    final_aeri_wnum = np.arange(int(len(final_aeri_spec)/2)) * final_aeri_dv
    final_aeri_spec = final_aeri_spec[0:len(final_aeri_wnum)]
    
    # And the last step: cut off data before and after the actual
    # minimum and maximum wavenumber intervals of the input data
    foo = np.where((minv+tapersize <= final_aeri_wnum) & (final_aeri_wnum <= maxv-tapersize))[0]
    if len(foo) == 0:
        print('ERROR: the spectral range is smaller than twice the tapersize -- aborting')
        return 0
    else:
        final_aeri_wnum = final_aeri_wnum[foo]
        final_aeri_spec = final_aeri_spec[foo]
    
    return {'wnum':final_aeri_wnum, 'spec':final_aeri_spec}

################################################################################
# This routine writes a netCDF file that looks like an ARM radiosonde file.
################################################################################

def write_arm_sonde_file(z, p, t, u, name, glatt=None, base_time=None, time_offset=None, silent=False):
    
    if (len(z) != len(p)) | (len(z) != len(t)) | (len(z) != len(u)):
        print('Error: Profiles are not the same length')
        print('No radiosonde netCDF file was created')
        return
    
    if not silent:
        print('Creating the file ' + name)
    
    fid = Dataset(name, 'w', format='NETCDF3_CLASSIC')
    did = fid.createDimension('time',None)
    b_time = fid.createVariable('base_time','i4')
    b_time.long_name = 'Time since 1970-01-01 00:00:00'
    b_time.units = 'seconds'
    if base_time is None:
        b_time.comment = 'This is a dummy field and has no real meaning it is only included to match the ARM data format'
    t_offset = fid.createVariable('time_offset', 'f8', ('time',))
    t_offset.long_name = 'Time since base_time'
    t_offset.units = 'seconds'
    if time_offset is None:
        t_offset.comment = 'This is a dummy field and has no real meaning it is only included to match the ARM data format'
    pres = fid.createVariable('pres', 'f4', ('time',))
    pres.long_name = 'Pressure'
    pres.units = 'hPa'
    tdry = fid.createVariable('tdry', 'f4', ('time',))
    tdry.long_name = 'Dry bulb temperature (i.e., ambient temperature)'
    tdry.units = 'C'
    rh = fid.createVariable('rh', 'f4', ('time',))
    rh.long_name = 'Relative humidity'
    rh.units = '%'
    alt = fid.createVariable('alt', 'f4', ('time',))
    alt.long_name = 'Altitude'
    alt.units = 'm MSL'
    if glatt is None:
        fid.comment = 'ad-hoc created sonde'
    else:
        comment = ''
        for i in range(glatt):
            comment = comment + glatt[i] + '; '
        fid.comment = comment
    if base_time is not None:
        b_time = base_time
    else:
        b_time = 0
    if time_offset is not None:
        if len(time_offset) == len(z):
            t_offset[0:len(z)] = time_offset
        else:
            t_offset[0:len(z)] = np.arange(len(z))*1.0
    else:
        t_offset[0:len(z)] = np.arange(len(z))*1.0
    
    pres[0:len(p)] = p[:]
    tdry[0:len(t)]= t[:]
    rh[0:len(u)] = u[:]
    alt[0:len(z)] = z[:]
    fid.close()

################################################################################
# The function to compute radiative transfer in the microwave regime
################################################################################

def radxfer_microwave(freq, t, od, sfc_t=None, sfc_e=None, upwelling=None):
    
    # Some basic checks to ensure that the arrays are dimesioned correctly
    if type(freq) != np.ndarray:
        opd = np.copy(od)
        a = opd.shape
        if len(a) != 1:
            print('Error: Optical depth array dimension does not match wnum dimension')
            return -1
    else:
        opd = np.copy(od)
        a = opd.shape
        if a[0] != len(freq):
            opd = opd.T
            a = opd.shape
            if a[0] != len(freq):
                print('Error: Optical depth array dimension does not match wnum dimension')
                return -3
    
    if a[0] != len(t)-1:
        if len(a) == 2:
            opd = opd.T
            a = opd.shape
        if a[0] == len(t)-1:
            looking_good = 1
        else:
            print('Error: Optical depth array dimension does not match temperature dimension')
            return -2
    foo = np.where(t < 0)[0]
    if len(foo) > 0:
        print('Error: temperature array has negative values; units should be Kelvin')
        return -4
    
    if upwelling:
        if sfc_e is None:
            if freq != np.ndarray:
                sfce = 1.0
            else:   
                sfce = np.ones(len(freq))
        else:
            sfce = np.copy(sfc_e)
        foo = np.where((sfce < 0) | (sfce > 1))
        if len(foo) > 0:
            print('Error: surface emissivity is outside [0,1]')
            return -5
            
        # Make sure that the surface emissivity is the right size
        if type(freq) != np.ndarray:
            if sfce == np.ndarray:
                print('Error: surface emissivity does not have the same dimension size as wnum')
                return -6
        else:
            if len(sfce) != len(freq):
                print('Error: surface emissivity does not have the same dimension size as wnum')
            return -6
        
        # If no surface temperatue is given then use temp of lowest level
        if sfc_t is None:
            sfct = t[0]
        else:
            sfct = np.copy(sfc_t)
        
        # set the indices for the loop
        k_start = 0
        k_end = len(opd[:,0])
        k_step = 1
    
    else:
        # No surface emission since this is downwelling
        if freq != np.ndarray:
            sfce = 0.0
        else:   
            sfce = np.zeros(len(freq))
        sfct = 2.75 # K, Temperature of deep space...
        
        # Set the indices for the loop
        k_start = len(opd[:,0])-1
        k_end = -1
        k_step = -1
    
    # Get the radiance for a blackbody emitting at sfct. Note that
    # this isn't correct for downwelling radiation, but in that case
    # the emissivity is zero so it doesn't matter
    
    sfcb = Calcs_Conversions.planck_microwave(freq,sfct)
    
    # Now we are set to do the radiative transfer
    # Will use the linear in tau approach
    foo = np.where(opd < 1.0e-6)[0]
    opd[foo] = 1.0e-6       # Add a small fudge factor to avoid singularity
    opd = opd.T
    rad = sfcb * sfce
    
    for k in range(k_start, k_end, k_step):
        # Get the transmission of the layer
        trans = np.exp(-1*opd[:,k])
        
        # Compute the blackbody radiance at the boundary closet to the observer
        if upwelling:
            b_boundary = Calcs_Conversions.planck_microwave(freq,t[k+1])
        else:
            b_boundary = Calcs_Conversions.planck_microwave(freq,t[k])
        
        # Compute the blackbody radiance at the middle of the layer (average temp)
        b_avg = Calcs_Conversions.planck_microwave(freq, (t[k]+t[k+1])/2.)
        
        # Compute the radiance using Eq 13 in Clough et al 1992
        rad = rad*trans + (1.-trans)*(b_boundary+2*(b_avg-b_boundary)*(1./opd[:,k] - trans/(1.-trans)))
    
    tb = Calcs_Conversions.inv_planck_microwave(freq, rad)
    
    return tb

################################################################################
# This function is part of the L-curve logic. It computes a curvature
# value as a function of the gamma value assumed. It came straight from
# the paper by Carissimo et al. 2005.
################################################################################

def compute_cgamma(ggamma, Xa, V, w, G, z):
    
    cgamma = np.zeros(len(ggamma))
    for ig in range(len(ggamma)):
        summ = np.diag(1./(ggamma[ig]+w))
        u0 = V.dot(summ).dot(V.T).dot(G.T).dot(z)
        u1 = -1 * (V.dot(summ).dot(V.T).dot(u0))
        u2 = -2 * (V.dot(summ).dot(V.T).dot(u1))
        a1 = u1.T.dot(u0) + u0.T.dot(u1)
        a2 = u2.T.dot(u0) + u0.T.dot(u2) + 2*(u1.T.dot(u1))
        b1 = (G.dot(u1)).T.dot(G.dot(u0)-z) + (G.dot(u0)-z).T.dot(G.dot(u1))
        b2 = (G.dot(u2)).T.dot(G.dot(u0)-z) + (G.dot(u0)-z).T.dot(G.dot(u2)) + 2*(G.dot(u1)).T.dot(G.dot(u1))
        cgamma[ig] = np.abs(a1*b2 - a2*b1) / ((a1**2 + b1**2)**(1.5))
    
    return cgamma
    
################################################################################
# This function is start of L-curve logic. It computes the components
# needed to find the optimal gamma value, following the logic in the paper
# by Carissimo et al. 2005.
################################################################################

def lcurve(ggamma, flagY, Y0, FXn0, Kij0, Xn0, Xa0, Sa0, Sm0, z0):
    
    # Keep on the state data associated with T and q profiless
    
    Kij = np.copy(Kij0)
    Xn = np.copy(Xn0)
    Xa = np.copy(Xa0)
    Sa = np.copy(Sa0)
    
    # Keep only the obs data associated with AERI radiances
    Y = np.copy(Y0)
    FXn = np.copy(FXn0)
    Kij = np.copy(Kij)
    Sm = np.copy(Sm0)
    
    # Do the Carissimo math
    Yn = (Y-FXn)[:,None] + np.dot(Kij,(Xn-Xa)[:,None])
    tempU,tempw,tempV = scipy.linalg.svd(Sm)
    sqrt_Sm = tempU.dot(np.diag(np.sqrt(tempw))).dot(tempU.T)
    tempU,tempw,tempV = scipy.linalg.svd(Sa)
    sqrt_Sa = tempU.dot(np.diag(np.sqrt(tempw))).dot(tempU.T)
    
    z = np.dot(np.linalg.pinv(sqrt_Sm), Yn) 
    Ji = np.dot(np.linalg.pinv(sqrt_Sm), Kij)
    G = np.dot(Ji,sqrt_Sa)
    
    U,w,V = scipy.linalg.svd(np.dot(G.T, G))
    
    cgamma = compute_cgamma(ggamma, Xa, V.T, w, G, z)
    idx = np.where(cgamma == np.max(cgamma))[0][0]
    gfac = ggamma[idx]         # this is the optimal gamma value
    print('Best gamma value using Carissimo logic is ' + str(gfac))
    # cgamma /= cgamma[idx]
    
    # # Compute the actual L-curve with the two norms
    # invSa = scipy.linalg.inv(Sa)
    # invSm = scipy.linalg.inv(Sm)
    # norm1 = np.zeros(len(ggamma))
    # norm2 = np.zeros(len(ggamma))
    # for ig in range(len(ggamma)):
    #     Xnp1 = (scipy.linalg.inv(ggamma[ig]*invSa + Kij.T.dot(invSm).dot(Kij))).dot(Kij.T).dot(invSm).dot(Yn)
    #     norm1[ig] = Xnp1.T.dot(invSa).dot(Xnp1)
    #     #tmp = Kij.dot(Xnp1) - Yn                  As stated in Carissimo paper
    #     tmp = Kij.dot(Xnp1) - Y                # What I think it should be based on Hansen paper
    #     norm2[ig] = tmp.T.dot(invSm).dot(tmp)
    
    # n = len(ggamma)
    # norm1 = np.log10(norm1)
    # norm1 = np.log10(norm2)
    
    # This is my (DDT's) way to determine the best gamma value by looking for the 
    # sharpest kink in the L-curve.  I do that by normalizing the length of
    # both legs of the L, and then looking for the point that is closest in
    # Euclidean distance to the origin (as was suggested by Hansen as one way
    # to do this).  Note I tried multiple ways to find the L-curve point including
    # looking at 2nd derivatives, rotating the L-curve spatially, etc. and this
    # method worked the best.  But I had to change the "norm2" calculation to be
    #  Kij # Xnp1 - Y (instead of subtracting the Yn) -- this seemed to be much
    # more in the spirit of how the method was supposed to work.
    #     This actually worked very well, much better than the Carissimo logic.
    #     It seems that the Carissimo logic always picks the point with the
    #     largest gamma value.  This works well, but as the iterations increase
    #     and thus Y and FXn get closer to each other, it seems that the gamma
    #     value found by the L-curve method does not decrease like I think it 
    #     should.  This makes the entire method kind of hokey.
    
    # x1 = np.copy(norm1)                      # normalized
    # y1 = np.copy(norm2)                      # shifted
    # x11 = (x1*((np.max(y1)-np.min(y1))/(np.max(x1)-np.min(x1))))       # normalized
    # y11 = y1 - np.median(y1) + np.median(x11)                          # shifted
    # r1 = np.sqrt(x11**2 + y11**2)                                      # Euclidean distance
    # ridx = np.where(r1 == np.min(r1))[0][0]
    # print('Best gamma value using Turner logic is ' + ggamma[ridx])
    
    return gfac
    
################################################################################
# The logic here is to only compute vres for the T and Q profiles, not from 
# any of the other variables. I am taking this logic from Hewison TGRS 2007,
# which is simpler than FwHM logic I was previously using...
################################################################################

def compute_vres_from_akern(akern, z, do_cdfs = False, do_area = False):
    
    vres = np.zeros((2,len(z)))
    cdfs = np.zeros((2,len(z)))
    area = np.zeros((2,len(z)))
    
    # First get the vertical spacing between the retrieval levels
    k = len(z)
    zres = [(z[1]-z[0])/2.]
    for i in range(1,k-1):
        zres.append((z[i+1]-z[i-1])/2.)
    
    zres.append((z[i]-z[i-1])*2)
    zres = np.array(zres)
    
    # Now scale the diagonal of the averaging kernal by this
    # vertical spacing
    tval = akern[0,0]
    qval = akern[k,k]
    for i in range(k):
        # Watch for zeros along the averaging kernal diagonal. If that
        # happens, then use the last good value for the vres calculation
        if akern[i,i] > 0:
           tval = akern[i,i]
        if akern[k+i,k+i] > 0:
           qval = akern[k+i,k+i]
        # capture the cumulative DFS profile
        if(i == 0):
            cdfs[0,i] = tval
            cdfs[1,i] = qval
        else:
            cdfs[0,i] = tval + cdfs[0,i-1]
            cdfs[1,i] = qval + cdfs[1,i-1]

        # This is the Hewison method
        vres[0,i] = zres[i] / tval            # temperature profile
        vres[1,i] = zres[i] / qval        # water vapor profile
    
    # Now compute the area of the averaging kernal (pg 56 in Rodgers)
    tmp = np.copy(akern[0:k,0:k])
    area[0,:] = np.squeeze(tmp.dot(np.ones(k)[:,None]))
    tmp = np.copy(akern[k:2*k,k:2*k])
    area[1,:] = np.squeeze(tmp.dot(np.ones(k)[:,None]))
    
    if do_cdfs and do_area:
        return vres, area, cdfs
    elif do_cdfs:
        return vres, cdfs
    elif do_area:
        return vres, area
    else:
        return vres

################################################################################
# Takes an AERI-like spectrum and performs Fourier interpolation from the nominal
# order 2,000 to 4,000 points to 2^14+1 points.  Copied from Paul van Delst's code
#  Inputs:
#      iwnum is in cm-1
#      irad  is in mW/(m2 sr cm-1)
#      channel is 1 or 2 (like the AERI)
#      epad is the number of wavenumbers to pad at the end of the input spectrum
#  Output:
#       2xn vector, where output(0,:) is the new wnum array
#                     and output(1,:) is the new radiance spectrum
################################################################################
def aeri_zerofill(iwnum,irad,channel,epad=0):
    eepad = np.abs(epad)
    band  = channel - 1
    v_laser = 15799.0
    rf      = [4.0, 2.0]
    v_nyquist = v_laser / (2. * rf[band])
    if channel == 1: n_pts = 4097
    elif channel == 2: n_pts = 8193
    else:
        print('ERROR in aeri_zerofill -- channel is not properly defined')
        sys.exit()
    
    v = v_nyquist * np.arange(n_pts) / (n_pts - 1)

    # Determine the temperature to use for the Planck function,
    # which is used to help pad the data
    v_teststart = [627.0, 2380.]
    v_testend   = [632.0, 2384.]
    loc = np.where((v_teststart[band] <= iwnum) & (iwnum <= v_testend[band]))
    tb = Calcs_Conversions.invplanck(iwnum[loc], irad[loc])
    avg_tb = np.mean(tb)
    r = Calcs_Conversions.planck(v, avg_tb)
    r[0] = 0.

    # Determine the bounds of the AERI data
    n_apts = len(iwnum)
    thres = 0.1
    count = 0
    ntries = 0
    while (count != n_apts) & (ntries < 10):
        loc = np.where((iwnum[0]-thres <= v) & (v <= iwnum[len(iwnum)-1]+thres))[0]
        count = len(loc)
        thres += 0.05
        ntries += 1

        if ntries >= 10:
            print('Error: unable to find the right bounds for the AERI in aeri_zerofill')
            sys.exit()

    # Replace the AERI data in the planck radiance curve
    r[loc] = irad

    # Fourier transform the radiance data
    rr = np.append(r, np.flip(r[1:n_pts-1]))
    # Quick trap to make sure we have this size right
    if(len(rr) != 2*n_pts-2):
        print('Problem in aeri_zerofill: the length of the appended vector is not correct 1')
        sys.exit()
    ifg = np.fft.fft(rr)
    
    # Zerofill the inteferogram
    n_total = 2**18
    fill = np.zeros(n_total - len(ifg))
    ifg  = np.concatenate((ifg[0:n_pts], fill, ifg[n_pts:2*(n_pts-1)]))
    # Quick trap to make sure we have this size right
    if(len(ifg) != n_total):
        print('Problem in aeri_zerofill: the length of the appended vector is not correct 2')
        sys.exit()

    # Fourier transform the zerofilled interferogram
    spc = np.fft.ifft(ifg) * len(ifg)
    spc = spc[0:int(n_total/2)]
    v2 = v_nyquist * np.arange(n_total/2) / (n_total/2)
    loc = np.where((iwnum[0]-eepad <= v2) & (v2 <= iwnum[n_apts-1]+eepad))
   
    return [v2[loc],spc[loc]]

################################################################################
# This function spectrally calibrates the AERi.  It requires input from a routine
# like "check_aeri_vlaser".  It only works with one channel at a time
################################################################################
def fix_aeri_vlaser_mod(wnum,irad,multiplier):
    # determine which channels this is
    foo = np.where((wnum > 900) & (wnum < 901))
    if(len(foo) > 0):
        channel = 1
    else:
        channel = 2

    nrad = irad * 0.
    for ii in range(0,len(irad[0,:])):
        result = aeri_zerofill(wnum, irad[:,ii], channel)
        newrad = np.interp(wnum, result[0,:]*multiplier, result[1,:])
        nrad[:,ii] = newrad

    return nrad

################################################################################
# This routine creates a vertical grid that can be added to the top of our profile
################################################################################
def compute_extra_layers(maxz):
    max_step_size = 5    # This is the maximum step size, in km
    max_ht        = 40   # This is the maximum height to take the extra RT layers
    stepsize = np.round(maxz / 4. * 10) / 10.
    rt_extra_layers = maxz + stepsize
    while(np.max(rt_extra_layers) < max_ht):
        stepsize = np.round(np.max(rt_extra_layers) / 4. * 10) / 10.
        if(stepsize > max_step_size):
            stepsize = max_step_size
        rt_extra_layers = np.append(rt_extra_layers, np.max(rt_extra_layers) + stepsize)
    
    return rt_extra_layers

################################################################################
# This function removes the original finite field of view correction applied 
# to the AERI data, and replaces it with a new one.  The correction is determined
# by the half angle of the field of view of the instrument (assuming the detector
# is perfectly on axis -- following Knuteson et al. 2004)
################################################################################
def change_aeri_ffovc(wnum,orad,orig_halfAngle,new_halfAngle):
    print('Stubbing this change_aeri_ffovc function out -- it does nothing')
    return orad

###############################################################################
# This function checks to make sure sounding data altitude is monotonic and if
# no it will return the indices to make the altitudes monotonic by removing the
# bad points. It is a bit hackish but it works.
###############################################################################
def make_monotonic(heights):
    
    z = np.copy(heights)
    indices = np.arange(len(z))
        
    # It is possible we will have to do this mulitple times (hopefully not)
    # This fixes those cases.
        
    bad = True          # A flag we will use to quit the loop
    while bad:
        foo = np.where(np.diff(z) <= 0)[0]
        if len(foo) > 0:
            z[foo+1] = -999
            fah = np.where(z != -999)[0]
            z = z[fah]
            indices = indices[fah]
        else:
            bad = False
        
    return indices


###############################################################################
# Get optical depth of clouds from model data. Right now we only calculate them
# for liquid cloud drops and combined snow/ice. If there is hail/rain the AERI
# or other radiance measuring instrument would likely not work anyway
############################################################################### 

def get_model_tau(effc,effsi,qx,ntx,rho,zdif,sspl,sspi):
    
    # Now we need to the the liquid and ice cloud optical depth (geometric-limit). Doing this
    # in a pretty convoluted way so might need to come back to this. Using Dave
    # Turner's lookup tables because the code for the Goddard radiation scheme
    # makes no sense
    
    # Get the extinction efficiency and extinction coefficient at the
    # reference wavenumber for the input (geometric-limit) optical depth
    # and cloud thickness...
    
    # Start with water. Note we are exluding rain since if it is raining the 
    # AERI would be off
    
    wnum_reff = 900.
    Qext = get_scat_properties('Qe',wnum_reff,effc,sspl)
    kext = get_scat_properties('ext_xsec',wnum_reff,effc,sspl)
    
    tauc = (2*numdens2od(kext,zdif,ntx['cloud']*rho)/Qext)
    
    # Now get the combined snow/ice optical depth

    Qext = get_scat_properties('Qe',wnum_reff,effsi,sspi)
    kext = get_scat_properties('ext_xsec',wnum_reff,effsi,sspi)
    taui = (2*numdens2od(kext,zdif,(ntx['ice'])*rho))/Qext
    
    # Now zero out od for points that out below particle threshold
    
    print(np.nanmax(qx['cloud']))
    foo = np.where(qx['cloud'] < 1e-6)
    tauc[foo] = 0
    
    print(np.nanmax(qx['ice']))
    foo = np.where(qx['ice'] < 1e-6)
    taui[foo] = 0
    
    return tauc, taui

###############################################################################
# This function calculates density
###############################################################################

def get_density(t,q,p):
    
    rd = 287.04
    tv = t*(1+0.609*(q/1000.))
    
    rho = p/(rd*tv)
    
    return rho

def fix_nonphysical_wv(wv, z, foo):
    
    old_wv = np.copy(wv)
    old_wv[foo] = np.nan
    
    new_wv = []
    for i, value in enumerate(old_wv):
        if np.isnan(value):
            # Find first good point
            j = i
            while np.isnan(old_wv[j]) and j > 0:
                j -= 1
        
            # Find second good point
            k = i
            while np.isnan(old_wv[k]) and k < len(old_wv) - 1:
                k += 1
        
            if np.isnan(old_wv[j]):
                # The first value is nonphysical so set to nearest good value after
                new_wv.append(old_wv[k])
            elif np.isnan(old_wv[k]):
                # The last value is nonphysical so set to nearest good value before
                new_wv.append(old_wv[j])
            else:
                # Interpolate across the good values
                new_wv.append(old_wv[j] + (old_wv[k] - old_wv[j]) * (z[i]-z[j])/(z[k]-z[j]))
        else:
            new_wv.append(value)
    
    return np.array(new_wv)