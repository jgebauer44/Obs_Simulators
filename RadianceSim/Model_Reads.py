import os
import numpy as np
import scipy.io
import pyproj
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator

import Calcs_Conversions
import cal_eff_rad
import Other_functions

###############################################################################
# This is just the driver function for reading the model data. It just farms
# out the work to the individual scripts for the different types of models.
###############################################################################
def read_model_data(time, namelist,sspl,sspi):
    
    success = 0
    if namelist['model'] == 1:
        data,flag = read_wrf(namelist['profiler_x'], namelist['profiler_y'],time,
                        namelist['model_dir'],namelist['model_prefix'],sspl,sspi,namelist['mp_scheme'],latlon=namelist['coordinate_type'],ndcnst=namelist['ndcnst'])
    elif namelist['model'] == 5:
        data, flag = read_cm1(namelist['profiler_x'], namelist['profiler_y'],time, namelist['model_frequency'],
                        namelist['model_dir'],namelist['model_prefix'],sspl,sspi,namelist['mp_scheme'],namelist['ndcnst'])
    else:
        print('Model type ' + str(namelist['model']) + ' is unknown')
        return -999.,success
    
    if flag == 0:
        return -999., success
    else:
        success = 1
        return data, success

###############################################################################
# This function reads in data that is needed for the radiative transfer
# from WRF model output.
###############################################################################

def read_wrf(x, y, time, model_dir, prefix, sspl, sspi, mp_scheme, latlon=1, ndcnst=250):
    
    # Account for formatting differences
    if os.path.isfile(model_dir + '/' + prefix + time.strftime('%Y-%m-%d_%H_%M_%S')):
        file = model_dir + '/' + prefix + time.strftime('%Y-%m-%d_%H_%M_%S')
    else:
        file = model_dir + '/' + prefix + time.strftime('%Y-%m-%d_%H:%M:%S')
    
    try:
        fid = Dataset(file,'r')
    except:
        print('Could not open ' + file)
        return -999., 0
    
    # Get the the location of the profiler based on the map projection
    if latlon == 1:
        
        # LCC projection
        if fid.MAP_PROJ == 1:
            wrf_proj = pyproj.Proj(proj='lcc',lat_1 = fid.TRUELAT1, lat_2 = fid.TRUELAT2, lat_0 = fid.MOAD_CEN_LAT, lon_0 = fid.STAND_LON, a = 6370000, b = 6370000)
            wgs_proj = pyproj.Proj(proj='latlong',datum='WGS84')
            transformer = pyproj.Transformer.from_proj(wgs_proj,wrf_proj)
        
        # Now transform the data
        e, n = transformer.transform(fid.CEN_LON, fid.CEN_LAT)
        dx,dy = fid.DX, fid.DY
        nx, ny = fid.dimensions['west_east'].size, fid.dimensions['south_north'].size
        x0 = -(nx-1) / 2. * dx + e
        y0 = -(ny-1) / 2. * dy + n
        x_grid = np.arange(nx) * dx + x0
        y_grid = np.arange(ny) * dy + y0
        xx, yy = np.meshgrid(np.arange(nx) * dx + x0, np.arange(ny) * dy + y0)
        prof_x_proj, prof_y_proj = transformer.transform(x, y)
        
    else:
        xx, yy = np.meshgrid(np.arange(fid.dimensions['west_east'].size) * fid.DX, np.arange(fid.dimensions['south_north'].size) * fid.DY)
        prof_x_proj = x
        prof_y_proj = y
    
    # Get the data
    zz = (fid.variables['PH'][0,:,:,:]+fid.variables['PHB'][0,:,:,:])/9.81
    
    fake_z = np.arange(zz.shape[0]-1)
    fake_z1 = np.arange(zz.shape[0])
    f = RegularGridInterpolator((fake_z1,y_grid,x_grid), zz, bounds_error=False)
    zz = f((fake_z1,np.ones(fake_z1.shape[0])*prof_y_proj,np.ones(fake_z1.shape[0])*prof_x_proj))
    zdif = zz[1:]-zz[:-1]
    zz = (zz[1:]+zz[:-1])/2.
    
    p = (fid.variables['P'][0,:,:,:] + fid.variables['PB'][0,:,:,:])/100
    f = RegularGridInterpolator((fake_z,y_grid,x_grid), p, bounds_error=False)
    p = f((fake_z,np.ones(fake_z.shape[0])*prof_y_proj,np.ones(fake_z.shape[0])*prof_x_proj))
    
    ground = fid.variables['HGT'][0,:,:]
    f = RegularGridInterpolator((y_grid,x_grid), ground, bounds_error=False)
    ground = f((prof_y_proj,prof_x_proj))
    
    #t = Calcs_Conversions.theta2t(f.variables['T'][0,:,:,:]+300, np.zeros(len(p)), p)
    t = fid.variables['T'][0] + 300
    f = RegularGridInterpolator((fake_z,y_grid,x_grid), t, bounds_error=False)
    t = f((fake_z,np.ones(fake_z.shape[0])*prof_y_proj,np.ones(fake_z.shape[0])*prof_x_proj))
    t = Calcs_Conversions.theta2t(t, np.zeros(len(p)), p)
    
    tsfc = fid.variables['T2'][0]-273.15
    f = RegularGridInterpolator((y_grid,x_grid), tsfc, bounds_error=False)
    tsfc = f((prof_y_proj,prof_x_proj))
    
    qsfc = fid.variables['Q2'][0]*1000
    f = RegularGridInterpolator((y_grid,x_grid), qsfc, bounds_error=False)
    qsfc = f((prof_y_proj,prof_x_proj))
    wsfc = Calcs_Conversions.q2w(qsfc)
    
    q = fid.variables['QVAPOR'][0]*1000
    f = RegularGridInterpolator((fake_z,y_grid,x_grid), q, bounds_error=False)
    q = f((fake_z,np.ones(fake_z.shape[0])*prof_y_proj,np.ones(fake_z.shape[0])*prof_x_proj))
    w = Calcs_Conversions.q2w(q)
    
    # Adding a check to see if surface pressure is in the file. If not calculate it
    if 'PSFC' in fid.variables.keys():
        psfc = fid.variables['PSFC'][0]
        f = RegularGridInterpolator((y_grid,x_grid), psfc, bounds_error=False)
        psfc = f((prof_y_proj,prof_x_proj))
    else:
        tv = (t[0]+273.16)*(1+0.61*q[0]/1000)
        psfc = p[0]*np.exp(9.81*(zz[0]-ground)/(287.*tv))

    rho = Other_functions.get_density(t+273.16,w,p)
    
    #--- Create Dictionaries to Store Hydrometeor information
    qx = {}   # kg kg^-1
    ntx = {}  # kg^-1
    rhox = {} # kg m^-3

    #--- Snow
    qx['snow']  = fid.variables['QSNOW'][0]
    f = RegularGridInterpolator((fake_z,y_grid,x_grid), qx['snow'], bounds_error=False)
    qx['snow'] = f((fake_z,np.ones(fake_z.shape[0])*prof_y_proj,np.ones(fake_z.shape[0])*prof_x_proj))
    
    ntx['snow'] = fid.variables['QNSNOW'][0]
    f = RegularGridInterpolator((fake_z,y_grid,x_grid), ntx['snow'], bounds_error=False)
    ntx['snow'] = f((fake_z,np.ones(fake_z.shape[0])*prof_y_proj,np.ones(fake_z.shape[0])*prof_x_proj))
    
    #--- Ice 
    qx['ice'] = fid.variables['QICE'][0]
    f = RegularGridInterpolator((fake_z,y_grid,x_grid), qx['ice'], bounds_error=False)
    qx['ice'] = f((fake_z,np.ones(fake_z.shape[0])*prof_y_proj,np.ones(fake_z.shape[0])*prof_x_proj))
    
    ntx['ice'] = fid.variables['QNICE'][0]
    f = RegularGridInterpolator((fake_z,y_grid,x_grid), ntx['ice'], bounds_error=False)
    ntx['ice'] = f((fake_z,np.ones(fake_z.shape[0])*prof_y_proj,np.ones(fake_z.shape[0])*prof_x_proj))
    
    qx['cloud'] = fid.variables['QCLOUD'][0]
    f = RegularGridInterpolator((fake_z,y_grid,x_grid), qx['cloud'], bounds_error=False)
    qx['cloud'] = f((fake_z,np.ones(fake_z.shape[0])*prof_y_proj,np.ones(fake_z.shape[0])*prof_x_proj))
    
    ntx['cloud'] = fid.variables['QNDROP'][0] 
    f = RegularGridInterpolator((fake_z,y_grid,x_grid), ntx['cloud'], bounds_error=False)
    ntx['cloud'] = f((fake_z,np.ones(fake_z.shape[0])*prof_y_proj,np.ones(fake_z.shape[0])*prof_x_proj))
    
    precip = np.nanmax(fid.variables['REFL_10CM'][0],axis=0)
    f = RegularGridInterpolator((y_grid,x_grid), precip, bounds_error=False)
    precip = f((prof_y_proj,prof_x_proj))
    
    # Calculate LWP
    lwp = fid.variables['LWP'][0]
    f = RegularGridInterpolator((y_grid,x_grid), lwp, bounds_error=False)
    lwp = f((prof_y_proj,prof_x_proj))
    
    fid.close()
    
    if mp_scheme == 1:
        rhox['snow'] = 100.
        rhox['ice'] = 900.
        rhox['cloud'] = 1000.
        effc = cal_eff_rad.cal_eff_rad_nssl(qx,ntx,rhox,rho,'cloud')
        effi = cal_eff_rad.cal_eff_rad_nssl(qx,ntx,rhox,rho,'ice')
    elif mp_scheme == 2:
        rhox['snow'] = 100.
        rhox['ice'] = 500.
        rhox['cloud'] = 997.
        effc = cal_eff_rad.cal_eff_rad_morrison(qx,ntx,rhox,rho,'cloud')
        effi = cal_eff_rad.cal_eff_rad_morrison(qx,ntx,rhox,rho,'ice')
    else:
        print('Error: Invalid microphysics scheme')
        return -999., 0
    
    # Have to set limits on the effective radius due to our scattering databases
    effc = np.where(effc <= 1,1.01,effc)
    effc = np.where(effc > 1000, 1000, effc)
    
    
    effi = np.where(effi <= 1,1.01,effi)
    effi = np.where(effi > 1000, 1000, effi)     
    
    # Now we need to the the liquid and ice cloud optical depth (geometric-limit). Doing this
    # in a pretty convoluted way so might need to come back to this. Using Dave
    # Turner's lookup tables because the code for the Goddard radiation scheme
    # makes no sense
    
    tauc, taui = Other_functions.get_model_tau(effc,effi,qx,ntx,rho,zdif,sspl,sspi)
    
    # Find the cloud base height and top using tauc and tausi. We know that any non-zero
    # optical depths mean there is cloud influence
    
    foo = np.where(tauc > 0)[0]
    
    if len(foo) > 0:
        temp_cbh = zz[foo[0]]
        temp_cth = zz[foo[-1]]
    else:
        print('No liquid clouds')
        temp_cbh = np.nan
        temp_cth = np.nan 
    
    foo = np.where(taui > 0)[0]
    if len(foo) > 0:
        if ((temp_cbh > zz[foo[0]]) | (np.isnan(temp_cbh))):
            cbh = zz[foo[0]]
        else:
            cbh = temp_cth
        
        if ((temp_cth < zz[foo[-1]]) | (np.isnan(temp_cth))):
            cth = zz[foo[-1]]
        else:
            cth = temp_cth
    else:
        print('No ice clouds')
        cbh = temp_cbh
        cth = temp_cth
            
    
    
    model = {'t':t, 'w':w, 'rho':rho,'p':p, 'psfc':psfc, 'tsfc':tsfc, 'wsfc':wsfc,
             'z':zz/1000, 'zdif':zdif/1000, 'effc':effc, 'effi':effi, 'tauc':tauc,
             'taui':taui, 'cbh':cbh/1000, 'cth':cth/1000, 'precip':precip, 'lwp':lwp,
             'alt':ground/1000}
    
    return model,1
    
    
###############################################################################
# This function reads in data that is needed for the radiative transer
# from CM1 model output.
###############################################################################

def read_cm1(x,y,time,model_frequency,model_dir,prefix,sspl,sspi,mp_scheme,ndcnst=250):
    
    if int(time/model_frequency) < 10:
        file = model_dir + '/' + prefix + '_00000' + str(int(time/model_frequency)) + '.nc'
    elif int(time/model_frequency) < 100:
        file = model_dir + '/' + prefix + '_0000' + str(int(time/model_frequency)) + '.nc'
    elif int(time/model_frequency) < 1000:
        file = model_dir + '/' + prefix + '_000' + str(int(time/model_frequency)) + '.nc'
    elif int(time/model_frequency) < 10000:
        file = model_dir + '/' + prefix + '_00' + str(int(time/model_frequency)) + '.nc'
    elif int(time/model_frequency) < 100000:
        file = model_dir + '/' + prefix + '_0' + str(int(time/model_frequency)) + '.nc'
    else:
        file = model_dir + '/' + prefix + '_' + str(int(time/model_frequency)) + '.nc'
        
    try:
        f = Dataset(file,'r')
    except:
        print('Could not open ' + file)
        return -999., 0
    
    xx = f.variables['xh'][:]*1000
    yy = f.variables['yh'][:]*1000
    zz = f.variables['zh'][:]*1000
    zdif = (f.variables['zf'][1:]-f.variables['zf'][:-1])*1000.
    
    # Find the indices of the location in the model where we need the data
    x_index = np.argmin(np.abs(x-xx))
    y_index = np.argmin(np.abs(y-yy))
    
    
    # Get the data
    p = f.variables['prs'][0,:,y_index,x_index]/100.
    psfc = f.variables['psfc'][0,y_index,x_index]/100
    
    t = Calcs_Conversions.theta2t(f.variables['th'][0,:,y_index,x_index], np.zeros(len(p)), p)
    tsfc = f.variables['t2'][0,y_index,x_index]
    qsfc = f.variables['q2'][0,y_index,x_index]
    w = f.variables['qv'][0,:,y_index,x_index]*1000
    rho = f.variables['rho'][0,:,y_index,x_index]
    
    #--- Create Dictionaries to Store Hydrometeor information
    qx = {}   # kg kg^-1
    ntx = {}  # kg^-1
    rhox = {} # kg m^-3
    
    #--- Snow
    qx['snow']  = f.variables['qs'][0,:,y_index,x_index]
    ntx['snow'] = f.variables['ncs'][0,:,y_index,x_index]
    rhox['snow'] = 100.
    #--- ice 
    qx['ice'] = f.variables['qi'][0,:,y_index,x_index]
    ntx['ice'] = f.variables['nci'][0,:,y_index,x_index]
    rhox['ice'] = 500.
   
    #---Cloud 
    qx['cloud'] = f.variables['qc'][0,:,y_index,x_index]
    ntx['cloud'] = (ndcnst * 1E6)/rho 
    rhox['cloud'] = 997.
    
    precip = f.variables['prate'][0,y_index,x_index]
    
    # Calculate LWP
    lwp = np.trapz((qx['cloud']+qx['rain'])*rho*1000,zz)
    
    # Calculate effective radius
    
    if mp_scheme == 1:
        rhox['snow'] = 100.
        rhox['ice'] = 900.
        rhox['cloud'] = 1000.
        effc = cal_eff_rad.cal_eff_rad_nssl(qx,ntx,rhox,rho,'cloud')
        effi = cal_eff_rad.cal_eff_rad_nssl(qx,ntx,rhox,rho,'ice')
    elif mp_scheme == 2:
        rhox['snow'] = 100.
        rhox['ice'] = 500.
        rhox['cloud'] = 997.
        effc = cal_eff_rad.cal_eff_rad_morrison(qx,ntx,rhox,rho,'cloud')
        effi = cal_eff_rad.cal_eff_rad_morrison(qx,ntx,rhox,rho,'ice')
    else:
        print('Error: Invalid microphysics scheme')
        return -999., 0
    
    # Have to set limits on the effective radius due to our scattering databases
    effc = np.where(effc <= 1,1.01,effc)
    effc = np.where(effc > 1000, 1000, effc)
    
    effi = np.where(effi <= 1,1.01,effi)
    effi = np.where(effi > 1000, 1000, effi)
    
    
    # Now we need to the the liquid and ice cloud optical depth (geometric-limit). Doing this
    # in a pretty convoluted way so might need to come back to this. Using Dave
    # Turner's lookup tables because the code for the Goddard radiation scheme
    # makes no sense
    
    tauc, taui = Other_functions.get_model_tau(effc,effi,qx,ntx,rho,zdif,sspl,sspi)
    
    # Find the cloud base height and top using tauc and tausi. We know that any non-zero
    # optical depths mean there is cloud influence
    
    foo = np.where(tauc > 0)[0]
    
    if len(foo) > 0:
        temp_cbh = zz[foo[0]]
        temp_cth = zz[foo[-1]]
    else:
        print('No liquid clouds')
        temp_cbh = np.nan
        temp_cth = np.nan 
    
    foo = np.where(taui > 0)[0]
    if len(foo) > 0:
        if ((temp_cbh > zz[foo[0]]) | (np.isnan(temp_cbh))):
            cbh = zz[foo[0]]
        else:
            cbh = temp_cth
        
        if ((temp_cth < zz[foo[-1]]) | (np.isnan(temp_cth))):
            cth = zz[foo[-1]]
        else:
            cth = temp_cth
    else:
        print('No ice clouds')
        cbh = temp_cbh
        cth = temp_cth
            
    
    
    model = {'t':t, 'w':w, 'rho':rho,'p':p, 'psfc':psfc, 'tsfc':tsfc, 'qsfc':qsfc,
             'z':zz/1000, 'zdif':zdif/1000, 'effc':effc, 'effi':effi, 'tauc':tauc,
             'taui':taui, 'cbh':cbh/1000, 'cth':cth/1000, 'precip':precip, 'lwp':lwp,
             'alt':0}
    
    return model,1
    
###############################################################################
# This function reads in simulated AERI and MWR for use with TROPoeSim
###############################################################################

def read_all_data(vip,date):
    
    success = 1 
    if ((vip['aeri'] == 0) & (vip['mwr'] == 0) & (vip['mwrscan'] == 0)):
        print('Error: Must have either simulated AERI or MWR data')
        return success, -999.,-999.,-999.
    
    if vip['mwr'] == 1:
        
        try:
            f = Dataset(vip['mwr_path'])
        except:
            print('Could not open simulated MWR data')
            return success, -999.,-999.,-999.
        
        freq = f.variables['freq'][:]
        bt = f.variables['brightness_temp'][:]
        noise = f.variables['noise'][:]
        psfc = f.variables['model_psfc'][:]
        wsfc = f.variables['model_wsfc'][:]
        tsfc = f.variables['model_tsfc'][:]
        precip = f.variables['precip'][:]
        cbh = f.variables['model_cbh'][:]
        time = f.variables['base_time'] + f.variables['time'][:]
        x = f.profiler_x
        y = f.profiler_y
        alt = f.profiler_alt
        
        cbh[np.isnan(cbh)] = vip['cbh_default_ht']
        
        cbhflag = np.ones(len(time))
        cbhflag[np.isnan(cbh)] =  0
        
        f.close()
        
        mwr = {'success':1, 'secs':time, 'freq':np.copy(freq), 'bt':bt, 'noise':noise,'tsfc':tsfc,'wsfc':wsfc,
                'psfc':psfc, 'cbh':cbh, 'cbhflag':cbhflag,'precip':precip, 'date':date, 'lat':y,'lon':x}
    
    else:
        mwr = {'success':1}
    
    if vip['mwrscan'] == 1:
        
        try:
            f = Dataset(vip['mwrscan_path'])
        except:
            print('Could not open simulated MWR data')
            return success, -999.,-999.,-999.
        
        freq = f.variables['freq'][:]
        elev = f.variables['elev'][:]
        bt = f.variables['brightness_temp'][:]
        noise = f.variables['noise'][:]
        psfc = f.variables['model_psfc'][:]
        wsfc = f.variables['model_wsfc'][:]
        tsfc = f.variables['model_tsfc'][:]
        precip = f.variables['precip'][:]
        cbh = f.variables['model_cbh'][:]
        time = f.variables['base_time'] + f.variables['time'][:]
        x = f.profiler_x
        y = f.profiler_y
        alt = f.profiler_alt
        
        
        cbh[np.isnan(cbh)] = vip['cbh_default_ht']
        
        cbhflag = np.ones(len(time))
        cbhflag[np.isnan(cbh)] =  0
        
        f.close()
        
        mwrscan = {'success':1, 'secs':time, 'freq':np.copy(freq), 'bt':bt, 'noise':noise,'tsfc':tsfc,'wsfc':wsfc,
                'elev':elev,'psfc':psfc, 'cbh':cbh, 'cbhflag':cbhflag,'precip':precip, 'date':date, 'lat':y,'lon':x}
    
    else:
        mwrscan = {'success':1}
        
    if vip['aeri'] == 1:
        
        try:
            f = Dataset(vip['aeri_path'])
        except:
            print('Could not open simulated AERI data')
            return success, -999.,-999.,-999.
        
        wnum = f.variables['wnum'][:]
        rad = f.variables['rad'][:]
        noise = f.variables['noise'][:]
        psfc = f.variables['model_psfc'][:]
        wsfc = f.variables['model_wsfc'][:]
        tsfc = f.variables['model_tsfc'][:]
        precip = f.variables['precip'][:]
        cbh = f.variables['model_cbh'][:]
        time = f.variables['base_time'] + f.variables['time'][:]
        x = f.profiler_x
        y = f.profiler_y
        alt = f.profiler_alt
        
        cbh[np.isnan(cbh)] = vip['cbh_default_ht']
        
        cbhflag = np.ones(len(time))
        cbhflag[np.isnan(cbh)] =  0
        
        f.close()
        
        if vip['irs_min_noise_flag'] != 0:
            nmessage = 0
            
            parts = vip['irs_min_noise_wnum'].split(',')
            fwnum = np.array(parts).astype(float)
            parts = vip['irs_min_noise_spec'].split(',')
            
            if len(parts) != len(fwnum):
                print('Error: The number of entered VIP.irs_min_noise_wnum does not match number of VIP.irs_min_noise_spec')
                return success,-999.,-999.,-999.
            else:
                fnoise = np.array(parts).astype(float)
            
            # Interpolate the input noise floor array to the current spectral grid (no extrapolation)
            floor = np.interp(wnum,fwnum,fnoise)

            for j in range(len(wnum)):
                foo = np.where(noise[:,j] < floor[j])[0]
                if len(foo) > 0:
                    noise[foo,j] = floor[j]
                    if((nmessage == 0)):
                        print('    Resetting some of IRS noise spectrum, which was below the noise floor')
                        nmessage = 1
                        
            
        aeri = {'success':1, 'secs':time, 'wnum':wnum, 'radmn':rad, 'noise':noise, 'tsfc':tsfc ,'wsfc':wsfc,
                'psfc':psfc, 'cbh':cbh, 'cbhflag':cbhflag,'precip':precip, 'date':date, 'lat':y,'lon':x,
                'alt':alt}
    
    else:
        # If there is no AERI data then there must be MWR data so set everything
        # to mwr values
        aeri = {'success':1, 'secs':time,'psfc':psfc,'cbh':cbh,'cbhflag':cbhflag,'precip':precip,
                'date':date, 'lat':y,'lon':x, 'alt':alt, 'tsfc':tsfc, 'wsfc':wsfc,}
    

    # Check to make sure the times are all the same
    

    if ((vip['aeri'] == 1) & (vip['mwr'] == 1)):
        diff = np.nanmax(np.abs(aeri['secs'] - mwr['secs']))
        if diff > 0.00001:
            print('Error: The AERI and MWR times do not match')
            return success,-999.,-999.,-999.
    
    success = 0
    
    return success, aeri, mwr, mwrscan

################################################################################
# This function recenters the prior information
################################################################################

def recenter_prior(z0, p0, Xa, Sa, input_value, sfc_or_pwv=0, changeTmethod=0, verbose=1):
    """
    This code recenters the mean of the prior dataset.
    The water vapor profile is recentered first, using a height-independent scale factor determined
    from either the surfaceWVMR value or the PWV (selected using the sfc_or_pwv flag).
    The temperature profile is then recentered, using either the "conserve-RH" or
    "conserve-covariance" methods.
    Note that the uncertainty of the water vapor is also recentered, but not the temperature.
    :z: The vertical grid of the prior
    :p: The mean pressure profile of the prior
    :Xa: The mean profiles of [temperature,waterVapor] (also called [T,q])
    :Sa: The covariance matrix of [[TT,Tq],[qT,qq]]
    :param sfc_or_pwv: This keyword indicates the what the input_value represents:
                            0-> the default value, which forces the user to actually think!
                            1-> implies that the input_value is the surface WVMR [g/kg]
                            2-> implies that the input_value is the column PWV [cm]
    :param changeTmethod: This keyword indicates which method is used to recenter the temperature
                            0-> the default value, which forces the user to actually think!
                            1-> indicates that the conserve-RH method should be used
                            2-> indicates that the conserve-covariance method should be used
    :param verbose: This keyword indicates how noisy the routine should be
    :return: successFlag, newXa, newSa
            SuccessFlag is 1 if the prior was successfully scaled, 0 if the function failed.
            newXa is the new mean prior
            newSa is the new prior covariance matrix
    """

    if ((sfc_or_pwv < 1) | (sfc_or_pwv > 2)):
        print('Error: the sfc_or_pwv keyword has an undefined value (must be 1 or 2) -- see usage')
        return 0
    if ((changeTmethod < 1) | (changeTmethod > 2)):
        print('Error: the changeTmethod keyword has an undefined value (must be 1 or 2) -- see usage')
        return 0

    # Extract out the mean temperature and humidity profiles
    k    = len(z0)
    t0   = Xa[0:k]
    q0   = Xa[k:2*k]

    # Compute the correlation matrix from the prior's covariance
    sig   = np.sqrt(np.diag(Sa))
    sigT0 = sig[0:k]
    sigQ0 = sig[k:2*k]
    corM  = np.copy(Sa)
    for i in range(len(sig)):
        for j in range(len(sig)):
            corM[i,j] = Sa[i,j] / (sig[i] * sig[j])

    # Calculate the RH and PWV from this prior
    u0   = Calcs_Conversions.w2rh(q0, p0, t0, 0)
    pwv0 = Calcs_Conversions.w2pwv(q0, p0)

    # Scale the WV profile
    if sfc_or_pwv == 1:
        if(verbose >= 2):
            print('    Recenter_prior is using the scale-by-surface method')
        input_comment = f"surface WVMR value of {input_value:5.2f} g/kg"
        sf = input_value / q0[0]
        sfact_comment = f'The WVMR profile was scaled by a factor of {sf:5.2f}'
        q1    = q0 * sf
        sigQ1 = sigQ0 * sf

    elif sfc_or_pwv == 2:
        if(verbose >= 2):
            print('    Recenter_prior is using the scale-by-PWV method')
        input_comment = f"column PWV value of {input_value:5.2f} cm"
        sf = input_value / pwv0
        sfact_comment = f'The WVMR profile was scaled by a factor of {sf:5.2f}'
        q1    = q0 * sf
        sigQ1 = sigQ0 * sf

    else:
        print("Error with sfc_or_pwv: This should not happen within recenter_prior")
        return 0

    if(verbose >= 2):
        print(f'    {sfact_comment}')

    # Adjust the temperature depending on the method selected
    if changeTmethod == 1:
            # Now iterate to find the best temperature, preserving the RH profile in the original prior
        tmethod_comment = 'converve-RH method'
        t1 = np.full_like(t0, -999.)     # Allocate an empty array
        off = np.arange(4001)/50. - 40.  # An array of temperature offsets

        for i in range(len(z0)):
            tmp = Calcs_Conversions.w2rh(q1[i], p0[i], t0[i] + off)
            foo = np.argmin(np.abs(tmp - u0[i]))
            t1[i] = t0[i] + off[foo]
    elif changeTmethod == 2:
        tmethod_comment = 'converve-covariance method'
        covTQ = np.zeros((len(z0),len(z0)))
        covQQ = np.zeros((len(z0),len(z0)))
        for i in range(len(z0)):
            for j in range(len(z0)):
                covTQ[i,j] = Sa[i,len(z0)+j]
                covQQ[i,j] = Sa[len(z0)+i,len(z0)+j]
        sf2qqInv = scipy.linalg.pinv(sf*sf*covQQ)
        slope = (sf*covTQ).dot(sf2qqInv)
        t1 = t0 + slope.dot(q1-q0)
    else:
        print("Error with changeTmethod: This should not happen within recenter_prior")
        return 0

    # Now create the new mean prior and its covariance matrix
    newXa  = np.append(t1, q1)
    newSig = np.append(sigT0, sigQ1)
    newSa  = np.copy(Sa)
    for i in range(len(newSig)):
        for j in range(len(newSig)):
            newSa[i,j] = corM[i,j] * (newSig[i] * newSig[j])

    comments = {'Comment_on_recentering1': 'The WVMR profile prior recentered using a '+input_comment,
                'Comment_on_recentering2': sfact_comment,
                'Comment_on_recentering3': 'The temperature prior was recentered using the '+tmethod_comment
    }

    # Echo some output, indicating how the prior was rescaled
    if(verbose >= 1):
        print('    The prior dataset was recentered')
        for i in range(len(list(comments.keys()))):
            print('      '+list(comments.keys())[i]+': '+comments[list(comments.keys())[i]])

    return 1, newXa, newSa, comments
    

        
    
    
    