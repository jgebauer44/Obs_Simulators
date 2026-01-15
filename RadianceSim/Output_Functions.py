import os
import numpy as np
from netCDF4 import Dataset
import scipy.io
import glob
import sys
from datetime import datetime

import Other_functions
import Calcs_Conversions
import VIP_Databases_functions

###############################################################################
# This function will write the output for the radiance/brightness temperature
# simulations
###############################################################################

def write_output(namelist, output_dir, data1, data2, data3, model, model_time, s, s0,ymd):
    
    snum = int(s-s0)
    
    if namelist['use_calendar'] == 1:
        outfile_path = output_dir + '/' + namelist['outfile_root'] + '_' + str(ymd) +'_' + model_time[0].strftime('%H%M%S') + '.nc'
    else:
        outfile_path = output_dir + '/' + namelist['outfile_root'] + '_' + str(ymd) +'_' + str(int(model_time[0])) + '.nc'
                                                                      
    if ((namelist['append'] == 0) & (not os.path.exists(outfile_path))):
        
        fid = Dataset(outfile_path,'w')
        
        zz = fid.createDimension('height',len(model['z']))
        tdim = fid.createDimension('time',None)
        
        # Store the model variables
        
        bt = fid.createVariable('base_time','i4')
        bt.long_name = 'Base time'
        
        if namelist['use_calendar'] == 1:
            bt.units = 'Seconds since 1970-01-01'
            bt[:] = (model_time[0] - datetime(1970,1,1)).total_seconds()
        else:
            bt.units = 'Model start time'
            bt[:] = model_time[0]
        
        time = fid.createVariable('time','f4',('time',))
        time.long_name = 'Time since base time'
        time.units = 'sec'
        
        tt = fid.createVariable('model_temp', 'f4', ('time','height',))
        tt.long_name = 'Model temperature'
        tt.units = 'degC'
        
        ww = fid.createVariable('model_waterVapor', 'f4', ('time','height',))
        ww.long_name = 'Model mixing ratio'
        ww.units = 'g/kg'
        
        rrho = fid.createVariable('model_density', 'f4', ('time','height',))
        rrho.long_name = 'Model density'
        rrho.units = 'kg/m^3'
        
        pp = fid.createVariable('model_pressure', 'f4', ('time','height',))
        pp.long_name = 'Model pressure'
        pp.units = 'hPa'
        
        zzz = fid.createVariable('model_heights', 'f4', ('time','height',))
        zzz.long_name = 'Model height'
        zzz.units = 'km (AGL)'
        
        effc = fid.createVariable('cloud_reff', 'f4', ('time','height',))
        effc.long_name = 'Effective radius of liquid cloud drops'
        effc.units = 'um'
        
        effi = fid.createVariable('ice_reff', 'f4', ('time','height',))
        effi.long_name = 'Effective radius of snow and ice clouds'
        effi.units = 'um'
        
        taul = fid.createVariable('model_liquid_od', 'f4', ('time','height',))
        taul.long_name = 'Optical depth of liquid clouds'
        taul.units = 'unitless'
        
        taui = fid.createVariable('model_ice_od', 'f4', ('time','height',))
        taui.long_name = 'Optical depth of ice clouds'
        taui.units = 'unitless'
        
        psfc = fid.createVariable('model_psfc','f4', ('time',))
        psfc.long_name = 'model surface pressure'
        psfc.units = 'hPa'
        
        tsfc = fid.createVariable('model_tsfc', 'f4', ('time',))
        tsfc.long_name = 'model surface temperature'
        tsfc.units = 'C'
        
        wsfc = fid.createVariable('model_wsfc', 'f4', ('time',))
        wsfc.long_name = 'model surface mixing ratio'
        wsfc.units = 'g/kg'
        
        lwp = fid.createVariable('model_lwp', 'f4', ('time',))
        lwp.long_name = 'model liquid water path'
        lwp.units = 'g/m^2'
        
        precip = fid.createVariable('precip', 'f4', ('time',))
        precip.long_name = 'model reflectivity'
        precip.units = 'dBZ'
        
    
        cbh = fid.createVariable('model_cbh','f4',('time',))
        cbh.long_name = 'model cloud base height'
        cbh.units = 'km'
        
        cth = fid.createVariable('model_cth','f4',('time',))
        cth.long_name = 'model cloud top height'
        cth.units = 'km'
        
        if namelist['profiler_type'] == 1:
            wwnum = fid.createDimension('wnum',len(data2))
            
            wnum = fid.createVariable('wnum','f4',('wnum',))
            wnum.long_name = 'wavenumber'
            wnum.units = 'cm-1'
            wnum[:] = data2
        
            rad = fid.createVariable('rad','f4',('time','wnum',))
            rad.long_name = 'Downwelling radiance'
            rad.units = 'mw/(m2 sr cm-1)'
            
            noise = fid.createVariable('noise', 'f4', ('time','wnum',))
            noise.long_name = 'Radiance noise'
            noise.units = 'mw/(m2 sr cm-1)'
        
        elif namelist['profiler_type'] == 2:
            ffreq = fid.createDimension('freq',len(data2))
            
            freq = fid.createVariable('freq','f8',('freq',))
            freq.long_name = 'MWR channel frequency'
            freq.units = 'GHz'
            freq[:] = data2
        
            bt = fid.createVariable('brightness_temp','f4',('time','freq',))
            bt.long_name = 'MWR brightness temperature'
            bt.units = 'K'
            
            noise = fid.createVariable('noise', 'f4', ('time','freq',))
            noise.long_name = 'Brightness temperature noise'
            noise.units = 'K'
        
        elif namelist['profiler_type'] == 3:
            freqel = fid.createDimension('bt_dim',len(data2['freq']))
            
            freq = fid.createVariable('freq','f4',('bt_dim',))
            freq.long_name = 'MWR channel frequency'
            freq.units = 'GHz'
            freq[:] = data2['freq']
            
            el = fid.createVariable('elev','f4',('bt_dim',))
            el.long_name = 'MWR elevation'
            el.units = 'degree'
            el[:] = data2['elev']
            
            bt = fid.createVariable('brightness_temp','f4',('time','bt_dim',))
            bt.long_name = 'MWR brightness temperature'
            bt.units = 'K'
            
            noise = fid.createVariable('noise', 'f4', ('time','bt_dim',))
            noise.long_name = 'Brightness temperature noise'
            noise.units = 'K'
            
        fid.RadianceSim_version = '0.0.1'
        if namelist['model'] == 1:
            fid.model = 'WRF'
        if namelist['model'] == 5:
            fid.model = 'CM1'
        fid.profiler_code = namelist['profiler_type']
        if namelist['profiler_type'] == 1:
            fid.profiler_type = 'IRS'
        if namelist['profiler_type'] == 2:
            fid.profiler_type = 'MWR'
        if namelist['profiler_type'] == 3:
            fid.profiler_type = 'MWRscan'
        fid.profiler_x = namelist['profiler_x']
        fid.profiler_y = namelist['profiler_y']
        fid.profiler_alt = model['alt']
        fid.start_time = namelist['start_time']
        fid.end_time = namelist['end_time']
        fid.aeri_noise_file = namelist['aeri_noise_file']
        fid.lbl_version = namelist['lbl_version']
        fid.lbl_std_atmosphere = namelist['lbl_std_atmos']
        fid.monortm_version = namelist['monortm_version']
        fid.sspl_file = namelist['lcloud_ssp']
        fid.sspi_file = namelist['icloud_ssp']
        fid.ymd = ymd
        fid.sim_number = 0
        
        fid.close()
    
    # Append the data to the file
    
    fid = Dataset(outfile_path,'a')
    
    bt = fid.variables['base_time'][0]
    time = fid.variables['time']
    tt = fid.variables['model_temp']
    ww = fid.variables['model_waterVapor']
    rrho = fid.variables['model_density']
    pp = fid.variables['model_pressure']
    zzz = fid.variables['model_heights']
    effc = fid.variables['cloud_reff']
    effi = fid.variables['ice_reff']
    taul = fid.variables['model_liquid_od']
    taui = fid.variables['model_ice_od']
    psfc = fid.variables['model_psfc']
    tsfc = fid.variables['model_tsfc']
    wsfc = fid.variables['model_wsfc']
    lwp = fid.variables['model_lwp']
    precip = fid.variables['precip']
    cbh = fid.variables['model_cbh']
    cth = fid.variables['model_cth']
    
    if namelist['use_calendar'] == 1:
        time[snum] = (model_time[snum]-datetime(1970,1,1)).total_seconds() - bt
    else:
        time[snum] = model_time[snum] - bt
    
    tt[snum] = model['t']
    ww[snum] = model['w']
    rrho[snum] = model['rho']
    pp[snum] = model['p']
    zzz[snum] = model['z']
    effc[snum] = model['effc']
    effi[snum] = model['effi']
    taul[snum] = model['tauc']
    taui[snum] = model['taui']
    psfc[snum] = model['psfc']
    tsfc[snum] = model['tsfc']
    wsfc[snum] = model['wsfc']
    lwp[snum] = model['lwp']
    precip[snum] = model['precip']
    cbh[snum] = model['cbh']
    cth[snum] = model['cth']
    if namelist['profiler_type'] == 1:
        rad = fid.variables['rad']
        rad[snum] = data1
        noise = fid.variables['noise']
        noise[snum] = data3
    elif namelist['profiler_type'] >= 2:
        bt = fid.variables['brightness_temp']
        bt[snum] = data1
        noise = fid.variables['noise']
        noise[snum] = data3
    
    
    
    fid.sim_number = snum
    
    fid.close()
    
    return 1

################################################################################
# This function writes out netCDF files with the output
################################################################################

def write_output_tropoe(vip, ext_prof, mod_prof, rass_prof, ext_tseries, globatt, xret, prior,
                fsample, version, exectime, modeflag, nfilename, location,
                verbose):

    success = 0
    # I will replace all temp/WVMR data below the chimney height with this
    # flag
    nochim = -888.

    # These are the derived indices that I will compute later one. I need to
    # define them here in order to build the netcdf file correctly
    dindex_name = ['pwv', 'pblh', 'sbih', 'sbim', 'lcl']
    dindex_units = ['cm', 'km AGL', 'km AGL', 'C', 'km AGL']

    # If fsample is zero, then we will create the netCDF file
    if fsample == 0:
        dt = datetime.utcfromtimestamp(xret[0]['secs'])
        hh = datetime.utcfromtimestamp(xret[0]['secs']).hour
        nn = datetime.utcfromtimestamp(xret[0]['secs']).minute
        ss = datetime.utcfromtimestamp(xret[0]['secs']).second
        hms = hh*10000 + nn*100 + ss

        nfilename = vip['output_path'] + '/' + vip['output_rootname'] + '.' + dt.strftime('%Y%m%d.%H%M%S') + '.cdf'

        if ((os.path.exists(nfilename)) & (vip['output_clobber'] == 0)):
            print('Error: output file exists -- aborting (' + nfilename + ')')
        elif (os.path.exists(nfilename)):
            print('Warning: clobbering existing output file (' +nfilename + ')')

        fid = Dataset(nfilename, 'w')
        tdim = fid.createDimension('time', None)
        nht = len(xret[0]['z'])
        hdim = fid.createDimension('height', nht)
        vdim = fid.createDimension('obs_dim', len(xret[0]['dimY']))
        gdim = fid.createDimension('gas_dim', 3)
        ddim = fid.createDimension('dfs', len(xret[0]['dfs']))
        if vip['output_file_keep_small'] == 0:
            adim = fid.createDimension('arb', len(xret[0]['Xn']))
        idim = fid.createDimension('index_dim', len(dindex_name))

        base_time = fid.createVariable('base_time','i4')
        base_time.long_name = 'Epoch time'
        base_time.units = 's since 1970/01/01 00:00:00 UTC'

        time_offset = fid.createVariable('time_offset', 'f8', ('time',))
        time_offset.long_name = 'Time offset from base_time'
        time_offset.units = 's'

        hour = fid.createVariable('hour', 'f8', ('time',))
        hour.long_name = 'Time'
        hour.units = 'Hours from 00:00 UTC'

        qc_flag = fid.createVariable('qc_flag', 'i2', ('time',))
        qc_flag.long_name = 'Manual QC flag'
        qc_flag.units = 'unitless'
        qc_flag.comment = 'value of 0 implies quality is ok; non-zero values indicate that the sample has suspect quality'
        qc_flag.value_1 = 'Implies hatch was not open for full observing period'
        qc_flag.value_2 = 'Implies retrieval did not converge'
        qc_flag.value_3 = 'Implies retrieval converged but RMS between the observed and computed spectrum is too large'
        qc_flag.RMS_threshold_used_for_QC = str(vip['qc_rms_value']) + ' [unitless]'

        height = fid.createVariable('height', 'f4', ('height',))
        height.long_name = 'height'
        height.units = 'km AGL'

        temperature = fid.createVariable('temperature', 'f4', ('time','height',))
        temperature.long_name = 'temperature'
        temperature.units = 'C'

        waterVapor = fid.createVariable('waterVapor', 'f4', ('time', 'height',))
        waterVapor.long_name = 'water vapor mixing ratio'
        waterVapor.units = 'g/kg'

        lwp = fid.createVariable('lwp', 'f4', ('time',))
        lwp.long_name = 'liquid water path'
        lwp.units = 'g/m2'

        lReff = fid.createVariable('lReff', 'f4', ('time',))
        lReff.long_name = 'liquid water effective radius'
        lReff.units = 'microns'

        iTau = fid.createVariable('iTau', 'f4', ('time',))
        iTau.long_name = 'ice cloud optical depth (geometric limit)'
        iTau.units = 'unitless'

        iReff = fid.createVariable('iReff', 'f4', ('time',))
        iReff.long_name = 'ice effective radius'
        iReff.units = 'microns'

        co2 = fid.createVariable('co2', 'f4', ('time','gas_dim',))
        co2.long_name = 'carbon dioxide concentration'
        co2.units = 'ppm'

        ch4 = fid.createVariable('ch4', 'f4', ('time', 'gas_dim',))
        ch4.long_name = 'methane concentration'
        ch4.units = 'ppm'

        n2o = fid.createVariable('n2o', 'f4', ('time', 'gas_dim',))
        n2o.long_name = 'nitrous oxide concentration'
        n2o.units = 'ppm'

        sigmaT = fid.createVariable('sigma_temperature', 'f4', ('time','height',))
        sigmaT.long_name = '1-sigma uncertainty in temperature'
        sigmaT.units = 'C'

        sigmaWV = fid.createVariable('sigma_waterVapor', 'f4', ('time','height',))
        sigmaWV.long_name = '1-sigma uncertainty in water vapor mixing vapor'
        sigmaWV.units = 'g/kg'

        sigma_lwp = fid.createVariable('sigma_lwp', 'f4', ('time',))
        sigma_lwp.long_name = '1-sigma uncertainty in liquid water path'
        sigma_lwp.units = 'g/m2'

        sigma_lReff = fid.createVariable('sigma_lReff', 'f4', ('time',))
        sigma_lReff.long_name = '1-sigma uncertainty in liquid water effective radius'
        sigma_lReff.units = 'microns'

        sigma_iTau = fid.createVariable('sigma_iTau', 'f4', ('time',))
        sigma_iTau.long_name = '1-sigma uncertainty in ice cloud optical depth (geometric limit)'
        sigma_iTau.units = 'unitless'

        sigma_iReff = fid.createVariable('sigma_iReff', 'f4', ('time',))
        sigma_iReff.long_name = '1-sigma uncertainty in ice effective radius'
        sigma_iReff.units = 'microns'

        sigma_co2 = fid.createVariable('sigma_co2', 'f4', ('time','gas_dim',))
        sigma_co2.long_name = '1-sigma uncertainty in carbon dioxide concentration'
        sigma_co2.units = 'ppm'

        sigma_ch4 = fid.createVariable('sigma_ch4', 'f4', ('time','gas_dim',))
        sigma_ch4.long_name = '1-sigma uncertainty in methane concentration'
        sigma_ch4.units = 'ppm'

        sigma_n2o = fid.createVariable('sigma_n2o', 'f4', ('time','gas_dim',))
        sigma_n2o.long_name = '1-sigma uncertaintiy in nitrous oxide concentration'
        sigma_n2o.units = 'ppm'

        converged_flag = fid.createVariable('converged_flag', 'i2', ('time',))
        converged_flag.long_name = 'convergence flag'
        converged_flag.units = 'unitless'
        converged_flag.value_0 = '0 indicates no convergence'
        converged_flag.value_1 = '1 indicates convergence in Rodgers sense (i.e., di2m << dimY)'
        converged_flag.value_2 = '2 indicates convergence (best rms after rms increased drastically'
        converged_flag.value_3 = '3 indicates convergence (best rms after max_iter)'
        converged_flag.value_9 = '9 indicates found NaN in Xnp1'

        gamma = fid.createVariable('gamma', 'f4', ('time',))
        gamma.long_name = 'gamma parameter'
        gamma.units = 'unitless'

        n_iter = fid.createVariable('n_iter', 'i2', ('time',))
        n_iter.long_name = 'number of iterations performed'
        n_iter.units = 'unitless'

        rmsr = fid.createVariable('rmsr', 'f4', ('time',))
        rmsr.long_name = 'root mean square error between AERI and MWR obs in the observation vector and the forward calculation'
        rmsr.units = 'unitless'
        rmsr.comment1 = 'Computed as sqrt( sum_over_i[ ((Y_i - F(Xn_i)) / Y_i)^2 ] / sizeY)'
        rmsr.comment2 = 'Only AERI radiance observations in the observation vector are used'

        rmsa = fid.createVariable('rmsa', 'f4', ('time',))
        rmsa.long_name = 'root mean square error between observation vector and the forward calculation'
        rmsa.units = 'unitless'
        rmsa.comment1 = 'Computed as sqrt( sum_over_i[ ((Y_i - F(Xn_i)) / Y_i)^2 ] / sizeY)'
        rmsa.comment2 = 'Entire observation vector used in this calculation'

        rmsp = fid.createVariable('rmsp', 'f4', ('time',))
        rmsp.long_name = 'root mean square error between prior T/q profile and the retrieved T/q profile'
        rmsp.units = 'unitless'
        rmsp.comment1 = 'Computed as sqrt( mean[ ((Xa - Xn) / sigma_Xa)^2 ] )'

        chi2 = fid.createVariable('chi2', 'f4', ('time',))
        chi2.long_name = 'Chi-square statistic of Y vs. F(Xn)'
        chi2.units = 'unitless'
        chi2.comment = 'Computed as sqrt( sum_over_i[ ((Y_i - F(Xn_i)) / Y_i)^2 ] / sizeY)'

        convergence_criteria = fid.createVariable('convergence_criteria', 'f4', ('time',))
        convergence_criteria.long_name = 'convergence criteria di^2'
        convergence_criteria.units = 'unitless'

        dfs = fid.createVariable('dfs', 'f4', ('time','dfs',))
        dfs.long_name = 'degrees of freedom of signal'
        dfs.units = 'unitless'
        dfs.comment = 'total DFS, then DFS for each of temperature, waterVapor, LWP, L_Reff, I_tau, I_Reff, carbonDioxide, methane, nitrousOxide'

        sic = fid.createVariable('sic', 'f4', ('time',))
        sic.long_name = 'Shannon information content'
        sic.units = 'unitless'

        vres_temp = fid.createVariable('vres_temperature', 'f4', ('time','height',))
        vres_temp.long_name = 'Vertical resolution of the temperature profile'
        vres_temp.units = 'km'
        vres_wv = fid.createVariable('vres_waterVapor', 'f4', ('time','height',))
        vres_wv.long_name = 'Vertical resolution of the water vapor profile'
        vres_wv.units = 'km'

        cdfs_temp = fid.createVariable('cdfs_temperature', 'f4', ('time','height',))
        cdfs_temp.long_name = 'Vertical profile of the cumulative degrees of freedom of signal for temperature'
        cdfs_temp.units = 'unitless'
        cdfs_wv = fid.createVariable('cdfs_waterVapor', 'f4', ('time','height',))
        cdfs_wv.long_name = 'Vertical profile of the cumulative degrees of freedom of signal for water vapor'
        cdfs_wv.units = 'unitless'

        hatchOpen = fid.createVariable('hatchOpen', 'i2', ('time',))
        hatchOpen.long_name = 'Flag indicating if the AERIs hatch was open'
        hatchOpen.units = 'unitless'
        hatchOpen.comment = '1 - hatch open, 0 - hatch closed, other values indicate hatch is either not working or indeterminant'

        cbh = fid.createVariable('cbh', 'f4', ('time',))
        cbh.long_name = 'Cloud base height'
        cbh.units = 'km AGL'

        cbh_flag = fid.createVariable('cbh_flag', 'i2', ('time',))
        cbh_flag.long_name = 'Flag indicating the source of the cbh'
        cbh_flag.units = 'unitless'
        cbh_flag.comment1 = 'Value 0 implies Clear Sky radiance'
        cbh_flag.comment2 = 'Value 1 implies Inner Window radiance'
        cbh_flag.comment3 = 'Value 2 implies Outer Window radiance'
        cbh_flag.comment4 = 'Value 3 implies Default CBH radiance'

        pressure = fid.createVariable('pressure', 'f4', ('time','height',))
        pressure.long_name = 'derived pressure'
        pressure.units = 'mb'
        pressure.comment = 'derived from AERI surface pressure observations and the hyposmetric calculation using the thermodynamic profiles'

        theta = fid.createVariable('theta', 'f4', ('time','height',))
        theta.long_name = 'potential temperature'
        theta.units = 'K'
        theta.comment = 'This field is derived from the retrieved fields'

        thetae = fid.createVariable('thetae', 'f4', ('time','height',))
        thetae.long_name = 'equivalent potential temperature'
        thetae.units = 'K'
        thetae.comment = 'This field is derived from the retrieved fields'

        rh = fid.createVariable('rh', 'f4', ('time','height',))
        rh.long_name = 'relative humidity'
        rh.units = '%'
        rh.comment = 'This field is derived from the retrieved field'

        dewpt = fid.createVariable('dewpt', 'f4', ('time','height',))
        dewpt.long_name = 'dew point temperature'
        dewpt.units = 'C'
        dewpt.comment = 'This field is derived from the retrieved fields'

        dindices = fid.createVariable('dindices', 'f4', ('time','index_dim',))
        dindices.long_name = 'derived indices'
        dindices.units = 'units depends on the index; see comments below'
        dindices.comment0 = 'This field is derived from the retrieved fields'
        dindices.comment1 = 'A value of -999 indicates that this inded could not be computed (typically because the value was aphysical)'
        dindices.field_0_name = 'pwv'
        dindices.field_0_units = 'cm'
        dindices.field_1_name = 'pblh'
        dindices.field_1_units = 'km AGL'
        dindices.field_2_name = 'sbih'
        dindices.field_2_units = 'km AGL'
        dindices.field_3_name = 'sbim'
        dindices.field_3_units = 'C'
        dindices.field_4_name = 'lcl'
        dindices.field_4_units = 'km AGL'

        sigma_dindices = fid.createVariable('sigma_dindices', 'f4', ('time','index_dim',))
        sigma_dindices.long_name = '1-sigma uncertainties in the derived indices'
        sigma_dindices.units = 'units depend on the index, see the field above '
        sigma_dindices.comment1 = 'This field is derived fro mthe retrieved fields'
        sigma_dindices.comment2 = 'The uncertainties were determined using a monte carlo sampling of the posterior covariance matrix'
        sigma_dindices.comment3 = 'A value of -999 indicates that the uncertainty in this inded could not be computed (typically because the values were all unphysical)'

        obs_flag = fid.createVariable('obs_flag', 'i2', ('obs_dim',))
        obs_flag.long_name = 'Flag indicating type of observation for each vector element'
        obs_flag.units = 'mixed units -- see comments below'

        # This will make sure that I capture all of the units right in
        # the metadata, but "blotting them out" as I add the comments

        marker = np.copy(xret[0]['flagY'])
        foo = np.where(xret[0]['flagY'] == 1)[0]
        if len(foo) > 0:
            obs_flag.value_01 = 'cm^(-1) (i.e., wavenumber)'
            marker[foo] = -1

        foo = np.where(xret[0]['flagY'] == 2)[0]
        if len(foo) > 0:
            obs_flag.value_02 = 'Brightness temperature in K from a zenith-microwave radiometer'
            marker[foo] = -1

        foo = np.where(xret[0]['flagY'] == 3)[0]
        if len(foo) > 0:
            obs_flag.value_03 = 'Brightness temperature in K from a scanning microwave radiometer'
            obs_flag.value_03_comment1 = 'Dimension is coded to be (frequency[GHz]*100)+(elevation_angle[deg]/1000)'
            marker[foo] = -1

        foo = np.where(xret[0]['flagY'] == 4)[0]
        if len(foo) > 0:
            obs_flag.value_04 = 'Water vapor in ' + ext_prof['qunit'] + ' from ' + ext_prof['qtype']
            marker[foo] = -1

        # foo = np.where(xret[0]['flagY'] == 5)[0]
        # if len(foo) > 0:
        #     obs_flag.value_05 = 'Surface met temeprature in ' + ext_tseries['tunit'] + ' from ' + ext_tseries['ttype']
        #     obs_flag.value_05_comment1 = 'Surface met station is ' + str(ext_tseries['sfc_relative_height']) + ' m above height=0 level'
        #     if ext_tseries['sfc_temp_rep_error'] > 0:
        #         obs_flag.value_05_comment2 = 'Adding ' + str(ext_tseries['sfc_temp_rep_error']) + ' C to uncertainty to account for representativeness error'
        #     marker[foo] = -1

        # foo = np.where(xret[0]['flagY'] == 6)[0]
        # if len(foo) > 0:
        #     obs_flag.value_06 = 'Surface met water vapor in ' + ext_tseries['qunit'] + ' from ' + ext_tseries['qtype']
        #     obs_flag.value_06_comment1 = 'Surface met station is ' + str(ext_tseries['sfc_relative_height']) + ' m above height=0 level'
        #     if np.abs(ext_tseries['sfc_wv_mult_error']-1) > 0.01:
        #         obs_flag.value_06_comment2 = 'Multiplying by ' + ext_tseries['sfc_wv_mult_error']
        #     if ext_tseries['sfc_wv_rep_error'] > 0:
        #         obs_flag.value_06_comment2 = 'Adding ' + str(ext_tseries['sfc_wv_rep_error']) + ' g/kg to uncertainty to account for representativeness error'
            marker[foo] = -1

        foo = np.where(xret[0]['flagY'] == 7)[0]
        if len(foo) > 0:
            obs_flag.value_07 = 'Temperature in ' + mod_prof['tunit'] + ' from ' + mod_prof['ttype']
            marker[foo] = -1

        foo = np.where(xret[0]['flagY'] == 8)[0]
        if len(foo) > 0:
            obs_flag.value_08 = 'Water vapor in ' + mod_prof['qunit'] + ' from '  + mod_prof['qtype']
            marker[foo] = -1

        foo = np.where(xret[0]['flagY'] == 9)[0]
        if len(foo) > 0:
            obs_flag.value_09 = 'CO2 in-situ obs in ' + ext_tseries['co2unit'] + ' from ' + ext_tseries['co2type']
            if ext_tseries['co2_sfc_rep_error'] > 0:
                obs_flag.value_09_comment1 = 'Adding ' + ext_tseries['co2_sfc_rep_error'] + ' ppm to uncertainty to account for representativeness error'
            marker[foo] = -1

        foo = np.where(xret[0]['flagY'] == 10)[0]
        if len(foo) > 0:
            obs_flag.value_10 = 'Brightness temperature in K from a scanning microwave radiometer'
            obs_flag.value_10_comment1 = 'Dimension is coded to be (frequency[GHz]*100)+(elevation_angle[deg]/1000)'
            marker[foo] = -1

        foo = np.where(xret[0]['flagY'] == 11)[0]
        if len(foo) > 0:
            obs_flag.value_11 = 'Virtual temperature in ' + rass_prof['tunit'] + ' from ' + rass_prof['ttype']
            marker[foo] = -1

        # If there were any values for marker that were not treated above,
        # then the code must assume that I've screwed up and should abort.

        # foo = np.where(marker >= 0)[0]
        # if len(foo) > 0:
        #     print('Error in write_output: there seems to be a unit that is not handled here properly')
        #     return success, nfilename

        obs_dim = fid.createVariable('obs_dim', 'f8', ('obs_dim',))
        obs_dim.long_name = 'Dimension of the observation vector'
        obs_dim.units = 'mixed units -- see obs_flag field above'

        obs_vector = fid.createVariable('obs_vector', 'f4', ('time','obs_dim',))
        obs_vector.long_name = 'Observation vector Y'
        obs_vector.units = 'mixed units -- see obs_flag field above'

        obs_vector_uncertainty = fid.createVariable('obs_vector_uncertainty', 'f4', ('time','obs_dim',))
        obs_vector_uncertainty.long_name = '1-sigma uncertainty in the observation vector (sigY)'
        obs_vector_uncertainty.units = 'mixed units -- see obs_flag field above'

        forward_calc = fid.createVariable('forward_calc', 'f4', ('time','obs_dim',))
        forward_calc.long_name = 'Forward calculation from state vector (i.e., F(Xn))'
        forward_calc.units = 'mixed units -- see obs_flag field above'

        # If we are trying to keep the output file small, then do not include
        # these fields in the output file
        if vip['output_file_keep_small'] == 0:
            arb = fid.createVariable('arb', 'i2', ('arb',))
            arb.long_name = 'arbitrary dimension'
            arb.units = 'mixed units'
            arb.comment = ('contains temeprature profile (1), water vapor profile (2)'
                       + ' liquid cloud path (3), liquid water Reff (4), '
                       + 'ice cloud optical depth (5), ice cloud Reff (6), carbon dioxide (7)'
                       + ' methane (8), nitrous oxide (9)')

            Xop = fid.createVariable('Xop', 'f4', ('time','arb',))
            Xop.long_name = 'optimal solution'
            Xop.units = 'mixed units -- see field arb above'

            Sop = fid.createVariable('Sop', 'f4', ('time','arb','arb',))
            Sop.long_name = 'covariance matrix of the solution'
            Sop.units = 'mixed units -- see field arb above'

            Akernal = fid.createVariable('Akernal', 'f4', ('time','arb','arb',))
            Akernal.long_name = 'averaging kernal'
            Akernal.units = 'mixed units -- see field arb above'

            Xa = fid.createVariable('Xa', 'f4', ('arb',))
            Xa.long_name = 'prior mean state'
            Xa.units = 'mixed units -- see field arb above'

            Sa = fid.createVariable('Sa', 'f4', ('arb','arb',))
            Sa.long_name = 'prior covariance'
            Sa.units = 'mixed units -- see field arb above'

        # These should be the last three variables in the file
        lat = fid.createVariable('lat', 'f4')
        lat.long_name = 'latitude'
        lat.units = 'degrees north'

        lon = fid.createVariable('lon', 'f4')
        lon.long_name = 'longitude'
        lon.units = 'degrees east'

        alt = fid.createVariable('alt', 'f4')
        alt.long_name = 'altitude'
        alt.units = 'm above MSL'

        # Add some global attributes
        for i in range(len(list(globatt.keys()))):
            fid.setncattr(list(globatt.keys())[i], globatt[list(globatt.keys())[i]])
        fid.Algorithm_version = version
        fid.Prior_dataset_comment = prior['comment']
        fid.Prior_dataset_filename = prior['filename']
        fid.Prior_dataset_number_profiles = prior['nsonde']
        fid.Prior_dataset_T_inflation_factor = str(vip['prior_t_ival']) + ' at the surface to 1.0 at ' + str(vip['prior_t_iht']) + ' km AGL'
        fid.Prior_dataset_Q_inflation_factor = str(vip['prior_q_ival']) + ' at the surface to 1.0 at ' + str(vip['prior_q_iht']) + ' km AGL'
        fid.Prior_dataset_TQ_correlation_reduction_factor = vip['prior_tq_cov_val']
        fid.Total_clock_execution_time_in_s = exectime
        fid.Retrieval_option_flags = '{:0d}, {:0d}, {:0d}, {:0d}, {:0d}, {:0d}, {:0d}, {:0d}, {:0d}'.format(modeflag[0], modeflag[1], modeflag[2], modeflag[3], modeflag[4], modeflag[5], modeflag[6], modeflag[7], modeflag[8])
        fid.vip_tres = (str(vip['tres']) + ' minutes. Note that the sample time corresponds to the '
                      + 'center of the averaging intervale. A value of 0 implies that no averaging was performed')
        #add_vip_to_global_atts(vip)

        # Add some of the static (non-time-dependent) data
        base_time[:] = xret[0]['secs']
        height[:] = xret[0]['z']
        obs_flag[:] = xret[0]['flagY']
        obs_dim[:] = xret[0]['dimY']

        ones = np.ones(nht)
        twos = np.ones(nht)*2
        tmp = np.append(ones,twos)
        tmp = np.append(tmp, np.array([3,4,5,6,7,7,7,8,8,8,9,9,9]))

        if vip['output_file_keep_small'] == 0:
            arb[:] = tmp
            Xa[:] = prior['Xa']
            Sa[:,:] = prior['Sa']

        # Fill these in, if available
        if type(location) is dict:
            if ((type(location['lat']) is int) or (type(location['lat'] is float))):
                try:
                    lat[:] = np.float(location['lat'])
                except TypeError:
                    # TODO - Allow changing lat/lon/alt. I.e. let the lat/lon/alt have a time dimension
                    lat[:] = np.float(np.mean(location['lat']))
            else:
                lat[:] = -999.0

            if ((type(location['lon']) is int) or (type(location['lon'] is float))):
                try:
                    lon[:] = np.float(location['lon'])
                except TypeError:
                    lon[:] = np.float(np.mean(location['lon']))
            else:
                lon[:] = -999.0

            if ((type(location['alt']) is int) or (type(location['alt'] is float))):
                try:
                    alt[:] = np.float(location['alt'])
                except TypeError:
                    alt[:] = np.float(np.mean(location['alt']))
            else:
                alt[:] = -999.0

        else:
            lat[:] = -999.
            lon[:] = -999.
            alt[:] = -999.

        fid.close()

    # I am also storing some derived fields, so compute them here
    # First, the profiles. Note I am NOT going to provide uncertainty
    # values for these
    nht = len(xret[0]['z'])
    theta_tmp = np.zeros((nht,len(xret)))
    thetae_tmp = np.zeros((nht,len(xret)))
    rh_tmp = np.zeros((nht,len(xret)))
    dewpt_tmp = np.zeros((nht,len(xret)))
    temp_tmp = np.zeros((nht,len(xret)))
    wvmr_tmp = np.zeros((nht,len(xret)))
    stemp_tmp = np.zeros((nht,len(xret)))
    swvmr_tmp = np.zeros((nht,len(xret)))
    for i in range(len(xret)):
        temp_tmp[:,i] = np.copy(xret[i]['Xn'][0:nht])
        wvmr_tmp[:,i] = np.copy(xret[i]['Xn'][nht:2*nht])
        sig = np.sqrt(np.diag(xret[i]['Sop']))
        stemp_tmp[:,i] = np.copy(sig[0:nht])
        swvmr_tmp[:,i] = np.copy(sig[nht:2*nht])
        theta_tmp[:,i] = Calcs_Conversions.t2theta(xret[i]['Xn'][0:nht], 0*xret[i]['Xn'][nht:2*nht], xret[i]['p'])
        thetae_tmp[:,i] = Calcs_Conversions.t2thetae(xret[i]['Xn'][0:nht], xret[i]['Xn'][nht:2*nht], xret[i]['p'])
        rh_tmp[:,i] = Calcs_Conversions.w2rh(xret[i]['Xn'][nht:2*nht], xret[i]['p'], xret[i]['Xn'][0:nht],0) * 100
        dewpt_tmp[:,i] = Calcs_Conversions.rh2dpt(xret[i]['Xn'][0:nht], rh_tmp[:,i]/100.)

    foo = np.where(xret[0]['z'] < vip['prior_chimney_ht'])[0]
    if len(foo) > 0:
        temp_tmp[foo,:] = nochim
        wvmr_tmp[foo,:] = nochim
        stemp_tmp[foo,:] = nochim
        swvmr_tmp[foo,:] = nochim
        theta_tmp[foo,:] = nochim
        thetae_tmp[foo,:] = nochim
        dewpt_tmp[foo,:] = nochim
        rh[foo,:] = nochim

    # Now the derived indices. I am going to perform a simple monte carlo
    # sampling here to derive some sense of the uncertainties in these indices.
    # Note that even thought the uncertainties might not be Gaussian distributed,
    # I am going to report a 1-sigma standard deviation

    num_mc = 20                      # Number of points to use in the MC sampling
    npts = len(xret) - fsample       # Number of times that need processed

    # The derived indices
    indices = np.zeros((len(dindex_name), npts))
    sigma_indices = np.zeros((len(dindex_name), npts))
    tmp = np.zeros(num_mc)
    tprofs = np.zeros((nht, num_mc))
    wprofs = np.zeros((nht, num_mc))
    zz = np.copy(xret[0]['z'])

    if len(dindex_name) != len(dindex_units):
        print('Error in write_output: there is a dimension mismatch in the derived indices dindex_')
        return success, nfilename

    for i in range(npts):
        # Extract out the temperature and water vapor profiles
        pp = np.copy(xret[i+fsample]['p'])
        tt = np.copy(xret[i+fsample]['Xn'][0:nht])
        ww = np.copy(xret[i+fsample]['Xn'][nht:2*nht])

        # Extract out the apriori temperature and water vapor profiles
        ta = np.copy(prior['Xa'][0:nht])
        wa = np.copy(prior['Xa'][nht:2*nht])

        # Extract out the posterior covariance matrix
        Sop_tmp = np.copy(xret[i]['Sop'])
        Sop_tmp = Sop_tmp[0:2*nht,0:2*nht]
        sig_t = np.sqrt(np.diag(Sop_tmp))[0:nht]

        # Perform SVD of posterior covariance matrix
        # Note that in order to follow the logic of the original IDL code I
        # have to do the SVD on the transpose of the posterior covariance matrix
        # since IDL is a column major language

        u, w, v = scipy.linalg.svd(Sop_tmp.T, False)

        b = np.zeros((2*nht,num_mc))
        for j in range(num_mc):
            b[:,j] = np.random.normal(size = 2*nht)
        pert = u.dot(np.diag(np.sqrt(w))).dot(b)
        tprofs = tt[:,None] + pert[0:nht,:]
        wprofs = ww[:,None] + pert[nht:2*nht,:]

        # Now compute the indices and their uncertainties
        for ii in range(len(dindex_name)):
            if dindex_name[ii] == 'pwv':
                indices[ii,i] = Calcs_Conversions.w2pwv(ww,pp)
                for j in range(num_mc):
                    tmp[j] = Calcs_Conversions.w2pwv(wprofs[:,j], pp)
                sigma_indices[ii,i] = np.nanstd(indices[ii,i] - tmp)

            elif dindex_name[ii] == 'pblh':
                minht = vip['min_PBL_height']
                maxht = vip['max_PBL_height']
                nudge = vip['nudge_PBL_height']
                indices[ii,i] = Other_functions.compute_pblh(zz, tt, pp, sig_t, minht=minht, maxht=maxht, nudge=nudge)
                for j in range(num_mc):
                    tmp[j] = Other_functions.compute_pblh(zz, tprofs[:, j], pp, sig_t, minht=minht, maxht=maxht, nudge=nudge)
                foo = np.where(tmp > 0)[0]
                if (len(foo) > 1) & (indices[ii,i] > 0):
                    sigma_indices[ii,i] = np.nanstd(indices[ii,i] - tmp[foo])
                else:
                    sigma_indices[ii,i] = -999.
                if (sigma_indices[ii,i] < vip['min_PBL_height']) & (indices[ii,i] <= vip['min_PBL_height']):
                    sigma_indices[ii,i] = vip['min_PBL_height']

            elif dindex_name[ii] == 'sbih':
                indices[ii,i] = Other_functions.compute_sbi(zz,tt)['sbih']
                for j in range(num_mc):
                    tmp[j] = Other_functions.compute_sbi(zz,tprofs[:,j])['sbih']
                foo = np.where(tmp > 0)[0]
                if ((len(foo) > 1) & (indices[ii,i] > 0)):
                    sigma_indices[ii,i] = np.nanstd(indices[ii,i] - tmp[foo])
                else:
                    sigma_indices[ii,i] = -999.

            elif dindex_name[ii] == 'sbim':
                indices[ii,i] = Other_functions.compute_sbi(zz,tt)['sbim']
                for j in range(num_mc):
                    tmp[j] = Other_functions.compute_sbi(zz,tprofs[:,j])['sbim']
                foo = np.where(tmp > 0)[0]
                if ((len(foo) > 1) & (indices[ii,i] > 0)):
                    sigma_indices[ii,i] = np.nanstd(indices[ii,i] - tmp[foo])
                else:
                    sigma_indices[ii,i] = -999.

            elif dindex_name[ii] == 'lcl':
                indices[ii,i] = Other_functions.compute_lcl(tt[0],ww[0],pp[0],pp,zz)
                for j in range(num_mc):
                    tmp[j] = Other_functions.compute_lcl(tprofs[0,j], wprofs[0,j], pp[0], pp, zz)
                    sigma_indices[ii,i] = np.nanstd(indices[ii,i] - tmp)

            else:
                print('WARNING: There is some derive index that is not properly being computed in aerioe.py')


    # Now append all of the samples from fsample onward into the file
    if verbose >= 3:
        print('Appending data to ' + nfilename)
    fid = Dataset(nfilename, 'a')
    fid.Total_clock_execution_time_in_s = str(exectime)

    time_offset = fid.variables['time_offset']
    hour = fid.variables['hour']
    qc_flag = fid.variables['qc_flag']

    temperature = fid.variables['temperature']
    waterVapor = fid.variables['waterVapor']
    lwp = fid.variables['lwp']
    lReff = fid.variables['lReff']
    iTau = fid.variables['iTau']
    iReff = fid.variables['iReff']
    co2 = fid.variables['co2']
    ch4 = fid.variables['ch4']
    n2o = fid.variables['n2o']

    sigmaT = fid.variables['sigma_temperature']
    sigmaWV = fid.variables['sigma_waterVapor']
    sigma_lwp = fid.variables['sigma_lwp']
    sigma_lReff = fid.variables['sigma_lReff']
    sigma_iTau = fid.variables['sigma_iTau']
    sigma_iReff = fid.variables['sigma_iReff']
    sigma_co2 = fid.variables['sigma_co2']
    sigma_ch4 = fid.variables['sigma_ch4']
    sigma_n2o = fid.variables['sigma_n2o']

    converged_flag = fid.variables['converged_flag']
    gamma = fid.variables['gamma']
    n_iter = fid.variables['n_iter']
    rmsr = fid.variables['rmsr']
    rmsa = fid.variables['rmsa']
    rmsp = fid.variables['rmsp']
    chi2 = fid.variables['chi2']
    convergence_criteria = fid.variables['convergence_criteria']
    dfs = fid.variables['dfs']
    sic = fid.variables['sic']
    vres_temp = fid.variables['vres_temperature']
    vres_wv = fid.variables['vres_waterVapor']
    cdfs_temp = fid.variables['cdfs_temperature']
    cdfs_wv = fid.variables['cdfs_waterVapor']

    hatchOpen = fid.variables['hatchOpen']
    cbh = fid.variables['cbh']
    cbh_flag = fid.variables['cbh_flag']
    pressure = fid.variables['pressure']

    theta = fid.variables['theta']
    thetae = fid.variables['thetae']
    rh = fid.variables['rh']
    dewpt = fid.variables['dewpt']
    dindices = fid.variables['dindices']
    sigma_dindices = fid.variables['sigma_dindices']

    obs_vector = fid.variables['obs_vector']
    obs_vector_uncertainty = fid.variables['obs_vector_uncertainty']
    forward_calc = fid.variables['forward_calc']

    if vip['output_file_keep_small'] == 0:
        Xop = fid.variables['Xop']
        Sop = fid.variables['Sop']
        Akernal = fid.variables['Akernal']

    npts = len(xret) - fsample
    basetime = fid.variables['base_time'][:]
    for i in range(npts):
        time_offset[fsample+i] = xret[fsample+i]['secs'] - basetime
        hour[fsample+i] = xret[fsample+i]['hour']
        qc_flag[fsample+i] = xret[fsample+i]['qcflag']

        did = np.where(np.array(list(fid.dimensions.keys())) == 'height')[0]
        if len(did) == 0:
            print('Whoaa -- this should not happen -- aborting')
            return success, nfilename

        if fid.dimensions['height'].size != len(xret[0]['z']):
            print('Whoaa -- this should not happen size -- aborting')
            return success, nfilename

        temperature[fsample+i,:] = xret[fsample+i]['Xn'][0:nht]
        waterVapor[fsample+i,:] = xret[fsample+i]['Xn'][nht:2*nht]
        lwp[fsample+i] = xret[fsample+i]['Xn'][2*nht]
        lReff[fsample+i] = xret[fsample+i]['Xn'][2*nht+1]
        iTau[fsample+i] = xret[fsample+i]['Xn'][2*nht+2]
        iReff[fsample+i] = xret[fsample+i]['Xn'][2*nht+3]
        co2[fsample+i,:] = xret[fsample+i]['Xn'][2*nht+4:2*nht+7]
        ch4[fsample+i,:] = xret[fsample+i]['Xn'][2*nht+7:2*nht+10]
        n2o[fsample+i,:] = xret[fsample+i]['Xn'][2*nht+10:2*nht+13]

        sigmaT[fsample+i,:] = stemp_tmp[:,fsample+i]
        sigmaWV[fsample+i,:] = swvmr_tmp[:,fsample+i]
        sig = np.sqrt(np.diag(xret[fsample+i]['Sop']))
        sigma_lwp[fsample+i] = sig[2*nht]
        sigma_lReff[fsample+i] = sig[2*nht+1]
        sigma_iTau[fsample+i] = sig[2*nht+2]
        sigma_iReff[fsample+i] = sig[2*nht+3]
        sigma_co2[fsample+i,:] = sig[2*nht+4:2*nht+7]
        sigma_ch4[fsample+i,:] = sig[2*nht+7:2*nht+10]
        sigma_n2o[fsample+i,:] = sig[2*nht+10:2*nht+13]

        converged_flag[fsample+i] = xret[fsample+i]['converged']
        gamma[fsample+i] = xret[fsample+i]['gamma']
        n_iter[fsample+i] = xret[fsample+i]['niter']
        rmsr[fsample+i] = xret[fsample+i]['rmsr']
        rmsa[fsample+i] = xret[fsample+i]['rmsa']
        rmsp[fsample+i] = xret[fsample+i]['rmsp']
        chi2[fsample+i] = xret[fsample+i]['chi2']
        convergence_criteria[fsample+i] = xret[fsample+i]['di2m']
        dfs[fsample+i,:] = xret[fsample+i]['dfs']
        sic[fsample+i] = xret[fsample+i]['sic']
        vres_temp[fsample+i,:] = xret[fsample+i]['vres'][0,:]
        vres_wv[fsample+i,:] = xret[fsample+i]['vres'][1,:]
        cdfs_temp[fsample+i,:] = xret[fsample+i]['cdfs'][0,:]
        cdfs_wv[fsample+i,:] = xret[fsample+i]['cdfs'][1,:]

        hatchOpen[fsample+i] = xret[fsample+i]['hatchopen']
        cbh[fsample+i] = xret[fsample+i]['cbh']
        cbh_flag[fsample+i] = xret[fsample+i]['cbhflag']
        pressure[fsample+i,:] = xret[fsample+i]['p'][0:nht]

        theta[fsample+i,:] = np.transpose(theta_tmp[:,fsample+i])
        thetae[fsample+i,:] = np.transpose(thetae_tmp[:,fsample+i])
        rh[fsample+i,:] = np.transpose(rh_tmp[:,fsample+i])
        dewpt[fsample+i,:] = np.transpose(dewpt_tmp[:,fsample+i])
        dindices[fsample+i,:] = np.transpose(indices[:,i])
        sigma_dindices[fsample+i,:] = np.transpose(sigma_indices[:,i])

        obs_vector[fsample+i,:] = xret[fsample+i]['Y']
        obs_vector_uncertainty[fsample+i,:] = xret[fsample+i]['sigY']
        forward_calc[fsample+i,:] = xret[fsample+i]['FXn']

        if vip['output_file_keep_small'] == 0:
            Xop[fsample+i,:] = xret[fsample+i]['Xn']
            Sop[fsample+i,:,:] = xret[fsample+i]['Sop']
            Akernal[fsample+i,:,:] = xret[fsample+i]['Akern']

    fid.close()
    success = 1

    return success, nfilename
    
################################################################################
# This function creates a "simulated" output structure needed for running the
# code in the "output_clobber eq 2" (append) mode. The number of fields, the
# dimensions and types of each of the fields, needs to be correct. Only the
# "secs" field needs to have the proper values though...
################################################################################

def create_xret(xret, fsample, vip, aeri, Xa, Sa, z, bands, obsdim, obsflag):

    # Find all of the output files with this date
    ymd = aeri['date']

    files = []
    filename = vip['output_path'] + '/' + vip['output_rootname'] + '.' + str(ymd[0]) + '.*.cdf'
    files = files + (glob.glob(filename))

    # If none are found, then just run code as normal. Note that xret and fsample
    if len(files) == 0:
        print('The flag output_clobber was set to 2 for append, but no prior file was found')
        print('    so code will run as normal')
        nfilename = ' '
        return xret, fsample, nfilename

    # Otherwise, let's initialize from the last file
    nfilename = files[len(files)-1]
    fid = Dataset(nfilename, 'r')
    bt = fid.variables['base_time'][:]
    to = fid.variables['time_offset'][:]
    xobsdim = fid.variables['obs_dim'][:]
    xobsflag = fid.variables['obs_flag'][:]
    xz = fid.variables['height'][:]
    xXa = fid.variables['Xa'][:]
    xSa = fid.variables['Sa'][:]
    fid.close()

    secs = bt+to

    # Set up some default values
    wnum = aeri['wnum']
    nY = len(xobsflag)
    Y = np.zeros(nY)
    Sop = np.diag(np.ones(len(Xa)))
    Kij = np.zeros((nY,len(Xa)))
    Gain = np.zeros((len(Xa),nY))
    Akern = np.copy(Sop)
    vres = np.zeros((2,len(z)))
    cdfs = np.zeros((2,len(z)))
    dfs = np.zeros(16)
    sic = 0.0

    # A few very basic checks to make sure that some of the variables in the
    # output file match the current ones (e.g., height, wavenumbers, etc)
    diff = np.abs(z-xz)
    foo = np.where(diff > 0.001)[0]
    if ((len(xz) != len(z)) | (len(foo) > 0)):
        print('Error: output_clobber is set to 2 (append), but there is a mismatch in heights')
        fsample = -1
        return xret, fsample, nfilename

    diff = np.abs(Xa-xXa)

    foo = np.where(diff > 0.001)[0]
    if ((len(Xa) != len(xXa)) | (len(foo) > 0)):
        print('Error: output_clobber is set to 2 (append), but there is a mismatch in Xa')
        fsample = -1
        return xret, fsample, nfilename

    diff = np.abs(obsdim - xobsdim)
    foo = np.where(diff > 0.001)[0]
    if ((len(obsdim) != len(xobsdim)) | (len(foo) > 0)):
        print('Error: output_clobber is set to 2 (append), but there is a mismatch in obs_dim')
        fsample = -1
        return xret, fsample, nfilename

    diff = np.abs(obsflag - xobsflag)
    foo = np.where(diff > 0.001)[0]
    if ((len(obsflag) != len(xobsflag)) | (len(foo) > 0)):
        print('Error: output_clobber is set to 2 (append), but there is a mismatch in obs_flag')
        fsample = -1
        return xret, fsample, nfilename

    # This structure must match that at the end of the main iteration loop
    # where the retrieval is performed. The values can be anything, with
    # the exception of the "secs" field

    xtmp = {'idx':0, 'secs':0.0, 'ymd':ymd[0], 'hour':0.0,
             'nX':len(Xa), 'nY':len(obsflag),
             'dimY':np.zeros(len(obsflag)),
             'Y':np.zeros(len(obsflag)),
             'sigY':np.zeros(len(obsflag)),
             'flagY':np.zeros(len(obsflag)), 'nitr':0,
             'z':z, 'p':np.zeros(len(z)), 'hatchopen':0,
             'cbh':0., 'cbhflag':0,
             'x0':Xa*0, 'Xn':Xa*0, 'Fxn':Y*0, 'Sop':Sop, 'K':Kij, 'Gain':Gain, 'Akern':Akern,
             'vres':vres, 'cdfs':cdfs, 'gamma':0., 'qcflag':0, 'sic':sic, 'dfs':dfs, 'di2m':0.,
             'rmsa':0., 'rmsr':0., 'rmsp':0., 'chi2':0., 'converged':0}

    xret = []
    xret.append(xtmp)
    xret[0]['secs'] = secs[0]
    for i in range(1,len(secs)):
        xret.append(xtmp)
        xret[i]['secs'] = secs[i]
    fsample = len(secs)

    return xret, fsample, nfilename      
        
        
        
            
        
        
            
        
        
        
        
