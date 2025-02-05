#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 12:47:48 2023

@author: joshua.gebauer
"""

import os
import sys
import numpy as np
import pyproj
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator
from argparse import ArgumentParser
from datetime import datetime, timedelta

def q2rh(q, p, t, ice=0):
    
    e = q2e(q, p)
    es = esat(t, ice) / 100.0          # To convert it to the correct units (mb)
    rh = e / es
    
    return rh*100

def q2e(q,p):
    qq = q/1000.                # Convert g/kg to g/g as a temporary variable
    w = qq/(1-qq)
    e = p * w / (0.622 + w)
    return e

def e2dew(e):
    
    dew = 243.5*np.log(e/6.112)/(17.67-np.log(e/6.112))
    return dew

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

def calc_dewpoint_error(temp,rh):
    
    rh_error = 0.03
    t_error = 0.5
    rh_min = 0.2
    rh_max = 1.00
    delta_rh = 0.01
    delta_t = 0.1
    
    if rh < rh_min:
        rh_tmp = rh_min
    else:
        rh_tmp = rh
    
    t1 = temp-delta_t
    t2 = temp+delta_t
    
    rh1 = rh_tmp-delta_rh
    rh2 = rh_tmp+delta_rh
    
    if rh1 < rh_min:
        rh1 = rh_min
    
    if rh2 > rh_max:
        rh2 = rh_max
        
    td_deriv_rh = ( rh2dpt(temp,rh2)-rh2dpt(temp,rh1) )/(rh2-rh1)
    td_deriv_t = ( rh2dpt(t1,rh_tmp)-rh2dpt(t2,rh_tmp) )/(t2-t1)
    
    td_error = np.sqrt(td_deriv_t*td_deriv_t*t_error*t_error +
                       td_deriv_rh*td_deriv_rh*rh_error*rh_error)
    
    return td_error

def read_namelist(filename):
    
    # This large structure will hold all of the namelist option. We are
    # predefining it so that default values will be used if they are not listed
    # in the namelist. The exception to this is if the user does not specify 
    # the model. In this scenerio the success flag will terminate the program.
    
    
    namelist = ({'success':0,
                 'model':0,              # Type of model used for the simulation. 1-WRF 5-CM1 (only one that works right now)
                 'model_frequency':0.,    # Frequency of the model output used for the simulation
                 'model_dir':'None',        # Directory with the model data
                 'model_prefix':'None',     # Prefix for the model data files
                 'outfile_root':'None',
                 'coordinate_type':2,             # 1-Lat/Lon, 2-x,y,z (only works with 2 for now)
                 'use_calendar':0,                # If 1 then the start and end times are defined by calendar If 0 they are in model integration time
                 'start_year':0,                  # Ignored if use_calendar is 0
                 'start_month':0,                 # Ignored if use_calendar is 0
                 'start_day':0,                   # Ignored if use_calendar is 0
                 'start_hour':0,                  # Ignored if use_calendar is 0
                 'start_min':0,                   # Ignored if use_calendar is 0
                 'start_sec':0,                   # Ignored if use_calendar is 0
                 'end_year':0,                    #Ignored is use_calendar is 0
                 'end_month':0,                   # Ignored if use_calendar is 0
                 'end_day':0,                     # Ignored if use_calendar is 0
                 'end_hour':0,                    # Ignored if use_calendar is 0
                 'end_min':0,                     # Ignored if use_calendar is 0
                 'end_sec':0,                     # Ignored if use_calendar is 0
                 'start_time':0.0,                # Start time of the profiler simulation (Ignored if used calendar is 1)
                 'end_time':86400.0,              # End time of the profiler simulation (Ignored if used calendar is 1)
                 'station_file':'None',              # Scan file that contains the elevations for the radar scans
                 'temp_error':1.0,                  # 1-sigma error in temperature in Kelvin
                 'rh_error':0.5,                    # 1-sigma rh error
                 'psfc_error':1.0,                   # 1-sigma surface pressure error
                 'wind_error':3.0,                   # 1-sigma error in wind components
                 'append':1,                       # =0 - don't append to file, 1 - append to file
                 'clobber':0                      # 0 - don't clobber file, 1 - clobber file (ignored if append = 1)
                 })
    
    if os.path.exists(filename):
        print('Reading the namelist: ' + filename)
        
        try:
            inputt = np.genfromtxt(filename, dtype=str, comments ='#',delimiter='=', autostrip=True)
        except:
            print ('ERROR: There was a problem reading the namelist')
            return namelist
    
    else:
        print('ERROR: The namelist file ' + filename + ' does not exist')
        return namelist
    
    if len(inputt) == 0:
        print('ERROR: There were not valid lines found in the namelist')
        return namelist
    
    # This is where the values in the namelist dictionary are changed
    
    nfound = 1
    for key in namelist.keys():
        if key != 'success':
            nfound += 1
            foo = np.where(key == inputt[:,0])[0]
            if len(foo) > 1:
                print('ERROR: There were multiple lines with the same key in the namelist: ' + key)
                return namelist
            
            elif len(foo) == 1:
                namelist[key] = type(namelist[key])(inputt[foo,1][0])
            
            else:
                nfound -= 1
    
    if namelist['use_calendar'] == 1:
        if ((namelist['start_year'] == 0) | (namelist['start_month'] == 0) | (namelist['start_day'] == 0) |
            (namelist['end_year'] == 0) | (namelist['end_month'] == 0) | (namelist['end_day'] == 0)):
                print('ERROR: If use_calendar = 1, then start_year, start_month, start_day, end_year, end_month, and end_year cannot be 0.')
                return namelist
        
        if namelist['model'] == 5:
            print('ERROR: CM1 requires use calendar to be 0')
            return namelist
    
    namelist['success'] = 1
    
    return namelist

def create_mesonet_wrf(stations,model_dir,time,prefix,namelist,latlon=1):
    
    print('Starting generation of obs for time: ' + time.strftime('%Y-%m-%d_%H:%M:%S'))
    file = model_dir + '/' + prefix + time.strftime('%Y-%m-%d_%H:%M:%S')
    
    try:
        f = Dataset(file,'r')
    except:
        print('Could not open ' + file)
        return -999., 0
    
    # Get the the location of the profiler based on the map projection
    if latlon == 1:
        
        # LCC projection
        if f.MAP_PROJ == 1:
            wrf_proj = pyproj.Proj(proj='lcc',lat_1 = f.TRUELAT1, lat_2 = f.TRUELAT2, lat_0 = f.MOAD_CEN_LAT, lon_0 = f.STAND_LON, a = 6370000, b = 6370000)
            wgs_proj = pyproj.Proj(proj='latlong',datum='WGS84')
            transformer = pyproj.Transformer.from_proj(wgs_proj,wrf_proj)
            
        # Now transform the data
        e, n = transformer.transform(f.CEN_LON, f.CEN_LAT)
        dx,dy = f.DX, f.DY
        nx, ny = f.dimensions['west_east'].size, f.dimensions['south_north'].size
        x0 = -(nx-1) / 2. * dx + e
        y0 = -(ny-1) / 2. * dy + n
        x_grid = np.arange(nx) * dx + x0
        y_grid = np.arange(ny) * dy + y0
        xx, yy = np.meshgrid(np.arange(nx) * dx + x0, np.arange(ny) * dy + y0)
        station_x_proj, station_y_proj = transformer.transform(stations[:,1], stations[:,0])
        
    else:
        xx, yy = np.meshgrid(np.arange(f.dimensions['west_east'].size) * f.DX, np.arange(f.dimensions['south_north'].size) * f.DY)
        station_x_proj = np.copy(stations[:,0])
        station_y_proj = np.copy(stations[:,1])
        
    t = f.variables['T2'][0]
    q = f.variables['Q2'][0]
    u = f.variables['U10'][0]
    v = f.variables['V10'][0]
    lat = f.variables['XLAT'][0]
    lon = f.variables['XLONG'][0]
    ground = f.variables['HGT'][0,:,:]
    psfc = f.variables['PSFC'][0]
    
    f.close()
    
    # Now get the data at the points we want
    
    f = RegularGridInterpolator((y_grid,x_grid), t, bounds_error=False)
    mesonet_t = f((station_y_proj,station_x_proj)) - 273.15
    
    f = RegularGridInterpolator((y_grid,x_grid), q, bounds_error=False)
    mesonet_q = f((station_y_proj,station_x_proj))*1000
    
    f = RegularGridInterpolator((y_grid,x_grid), u, bounds_error=False)
    mesonet_u = f((station_y_proj,station_x_proj))
    
    f = RegularGridInterpolator((y_grid,x_grid), v, bounds_error=False)
    mesonet_v = f((station_y_proj,station_x_proj))
    
    f = RegularGridInterpolator((y_grid,x_grid), ground, bounds_error=False)
    mesonet_alt = f((station_y_proj,station_x_proj))
    
    f = RegularGridInterpolator((y_grid,x_grid), psfc, bounds_error=False)
    mesonet_psfc = f((station_y_proj,station_x_proj))/100
    
    # Convert the specific humidity to relative humidity
    mesonet_rh = q2rh(mesonet_q,mesonet_psfc,mesonet_t)
    
    # Now lets add the specified errors to the observations
    mesonet_t = mesonet_t + np.random.default_rng().normal(0, namelist['temp_error'],len(mesonet_t))
    mesonet_rh = mesonet_rh + np.random.default_rng().normal(0, namelist['rh_error'],len(mesonet_rh))
    mesonet_psfc = mesonet_psfc + np.random.default_rng().normal(0, namelist['psfc_error'],len(mesonet_psfc))
    
    # Check to make sure the random error didn't make an unphysical relative humidity
    mesonet_rh[mesonet_rh > 100] = 100
    mesonet_rh[mesonet_rh <= 0] = 0.1
    
    # Convert the RH to dewpoint
    mesonet_dew = rh2dpt(mesonet_t,mesonet_rh/100)
    
    mesonet_u = mesonet_u + np.random.default_rng().normal(0, namelist['wind_error'],len(mesonet_u))
    mesonet_v = mesonet_v + np.random.default_rng().normal(0, namelist['wind_error'],len(mesonet_v))
    
    mesonet_wspd = np.sqrt(mesonet_u**2+mesonet_v**2)
    mesonet_wdir = np.arctan2(-mesonet_u,-mesonet_v) * 180/np.pi
    foo = np.where(mesonet_wdir < 0)
    mesonet_wdir[foo] += 360
    
    # Return mesonet data in a dictionary
    
    mesonet = {'lat':stations[:,0], 'lon':stations[:,1], 'alt':mesonet_alt, 'temp':mesonet_t,
               'dew':mesonet_dew, 'rh':mesonet_rh, 'wspd':mesonet_wspd, 'wdir':mesonet_wdir,
               'u':mesonet_u, 'v':mesonet_v, 'psfc':mesonet_psfc}
    
    return 1, mesonet

def write_to_file(mesonet,output_dir, namelist, model_time, snum, start_time, end_time):
    
    
    if namelist['use_calendar'] == 1:
        start_str = start_time.strftime('%Y%m%d_%H%M%S')
        end_str = end_time.strftime('%Y%m%d_%H%M%S')
    else:
        start_str = str(start_time)
        end_str = str(end_time)
    
    outfile_path = output_dir + namelist['outfile_root'] + '_' + start_str + '_' + end_str + '.nc'
    
    # We don't want to append the file needs to be created the first time  
    if ((namelist['append'] == 0) & (not os.path.exists(output_dir + namelist['outfile_root'] + '_' + start_str + '_' + end_str + '.nc'))):
        fid = Dataset(output_dir + namelist['outfile_root'] + '_' + start_str + '_' + end_str + '.nc','w')
        
        tdim = fid.createDimension('time',None)
        sdim = fid.createDimension('station',len(mesonet['lat']))
        
        base_time = fid.createVariable('base_time','i4')
        base_time.long_name = 'Epoch time'
        if namelist['use_calendar'] == 1:
            base_time.units = 's since 1970/01/01 00:00:00 UTC'
            base_time[:] = (start_time - datetime(1970,1,1)).total_seconds()
        else:
            base_time.units = 's from model start time'
            base_time[:] = start_time
        
        time = fid.createVariable('time_offset','f4',('time',))
        time.long_name = 'Time since base time'
        time.units = 'sec'
        
        lat = fid.createVariable('latitude','f4',('station',))
        lat.long_name = 'Latitude'
        lat.units = 'degree'
        
        lon = fid.createVariable('longitude','f4',('station',))
        lon.long_name = 'Longitude'
        lon.units = 'degree'
        
        alt = fid.createVariable('altitude','f4',('station',))
        alt.long_name = 'Altitude above sea level'
        alt.units = 'm'
        
        temp = fid.createVariable('temperature','f4',('time','station',))
        temp.long_name = 'Temperature'
        temp.units = 'degC'
        
        dew = fid.createVariable('dewpoint','f4',('time','station',))
        dew.long_name = 'Dewpoint'
        dew.units = 'degC'
        
        rh = fid.createVariable('relative_humidity','f4',('time','station',))
        rh.long_name = 'Relative Humidity'
        rh.units = '%'
        
        wspd = fid.createVariable('wind_speed','f4',('time','station',))
        wspd.long_name = 'Wind Speed'
        wspd.units = 'm/s'
        
        wdir = fid.createVariable('wind_direction','f4',('time','station',))
        wdir.long_name = 'Wind Direction'
        wdir.units = 'degree'
        
        u = fid.createVariable('u_wind','f4',('time','station',))
        u.long_name = 'U component of wind'
        u.units = 'm/s'
        
        v = fid.createVariable('v_wind','f4',('time','station',))
        v.long_name = 'V component of wind'
        v.units = 'm/s'
        
        p = fid.createVariable('surface_pressure', 'f4', ('time','station',))
        p.long_name = 'Surface pressure '
        p.units = 'hPa'
        
        if namelist['model'] == 1:
            fid.model = 'WRF'
        
        if namelist['use_calendar'] == 0:
            
            fid.start_time = start_time
            fid.end_time = end_time
        
        else:
            fid.start_time = start_time.strftime('%Y%m%d_%H%M%S')
            fid.end_time = end_time.strftime('%Y%m%d_%H%M%S')
        
        lat[:] = mesonet['lat']
        lon[:] = mesonet['lon']
        alt[:] = mesonet['alt']
    # Append the data to the file
    
    fid = Dataset(outfile_path,'a')
    
    bt = fid.variables['base_time']
    time = fid.variables['time_offset']
    lat = fid.variables['latitude']
    lon = fid.variables['longitude']
    alt = fid.variables['altitude']
    temp = fid.variables['temperature']
    dew = fid.variables['dewpoint']
    rh = fid.variables['relative_humidity']
    wspd = fid.variables['wind_speed']
    wdir = fid.variables['wind_direction']
    u = fid.variables['u_wind']
    v = fid.variables['v_wind']
    p = fid.variables['surface_pressure']
    
    
    if namelist['use_calendar'] == 0:
        time[snum] = model_time-start_time
    else:
        time[snum] = (model_time - datetime(1970,1,1)).total_seconds() - bt[0]
    
    temp[snum,:] = mesonet['temp'][:]
    dew[snum,:] = mesonet['dew'][:]
    rh[snum,:] = mesonet['rh'][:]
    wspd[snum,:] = mesonet['wspd']
    wdir[snum,:] = mesonet['wdir']
    u[snum,:] = mesonet['u']
    v[snum,:] = mesonet['v']
    p[snum,:] = mesonet['psfc']
    
    fid.sim_number = snum
    
    fid.close()

def write_dart_obs_seq(mesonet,output_dir,namelist,time):
    
    lats = np.radians(mesonet['lat'])
    lons = np.radians(mesonet['lon'])
    hgts = mesonet['alt']
    
    vert_coord = -1
    truth = 1.0
    
    lons = np.where(lons > 0.0, lons, lons+(2.0*np.pi))
    
    # We are adding temperature, dewpoint, u, v, and psfc
    data_length = len(np.where(~np.isnan(mesonet['temp']))[0])*5
    
    filename = output_dir + 'obs_seq_' +  time.strftime('%Y%m%d_%H%M%S')
    
    f = open(filename,"w")
    
    # Start with temperature
    
    nobs = 0
    
    for i in range(len(mesonet['temp'])):
        
        
        if np.isnan(mesonet['temp'][i]):
            pass
        else:
            nobs += 1
            
            sw_time = time - datetime(1601,1,1,0,0,0)
            
            days = sw_time.days
            seconds = sw_time.seconds
            
            f.write(" OBS            %d\n" % (nobs) )
            
            f.write("   %20.14f\n" % (mesonet['temp'][i] + 273.15) )
            f.write("   %20.14f\n" % truth  )
            
            if nobs == 1:
                f.write(" %d %d %d\n" % (-1, nobs+1, -1) ) # First obs.
            elif nobs == data_length:
                f.write(" %d %d %d\n" % (nobs-1, -1, -1) ) # Last obs.
            else:
                f.write(" %d %d %d\n" % (nobs-1, nobs+1, -1) )
            
            f.write("obdef\n")
            f.write("loc3d\n")
            
            f.write("    %20.14f          %20.14f          %20.14f     %d\n" % 
                  (lons[i], lats[i], hgts[i], vert_coord))
            
            f.write("kind\n")
            
            f.write("     %d     \n" % 27 )
            
            f.write("    %d          %d     \n" % (seconds, days) )
            
            f.write("    %20.14f  \n" % namelist['temp_error']**2 )
    
    for i in range(len(mesonet['dew'])):
        
        
        if np.isnan(mesonet['dew'][i]):
            pass
        else:
            nobs += 1
            
            sw_time = time - datetime(1601,1,1,0,0,0)
            
            days = sw_time.days
            seconds = sw_time.seconds
            
            f.write(" OBS            %d\n" % (nobs) )
            
            f.write("   %20.14f\n" % (mesonet['dew'][i] + 273.15) )
            f.write("   %20.14f\n" % truth  )
            
            if nobs == 1:
                f.write(" %d %d %d\n" % (-1, nobs+1, -1) ) # First obs.
            elif nobs == data_length:
                f.write(" %d %d %d\n" % (nobs-1, -1, -1) ) # Last obs.
            else:
                f.write(" %d %d %d\n" % (nobs-1, nobs+1, -1) )
            
            f.write("obdef\n")
            f.write("loc3d\n")
            
            f.write("    %20.14f          %20.14f          %20.14f     %d\n" % 
                  (lons[i], lats[i], hgts[i], vert_coord))
            
            f.write("kind\n")
            
            f.write("     %d     \n" % 66 )
            
            f.write("    %d          %d     \n" % (seconds, days) )
            
            tmp_error = calc_dewpoint_error(mesonet['temp'][i], mesonet['rh'][i]/100)
            
            f.write("    %20.14f  \n" % tmp_error**2 )
        
    for i in range(len(mesonet['psfc'])):
        
        
        if np.isnan(mesonet['psfc'][i]):
            pass
        else:
            nobs += 1
            
            sw_time = time - datetime(1601,1,1,0,0,0)
            
            days = sw_time.days
            seconds = sw_time.seconds
            
            f.write(" OBS            %d\n" % (nobs) )
            
            f.write("   %20.14f\n" % mesonet['psfc'][i])
            f.write("   %20.14f\n" % truth  )
            
            if nobs == 1:
                f.write(" %d %d %d\n" % (-1, nobs+1, -1) ) # First obs.
            elif nobs == data_length:
                f.write(" %d %d %d\n" % (nobs-1, -1, -1) ) # Last obs.
            else:
                f.write(" %d %d %d\n" % (nobs-1, nobs+1, -1) )
            
            f.write("obdef\n")
            f.write("loc3d\n")
            
            f.write("    %20.14f          %20.14f          %20.14f     %d\n" % 
                  (lons[i], lats[i], hgts[i], vert_coord))
            
            f.write("kind\n")
            
            f.write("     %d     \n" % 78 )
            
            f.write("    %d          %d     \n" % (seconds, days) )
            
            f.write("    %20.14f  \n" % 1)
        
    for i in range(len(mesonet['u'])):
        
        
        if np.isnan(mesonet['u'][i]):
            pass
        else:
            nobs += 1
            
            sw_time = time - datetime(1601,1,1,0,0,0)
            
            days = sw_time.days
            seconds = sw_time.seconds
            
            f.write(" OBS            %d\n" % (nobs) )
            
            f.write("   %20.14f\n" % mesonet['u'][i])
            f.write("   %20.14f\n" % truth  )
            
            if nobs == 1:
                f.write(" %d %d %d\n" % (-1, nobs+1, -1) ) # First obs.
            elif nobs == data_length:
                f.write(" %d %d %d\n" % (nobs-1, -1, -1) ) # Last obs.
            else:
                f.write(" %d %d %d\n" % (nobs-1, nobs+1, -1) )
            
            f.write("obdef\n")
            f.write("loc3d\n")
            
            f.write("    %20.14f          %20.14f          %20.14f     %d\n" % 
                  (lons[i], lats[i], hgts[i], vert_coord))
            
            f.write("kind\n")
            
            f.write("     %d     \n" % 25 )
            
            f.write("    %d          %d     \n" % (seconds, days) )
            
            f.write("    %20.14f  \n" % namelist['wind_error']**2)
    
    for i in range(len(mesonet['v'])):
        
        
        if np.isnan(mesonet['v'][i]):
            pass
        else:
            nobs += 1
            
            sw_time = time - datetime(1601,1,1,0,0,0)
            
            days = sw_time.days
            seconds = sw_time.seconds
            
            f.write(" OBS            %d\n" % (nobs) )
            
            f.write("   %20.14f\n" % mesonet['v'][i])
            f.write("   %20.14f\n" % truth  )
            
            if nobs == 1:
                f.write(" %d %d %d\n" % (-1, nobs+1, -1) ) # First obs.
            elif nobs == data_length:
                f.write(" %d %d %d\n" % (nobs-1, -1, -1) ) # Last obs.
            else:
                f.write(" %d %d %d\n" % (nobs-1, nobs+1, -1) )
            
            f.write("obdef\n")
            f.write("loc3d\n")
            
            f.write("    %20.14f          %20.14f          %20.14f     %d\n" % 
                  (lons[i], lats[i], hgts[i], vert_coord))
            
            f.write("kind\n")
            
            f.write("     %d     \n" % 26 )
            
            f.write("    %d          %d     \n" % (seconds, days) )
            
            f.write("    %20.14f  \n" % namelist['wind_error']**2)
    
    f.close()
    
    # Now write out header informations
    f = open(filename,'r')
    f_obs_seq = f.read()
    f.close()
    
    f = open(filename,'w')
    
    f.write(" obs_sequence\n")
    f.write("obs_kind_definitions\n")
    
    f.write("       %d\n" % 5)
    f.write("    %d          %s   \n" % (25, "LAND_SFC_U_WIND_COMPONENT") )
    f.write("    %d          %s   \n" % (26, "LAND_SFC_V_WIND_COMPONENT") )
    f.write("    %d          %s   \n" % (27, "LAND_SFC_TEMPERATURE") )
    f.write("    %d          %s   \n" % (66, "LAND_SFC_DEWPOINT") )
    f.write("    %d          %s   \n" % (78, "LAND_SFC_ALTIMETER") )
    f.write("  num_copies:            %d  num_qc:            %d\n" % (1, 1))
    f.write(" num_obs:       %d  max_num_obs:       %d\n" % (nobs, nobs) )
    f.write("observations\n")
    f.write("QC radar\n")
    f.write("  first:            %d  last:       %d\n" % (1, nobs) )
    
    f.write(f_obs_seq)
  
    f.close()
    
##############################################################################          
#Create parser for command line arguments
parser = ArgumentParser()

parser.add_argument("namelist_file", help="Name of the namelist file (string)")
parser.add_argument("--output_dir", help="Path to output directory")
parser.add_argument("--write_dart",action="store_true",help="Flag to write out DART obs_seq files")
parser.add_argument("--debug", action="store_true", help="Set this to turn on the debug mode")

args = parser.parse_args()

namelist_file = args.namelist_file
output_dir = args.output_dir
write_dart = args.write_dart
debug = args.debug
    
if output_dir is None:
    output_dir = os.getcwd() + '/'

print("-----------------------------------------------------------------------")
print("Starting Synthetic Mesonet")
print("Output directory set to " + output_dir)

# Read the namelist file
namelist = read_namelist(namelist_file)
namelist['output_dir'] = output_dir
if namelist['success'] != 1:
    print('>>> Synthetic Mesonet FAILED and ABORTED <<<')
    print("-----------------------------------------------------------------------")
    sys.exit()

# Read in the radar scan file
print('Reading in mesonet station file')
try:
    stations = np.genfromtxt(namelist['station_file'], delimiter= ' ',autostrip=True)
except:
    print('ERROR: Something went wrong reading radar scan')
    print('>>> LidarSim FAILED and ABORTED <<<')
    print("-----------------------------------------------------------------------")
    sys.exit()


if namelist['use_calendar'] == 0:
    
    start_time = namelist['start_time']
    end_time = namelist['end_time']
    
    model_time = np.arange(namelist['start_time'],namelist['end_time']+namelist['model_frequency'],
                           namelist['model_frequency'])
    
    snum = ((model_time-start_time)/namelist['model_frequency']).astype(int)
     
else:
    
    start_time = datetime(namelist['start_year'],namelist['start_month'],namelist['start_day'],namelist['start_hour'],namelist['start_min'],namelist['start_sec'])
    end_time = datetime(namelist['end_year'],namelist['end_month'],namelist['end_day'],namelist['end_hour'],namelist['end_min'],namelist['end_sec'])
    
    model_time = np.arange(start_time,end_time+timedelta(seconds=namelist['model_frequency']),timedelta(seconds=namelist['model_frequency'])).astype(datetime)

    snum = (np.array([(x - start_time).seconds for x in model_time])/namelist['model_frequency']).astype(int)

if namelist['use_calendar'] == 1:
    start_str = model_time[0].strftime('%Y%m%d_%H%M%S')
    end_str = model_time[-1].strftime('%Y%m%d_%H%M%S')
else:
    start_str = str(model_time[0])
    end_str = str(model_time[-1])
    
last_snum = -1
# Check to see if this is an append run. 
if namelist['append'] == 1:
        
    # Make sure the files exists
    if os.path.exists(namelist['output_dir'] + namelist['outfile_root'] + '_' + start_str + '_' + end_str + '.nc'):
        out = Dataset(namelist['output_dir'] + namelist['outfile_root'] + '_' + start_str + '_' + end_str + '.nc')
        
        # Make sure that the start time is the same
        
        if namelist['use_calendar'] == 0:
            if namelist['start_time'] != int(out.start_time):
                print('Append mode was selected, but the start time is not the same.')
                print(('>>> SyntheticMesonet FAILED and ABORTED'))
                print('--------------------------------------------------------------------')
                print(' ')
                sys.exit()
        
        else:
           if start_time.strftime('%Y%m%d_%H%M%S') != out.start_time:
               print('Append mode was selected, but the start time is not the same.')
               print(('>>> SyntheticMesonet FAILED and ABORTED'))
               print('--------------------------------------------------------------------')
               print(' ')
               sys.exit()
        
        
        # Get the scan number in the output file 
        last_snum = int(out.sim_number)
        out.close()
        
    else:
        print('Append mode was selcted, but ' +namelist['output_dir'] + namelist['outfile_root'] + '_' + start_str + '_' + end_str + '.nc' + ' does not exist.')
        print('A new output file will be created!')
        namelist['append'] = 0

else:
    if os.path.exists(namelist['output_dir'] + namelist['outfile_root'] + '_' + start_str + '_' + end_str + '.nc'):
        if namelist['clobber'] == 1:
            print(namelist['output_dir'] + namelist['outfile_root'] + '_' + start_str + '_' + end_str + '.nc' + ' exists and will be clobbered!')
            os.remove(namelist['output_dir'] + namelist['outfile_root'] + '_' + start_str + '_' + end_str + '.nc')
        else:
            print('ERROR:' + namelist['output_dir'] + namelist['outfile_root'] + '_' + start_str + '_' + end_str + '.nc' + ' exists and clobber is set to 0.')
            print('       Must abort to prevent file from being overwritten!')
            sys.exit()


for index in range(len(snum)):
    
    print('Performing simulation number ' + str(snum[index]) +  ' at time ' + str(model_time[index]))
    
    if index <= last_snum:
        '        ...but was already processed. Continuing.'
        continue
    
    if namelist['model'] == 1:
        success, mesonet = create_mesonet_wrf(stations,namelist['model_dir'],model_time[index],
                                     namelist['model_prefix'], namelist, namelist['coordinate_type'])
    
        if success != 1:
            print('Something went wrong collecting the radar obs for ', model_time)
            print('Skipping model time')
            continue
    else:
        print('Error: Model type ' + namelist['model_type'] + ' is not a valid option')
        
        
    write_to_file(mesonet,output_dir, namelist, model_time[index], index, start_time, end_time)
    
    if write_dart:
        write_dart_obs_seq(mesonet,output_dir,namelist,model_time[index])
