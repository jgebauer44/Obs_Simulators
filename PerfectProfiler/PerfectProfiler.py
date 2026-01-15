#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 16:16:29 2024

@author: joshua.gebauer
"""

import os
import sys
import numpy as np
import glob
import pyproj
import struct
import xarray as xr
from netCDF4 import Dataset
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
from scipy.special import erf
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
    
    yo = np.where(np.atleast_1d(temp) == 0)[0]
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

def theta2t(theta, w, p, p0 = 1000.):
    th = theta/ ( 1 + 0.61 * (w/1000.))
    t = th / ( (p0/p)**0.286 ) - 273.16
    
    return t

def calc_dewpoint_error(temp,rh):
    
    rh_error = 0.05
    t_error = 1.0
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
                 'append':1,                       # =0 - don't append to file, 1 - append to file
                 'max_height':5000,               # Height in meters
                 'vertical_res':50,               # Resolution in meters
                 'temp_error':1.0,                  # 1-sigma error in temperature in Kelvin
                 'rh_error':0.5,                    # 1-sigma rh error
                 'psfc_error':1.0,                   # 1-sigma surface pressure error
                 'wind_error':3.0,                   # 1-sigma error in wind components
                 'clobber':0,                      # 0 - don't clobber file, 1 - clobber file (ignored if append = 1)
                 'level_spacing':10                # Spacing between height levels
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

def create_profiler_wrf(stations,model_dir,time,prefix,levels, namelist,latlon=1):
    
    print('Starting generation of obs for time: ' + time.strftime('%Y-%m-%d_%H:%M:%S'))
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
        station_x_proj, station_y_proj = transformer.transform(stations[:,1], stations[:,0])
        
    else:
        xx, yy = np.meshgrid(np.arange(fid.dimensions['west_east'].size) * fid.DX, np.arange(fid.dimensions['south_north'].size) * fid.DY)
        station_x_proj = np.copy(stations[:,0])
        station_y_proj = np.copy(stations[:,1])
        
    
    # Now get the data at the points we want
    
    #Get the data
    zz = (fid.variables['PH'][0,:,:,:]+fid.variables['PHB'][0,:,:,:])/9.81
    
    fake_z = np.arange(zz.shape[0]-1)
    fake_z1 = np.arange(zz.shape[0])
    f_z = RegularGridInterpolator((fake_z1,y_grid,x_grid), zz, bounds_error=False)
    
    ground = fid.variables['HGT'][0,:,:]
    f_g = RegularGridInterpolator((y_grid,x_grid), ground, bounds_error=False)
    
    p = (fid.variables['P'][0,:,:,:] + fid.variables['PB'][0,:,:,:])/100
    f_p = RegularGridInterpolator((fake_z,y_grid,x_grid), p, bounds_error=False)
    
    t = fid.variables['T'][0] + 300
    f_t = RegularGridInterpolator((fake_z,y_grid,x_grid), t, bounds_error=False)
    
    q = fid.variables['QVAPOR'][0]*1000
    f_q = RegularGridInterpolator((fake_z,y_grid,x_grid), q, bounds_error=False)
    
    rain = np.nanmax(fid.variables['REFL_10CM'][0],axis=0)
    f_rain = RegularGridInterpolator((y_grid,x_grid), rain, bounds_error=False)
    
    sinalpha = fid['SINALPHA'][0,:,:]
    cosalpha = fid['COSALPHA'][0,:,:]
    
    u = (fid.variables['U'][0,:,:,1:] + fid.variables['U'][0,:,:,:-1])/2
    v = (fid.variables['V'][0,:,1:,:] + fid.variables['V'][0,:,:-1,:])/2
    
    utmp = u*cosalpha[None,:,:] - v*sinalpha[None,:,:]
    vtmp = v*cosalpha[None,:,:] + u*sinalpha[None,:,:]
    
    f_u = RegularGridInterpolator((fake_z,y_grid,x_grid), utmp, bounds_error=False)
    f_v = RegularGridInterpolator((fake_z,y_grid,x_grid), vtmp, bounds_error=False)
    
    # We want all of the data on model levels from above the surface to max height
    
    if levels[0] < 0:
        nlevels = t.shape[0]
    else:
        nlevels = len(levels)

    profiler_t = np.ones((nlevels,len(station_y_proj)))*np.nan
    profiler_u = np.ones((nlevels,len(station_y_proj)))*np.nan
    profiler_v = np.ones((nlevels,len(station_y_proj)))*np.nan
    profiler_dew = np.ones((nlevels,len(station_y_proj)))*np.nan
    profiler_alt = np.ones((nlevels,len(station_y_proj)))*np.nan
    profiler_ground = np.ones(len(station_y_proj))*np.nan
    profiler_rh = np.ones((nlevels,len(station_y_proj)))*np.nan
    profiler_q = np.ones((nlevels,len(station_y_proj)))*np.nan
    profiler_p = np.ones((nlevels,len(station_y_proj)))*np.nan

    profiler_p = np.ones((nlevels,len(station_y_proj)))*np.nan

    profiler_tsfc = np.ones(len(station_y_proj))*np.nan
    profiler_qsfc = np.ones(len(station_y_proj))*np.nan
    
    
    for i in range(len(station_y_proj)):
        
        zz = f_z((fake_z1,np.ones(fake_z1.shape[0])*station_y_proj[i],np.ones(fake_z1.shape[0])*station_x_proj[i]))
        zz = (zz[1:]+zz[:-1])/2.
    
    
        pp = f_p((fake_z,np.ones(fake_z.shape[0])*station_y_proj[i],np.ones(fake_z.shape[0])*station_x_proj[i]))
   
        ground0 = f_g((station_y_proj[i],station_x_proj[i]))
    
        tt= f_t((fake_z,np.ones(fake_z.shape[0])*station_y_proj[i],np.ones(fake_z.shape[0])*station_x_proj[i]))
        tt = theta2t(tt, np.zeros(len(pp)), pp)
   
    
        qq = f_q((fake_z,np.ones(fake_z.shape[0])*station_y_proj[i],np.ones(fake_z.shape[0])*station_x_proj[i]))
    
        # Convert the specific humidity to relative humidity
        rh = q2rh(qq,pp,tt)
        
        uu = f_u((fake_z,np.ones(fake_z.shape[0])*station_y_proj[i],np.ones(fake_z.shape[0])*station_x_proj[i]))
        
        vv = f_v((fake_z,np.ones(fake_z.shape[0])*station_y_proj[i],np.ones(fake_z.shape[0])*station_x_proj[i]))
        
        rr = f_rain((station_y_proj[i],station_x_proj[i]))
        
        if rr > 15:
            rh[:] = np.nan
            tt[:] = np.nan
            uu[:] = np.nan
            vv[:] = np.nan
            qq[:] = np.nan
            pp[:] = np.nan
        
        profiler_t[:,i] = np.interp(levels,zz-ground0,tt)
        profiler_dew[:,i] = np.interp(levels,zz-ground0,rh2dpt(tt,rh/100))
        profiler_u[:,i] = np.interp(levels,zz-ground0,uu)
        profiler_v[:,i] = np.interp(levels,zz-ground0,vv)
        profiler_rh[:,i] = np.interp(levels,zz-ground0,rh)
        profiler_q[:,i] = np.interp(levels,zz-ground0,qq)
        profiler_p[:,i] = np.interp(levels,zz-ground0,pp)
        
        profiler_alt[:,i] = levels + ground0
        profiler_ground[i] = np.copy(ground0)
        
    # Return mesonet data in a dictionary
    
    profiler = {'lat':stations[:,0], 'lon':stations[:,1], 'alt':profiler_ground, 'temp':profiler_t,
               'dew':profiler_dew, 'u':profiler_u, 'v':profiler_v, 'rh':profiler_rh, 'z':profiler_alt,
               'q':profiler_q, 'p':profiler_p}
    
    return 1, profiler

def write_to_file(profiler,output_dir, namelist, model_time, snum, start_time, end_time):
    
    
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
        sdim = fid.createDimension('station',len(profiler['lat']))
        hdim = fid.createDimension('height'),len(profiler['z'])
        
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
        
        height = fid.createVariable('height','f4',('time','height','station'))
        height.long_name = 'Height above ground level'
        height.units = 'm'
        
        temp = fid.createVariable('temperature','f4',('time','height','station',))
        temp.long_name = 'Temperature'
        temp.units = 'degC'
        
        dew = fid.createVariable('dewpoint','f4',('time','height','station',))
        dew.long_name = 'Dewpoint'
        dew.units = 'degC'
        
        rh = fid.createVariable('relative_humidity', 'f4', ('time','height','station',))
        rh.long_name = 'Relative Humidity'
        rh.units = '%'
        
        u = fid.createVariable('u_wind','f4',('time','height','station',))
        u.long_name = 'U component of wind'
        u.units = 'm/s'
        
        v = fid.createVariable('v_wind','f4',('time','height','station',))
        v.long_name = 'V component of wind'
        v.units = 'm/s'
        
        q = fid.createVariable('qvapor','f4',('time','height','station',))
        q.long_name = 'Specific humidity'
        q.units = 'g/kg'
        
        p = fid.createVariable('pres','f4',('time','height','station',))
        p.long_name = 'Pressure'
        p.units = 'hPa'
        
        if namelist['model'] == 1:
            fid.model = 'WRF'
        
        if namelist['use_calendar'] == 0:
            
            fid.start_time = start_time
            fid.end_time = end_time
        
        else:
            fid.start_time = start_time.strftime('%Y%m%d_%H%M%S')
            fid.end_time = end_time.strftime('%Y%m%d_%H%M%S')
        
        lat[:] = profiler['lat']
        lon[:] = profiler['lon']
        alt[:] = profiler['alt']
    # Append the data to the file
    
    fid = Dataset(outfile_path,'a')
    
    bt = fid.variables['base_time']
    time = fid.variables['time_offset']
    height= fid.variables['height']
    temp = fid.variables['temperature']
    dew = fid.variables['dewpoint']
    u = fid.variables['u_wind']
    v = fid.variables['v_wind']
    q = fid.variables['qvapor']
    p = fid.variables['pres']
    rh = fid.variables['relative_humidity']
    
    
    if namelist['use_calendar'] == 0:
        time[snum] = model_time-start_time
    else:
        time[snum] = (model_time - datetime(1970,1,1)).total_seconds() - bt[0]
    
    height[snum,:] = profiler['z'][:]
    temp[snum,:] = profiler['temp'][:]
    dew[snum,:] = profiler['dew'][:]
    u[snum,:] = profiler['u'][:]
    v[snum,:] = profiler['v'][:]
    q[snum,:] = profiler['q'][:]
    p[snum,:] = profiler['p'][:]
    rh[snum,:] = profiler['rh'][:]
    
    fid.sim_number = snum
    
    fid.close()
    
def write_dart_obs_seq(profiler,output_dir,namelist,time):
    
    lats = np.radians(profiler['lat'])
    lons = np.radians(profiler['lon'])
    hgts = profiler['alt']
    
    vert_coord = 3
    truth = 1.0
    
    lons = np.where(lons > 0.0, lons, lons+(2.0*np.pi))
    
    # We are adding temperature, dewpoint, u, v, and psfc
    data_length = len(np.where(~np.isnan(profiler['temp']))[0])*4
    
    filename = output_dir + 'obs_seq_' +  time.strftime('%Y%m%d_%H%M%S')
    
    f = open(filename,"w")
    
    # Start with temperature
    
    nobs = 0
    
    for i in range(profiler['temp'].shape[1]):
        for j in range(profiler['temp'].shape[0]):
        
        
            if np.isnan(profiler['temp'][j,i]):
                pass
            else:
                nobs += 1
            
                sw_time = time - datetime(1601,1,1,0,0,0)
            
                days = sw_time.days
                seconds = sw_time.seconds
            
                f.write(" OBS            %d\n" % (nobs) )
            
                f.write("   %20.14f\n" % (profiler['temp'][j,i] + 273.15) )
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
                        (lons[i], lats[i], profiler['z'][j,i], vert_coord))
            
                f.write("kind\n")
                
                f.write("     %d     \n" % 14 )
            
                f.write("    %d          %d     \n" % (seconds, days) )
            
                f.write("    %20.14f  \n" % namelist['temp_error']**2 )
    
    for i in range(profiler['dew'].shape[1]):
        for j in range(profiler['dew'].shape[0]):
        
            if np.isnan(profiler['dew'][j,i]):
                pass
            else:
                nobs += 1
                
                sw_time = time - datetime(1601,1,1,0,0,0)
                
                days = sw_time.days
                seconds = sw_time.seconds
                
                f.write(" OBS            %d\n" % (nobs) )
            
                f.write("   %20.14f\n" % (profiler['dew'][j,i] + 273.15) )
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
                        (lons[i], lats[i], profiler['z'][j,i], vert_coord))
            
                f.write("kind\n")
            
                f.write("     %d     \n" % 16 )
            
                f.write("    %d          %d     \n" % (seconds, days) )
                
                tmp_error = calc_dewpoint_error(profiler['temp'][j,i], profiler['rh'][j,i]/100)
                
                f.write("    %20.14f  \n" % tmp_error**2 )
        
    for i in range(profiler['u'].shape[1]):
        for j in range(profiler['u'].shape[0]):
        
            if np.isnan(profiler['u'][j,i]):
                pass
            else:
                nobs += 1
            
                sw_time = time - datetime(1601,1,1,0,0,0)
            
                days = sw_time.days
                seconds = sw_time.seconds
            
                f.write(" OBS            %d\n" % (nobs) )
            
                f.write("   %20.14f\n" % profiler['u'][j,i])
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
                        (lons[i], lats[i], profiler['z'][j,i], vert_coord))
            
                f.write("kind\n")
            
                f.write("     %d     \n" % 17 )
            
                f.write("    %d          %d     \n" % (seconds, days) )
            
                f.write("    %20.14f  \n" % namelist['wind_error']**2)
    
    for i in range(profiler['v'].shape[1]):
        for j in range(profiler['v'].shape[0]):
        
        
            if np.isnan(profiler['v'][j,i]):
                pass
            else:
                nobs += 1
                
                sw_time = time - datetime(1601,1,1,0,0,0)
                
                days = sw_time.days
                seconds = sw_time.seconds
                
                f.write(" OBS            %d\n" % (nobs) )
                
                f.write("   %20.14f\n" % profiler['v'][j,i])
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
                        (lons[i], lats[i], profiler['z'][j,i], vert_coord))
            
                f.write("kind\n")
            
                f.write("     %d     \n" % 18 )
            
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
    
    f.write("       %d\n" % 4)
    f.write("    %d          %s   \n" % (14, "RADIOSONDE_TEMPERATURE") )
    f.write("    %d          %s   \n" % (16, "RADIOSONDE_DEWPOINT") )
    f.write("    %d          %s   \n" % (17, "RADIOSONDE_U_WIND_COMPONENT") )
    f.write("    %d          %s   \n" % (18, "RADIOSONDE_V_WIND_COMPONENT") )
    f.write("  num_copies:            %d  num_qc:            %d\n" % (1, 1))
    f.write(" num_obs:       %d  max_num_obs:       %d\n" % (nobs, nobs) )
    f.write("observations\n")
    f.write("QC obs\n")
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
print("Starting PerfectProfiler")
print("Output directory set to " + output_dir)

# Read the namelist file
namelist = read_namelist(namelist_file)
namelist['output_dir'] = output_dir
if namelist['success'] != 1:
    print('>>> PerfectProfiler FAILED and ABORTED <<<')
    print("-----------------------------------------------------------------------")
    sys.exit()

# Read in the radar scan file
print('Reading in profiler station file')
try:
    stations = np.genfromtxt(namelist['station_file'],autostrip=True)
except:
    print('ERROR: Something went wrong reading profiler station file')
    print('>>> PerfectProfiler FAILED and ABORTED <<<')
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

# Get the levels for the perfect profilers
if namelist['level_spaing'] < 0:
    levels = np.array([-1])
else:
    levels = np.arange(namelist['level_spacing'],namelist['max_height']+1,namelist['level_spacing'])

last_snum = -1
# Check to see if this is an append run. 
if namelist['append'] == 1:
    print('Error: Append mode is currently broken. Blame Josh Gebauer')
    sys.exit()
    
    # Make sure the files exists
    if os.path.exists(namelist['output_dir'] + namelist['outfile_root'] + '_' + start_str + '_' + end_str + '.nc'):
        out = Dataset(namelist['output_dir'] + namelist['outfile_root'] + '_' + start_str + '_' + end_str + '.nc')
        
        # Make sure that the start time is the same
        
        if namelist['use_calendar'] == 0:
            if namelist['start_time'] != int(out.start_time):
                print('Append mode was selected, but the start time is not the same.')
                print(('>>> PerfectProfiler FAILED and ABORTED'))
                print('--------------------------------------------------------------------')
                print(' ')
                sys.exit()
        
        else:
           if start_time.strftime('%Y%m%d_%H%M%S') != out.start_time:
               print('Append mode was selected, but the start time is not the same.')
               print(('>>> PerfectProfiler FAILED and ABORTED'))
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
        success, profiler = create_profiler_wrf(stations,namelist['model_dir'],model_time[index],
                                     namelist['model_prefix'], levels, namelist, namelist['coordinate_type'])
    
        if success != 1:
            print('Something went wrong collecting the radar obs for ', model_time[index])
            print('Skipping model time')
            continue
    else:
        print('Error: Model type ' + namelist['model_type'] + ' is not a valid option')
        
        
    write_to_file(profiler,output_dir, namelist, model_time[index], index, start_time, end_time)
    
    if write_dart:
        write_dart_obs_seq(profiler,output_dir,namelist,model_time[index])
