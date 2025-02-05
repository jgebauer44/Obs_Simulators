#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 11:18:57 2024

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

def calc_dewpoint_error(temp,rh):
    
    rh_error = 0.05
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


def write_to_file(profiler,output_dir, flight_stime, uas_time, start_time, end_time):
    
    
    start_str = start_time.strftime('%Y%m%d_%H%M%S')
    end_str = end_time.strftime('%Y%m%d_%H%M%S')
    
    outfile_path = output_dir +  'UAS_' + start_str + '_' + end_str + '.nc'
   
    fid = Dataset(outfile_path,'w')
        
    tdim = fid.createDimension('flight',None)
    sdim = fid.createDimension('station',len(profiler['lat']))
    hdim = fid.createDimension('height'),len(profiler['z'])
        
    base_time = fid.createVariable('base_time','i4',('flight',))
    base_time.long_name = 'Epoch time'
    base_time.units = 's since 1970/01/01 00:00:00 UTC'
    base_time[:] = flight_stime
        
    time = fid.createVariable('time_offset','f4',('flight',))
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
        
    height = fid.createVariable('height','f4',('height'))
    height.long_name = 'Height above ground level'
    height.units = 'm'
        
    temp = fid.createVariable('temperature','f4',('flight','height','station',))
    temp.long_name = 'Temperature'
    temp.units = 'degC'
        
    dew = fid.createVariable('dewpoint','f4',('flight','height','station',))
    dew.long_name = 'Dewpoint'
    dew.units = 'degC'
        
    rh = fid.createVariable('relative_humidity', 'f4', ('flight','height','station',))
    rh.long_name = 'Relative Humidity'
    rh.units = '%'
        
    u = fid.createVariable('u_wind','f4',('flight','height','station',))
    u.long_name = 'U component of wind'
    u.units = 'm/s'
        
    v = fid.createVariable('v_wind','f4',('flight','height','station',))
    v.long_name = 'V component of wind'
    v.units = 'm/s'
        
    p = fid.createVariable('pres','f4',('flight','height','station',))
    p.long_name = 'Pressure'
    p.units = 'hPa'
        
    fid.model = 'WRF'
        
    fid.start_time = start_time.strftime('%Y%m%d_%H%M%S')
    fid.end_time = end_time.strftime('%Y%m%d_%H%M%S')
        
    lat[:] = profiler['lat']
    lon[:] = profiler['lon']
    alt[:] = profiler['alt']
    
    height[:] = profiler['z'][:]
    temp[:] = profiler['temp'][:]
    dew[:] = profiler['dew'][:]
    u[:] = profiler['u'][:]
    v[:] = profiler['v'][:]
    p[:] = profiler['p'][:]
    rh[:] = profiler['rh'][:]
    
    fid.close()
    
def write_dart_obs_seq(profiler,output_dir,time,temp_error,rh_error,wind_error):
    
    lats = np.radians(profiler['lat'])
    lons = np.radians(profiler['lon'])
    hgts = profiler['alt']
    
    vert_coord = 3
    truth = 1.0
    
    lons = np.where(lons > 0.0, lons, lons+(2.0*np.pi))
    
    for k in range(len(time)):
        # We are adding temperature, dewpoint, u, v, and psfc
        data_length = len(np.where(~np.isnan(profiler['temp'][k]))[0])*4
    
        filename = output_dir + 'obs_seq_' +  time[k].strftime('%Y%m%d_%H%M%S')
    
        f = open(filename,"w")
        
        print(filename)
        # Start with temperature
    
        nobs = 0
    
        for i in range(profiler['temp'][k].shape[1]):
            for j in range(profiler['temp'][k].shape[0]):
        
        
                if np.isnan(profiler['temp'][k,j,i]):
                    pass
                else:
                    nobs += 1
                    
                    sw_time = time[k] - datetime(1601,1,1,0,0,0)
                    
                    days = sw_time.days
                    seconds = sw_time.seconds
                    
                    f.write(" OBS            %d\n" % (nobs) )
                    
                    f.write("   %20.14f\n" % (profiler['temp'][k,j,i] +273.15))
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
                            (lons[i], lats[i], profiler['z'][j]+hgts[i], vert_coord))
            
                    f.write("kind\n")
                
                    f.write("     %d     \n" % 14 )
            
                    f.write("    %d          %d     \n" % (seconds, days) )
            
                    f.write("    %20.14f  \n" % temp_error**2 )
    
        for i in range(profiler['dew'][k].shape[1]):
            for j in range(profiler['dew'][k].shape[0]):
        
                if np.isnan(profiler['dew'][k,j,i]):
                    pass
                else:
                    nobs += 1
                
                    sw_time = time[k] - datetime(1601,1,1,0,0,0)
                
                    days = sw_time.days
                    seconds = sw_time.seconds
                
                    f.write(" OBS            %d\n" % (nobs) )
            
                    f.write("   %20.14f\n" % (profiler['dew'][k,j,i] + 273.15) )
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
                            (lons[i], lats[i], profiler['z'][j]+hgts[i], vert_coord))
            
                    f.write("kind\n")
            
                    f.write("     %d     \n" % 16 )
            
                    f.write("    %d          %d     \n" % (seconds, days) )
                
                    tmp_error = calc_dewpoint_error(profiler['temp'][k,j,i], profiler['rh'][k,j,i]/100)
                
                    f.write("    %20.14f  \n" % tmp_error**2 )
        
        for i in range(profiler['u'][k].shape[1]):
            for j in range(profiler['u'][k].shape[0]):
        
                if np.isnan(profiler['u'][k,j,i]):
                    pass
                else:
                    nobs += 1
            
                    sw_time = time[k] - datetime(1601,1,1,0,0,0)
            
                    days = sw_time.days
                    seconds = sw_time.seconds
            
                    f.write(" OBS            %d\n" % (nobs) )
            
                    f.write("   %20.14f\n" % profiler['u'][k,j,i])
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
                            (lons[i], lats[i], profiler['z'][j]+hgts[i], vert_coord))
            
                    f.write("kind\n")
            
                    f.write("     %d     \n" % 17 )
            
                    f.write("    %d          %d     \n" % (seconds, days) )
            
                    f.write("    %20.14f  \n" % wind_error**2)
    
        for i in range(profiler['v'][k].shape[1]):
            for j in range(profiler['v'][k].shape[0]):
        
        
                if np.isnan(profiler['v'][k,j,i]):
                    pass
                else:
                    nobs += 1
                
                    sw_time = time[k] - datetime(1601,1,1,0,0,0)
                    
                    days = sw_time.days
                    seconds = sw_time.seconds
                
                    f.write(" OBS            %d\n" % (nobs) )
                
                    f.write("   %20.14f\n" % profiler['v'][k,j,i])
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
                            (lons[i], lats[i], profiler['z'][j]+hgts[i], vert_coord))
            
                    f.write("kind\n")
            
                    f.write("     %d     \n" % 18 )
            
                    f.write("    %d          %d     \n" % (seconds, days) )
            
                    f.write("    %20.14f  \n" % wind_error**2)
    
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

###############################################################################

temp_error = 0.5
rh_error = 3
wind_error = 0.6
error_lag = 0.5

temp_rep_error = 0
rh_rep_error = 0
wind_rep_error = 0.4 


ascent_rate = 5
# first_flight = datetime(2023,4,19,21,10,0)
# first_DA_time = datetime(2023,4,19,21,15,0)
# last_flight = datetime(2023,4,19,22,10,0)
# last_DA_time = datetime(2023,4,19,22,15,0)
first_flight = datetime(2023,2,27,0,10,0)
first_DA_time = datetime(2023,2,27,0,15,0)
last_flight = datetime(2023,2,27,1,10,0)
last_DA_time = datetime(2023,2,27,1,15,0)
flight_frequency = timedelta(hours=1)
max_wind = 35
delta_z = 500.
da_heights = np.arange(250,3001,500)
output_dir = 'obsseq/UAS/20230226/'

flight_stime = np.arange(first_flight,last_flight+flight_frequency,flight_frequency).astype(datetime)
DA_times = np.arange(first_DA_time, last_DA_time+flight_frequency,flight_frequency).astype(datetime)

flight_stime_epoch = np.array([(x - datetime(1970,1,1)).total_seconds() for x in flight_stime])

file = 'PerfectProfiler/48_UAS_20230227_000000_20230227_013000.nc'

f = Dataset(file)

t = f.variables['temperature'][:]
rh = f.variables['relative_humidity'][:]
u = f.variables['u_wind'][:]
v = f.variables['v_wind'][:]
p = f.variables['pres'][:]
z = np.nanmean(f.variables['height'][0]-f.variables['altitude'][:],axis=1)
alt = f.variables['altitude'][:]
time = f.variables['base_time'][:] + f.variables['time_offset'][:]
lat = f.variables['latitude'][:]
lon = f.variables['longitude'][:]

uas_t = np.ones((len(flight_stime),t.shape[1],t.shape[2]))*np.nan
uas_rh = np.ones((len(flight_stime),t.shape[1],t.shape[2]))*np.nan
uas_u = np.ones((len(flight_stime),t.shape[1],t.shape[2]))*np.nan
uas_v = np.ones((len(flight_stime),t.shape[1],t.shape[2]))*np.nan
uas_pres = np.ones((len(flight_stime),t.shape[1],t.shape[2]))*np.nan
uas_maxz = np.ones((len(flight_stime),t.shape[2]))*np.nan
uas_times = np.ones((len(flight_stime),t.shape[1]))*np.nan

for i in range(len(flight_stime)):
    
    flight_time = flight_stime_epoch[i] + (z[:]/ascent_rate).astype(float)
    t_grid, z_grid = np.meshgrid(flight_time,z[:])
    
    uas_times[i,:] = np.copy(flight_time)
    
    
    for j in range(t.shape[2]):
        
        if np.isnan(t[0,0,j]):
            continue
        
        t_interp = RegularGridInterpolator((time,z),t[:,:,j],bounds_error = False, fill_value=None)
        uas_t[i,:,j] = t_interp((flight_time,z))
        
        rh_interp = RegularGridInterpolator((time,z),rh[:,:,j],bounds_error = False,fill_value=None)
        uas_rh[i,:,j] = rh_interp((flight_time,z))
        
        u_interp = RegularGridInterpolator((time,z),u[:,:,j],bounds_error = False,fill_value=None)
        uas_u[i,:,j] = u_interp((flight_time,z))
        
        v_interp = RegularGridInterpolator((time,z),v[:,:,j],bounds_error = False,fill_value=None)
        uas_v[i,:,j] = v_interp((flight_time,z))
        
        p_interp = RegularGridInterpolator((time,z),p[:,:,j],bounds_error = False,fill_value=None)
        uas_pres[i,:,j] = p_interp((flight_time,z))
        


     
t_err = np.zeros((uas_t.shape[0],uas_t.shape[2]))
rh_err = np.zeros((uas_t.shape[0],uas_t.shape[2]))
u_err = np.zeros((uas_t.shape[0],uas_t.shape[2]))
v_err = np.zeros((uas_t.shape[0],uas_t.shape[2]))     

for i in range(uas_t.shape[1]):
    t_err = np.random.normal(0,temp_error,size=(uas_t.shape[0],uas_t.shape[2])) + error_lag*t_err
    rh_err = np.random.normal(0,rh_error,size=(uas_rh.shape[0],uas_rh.shape[2])) + error_lag*rh_err
    u_err = np.random.normal(0,wind_error,size=(uas_u.shape[0],uas_u.shape[2])) + error_lag*u_err
    v_err = np.random.normal(0,wind_error,size=(uas_v.shape[0],uas_v.shape[2])) + error_lag*v_err
    
    print(t_err[0,0])
    uas_t[:,i,:] += t_err
    uas_rh[:,i,:] += rh_err
    uas_u[:,i,:] += u_err
    uas_v[:,i,:] += v_err

# Make sure random error didn't make bad RH values
uas_rh[uas_rh > 100] = 100
uas_rh[uas_rh <= 0] = 0.1

uas_dew = rh2dpt(uas_t,uas_rh/100)

t4DA = np.ones((len(flight_stime),len(da_heights),t.shape[2]))*np.nan
rh4DA = np.ones((len(flight_stime),len(da_heights),t.shape[2]))*np.nan
u4DA = np.ones((len(flight_stime),len(da_heights),t.shape[2]))*np.nan
v4DA = np.ones((len(flight_stime),len(da_heights),t.shape[2]))*np.nan

for i in range(len(flight_stime)):
    for j in range(uas_t.shape[2]):
        
        foo = np.where(np.sqrt(uas_u[i,:,j]**2 + uas_v[i,:,j]**2) > max_wind)[0]
        if len(foo) == 0:
            uas_maxz[i,j] = z[-1]
        else:
            uas_maxz[i,j] = z[foo[0]]

        foo = np.where(~np.isnan(uas_t[i,:,j]))[0]
        
        if len(foo) == 0:
            continue
    
        for k in range(t4DA.shape[1]):
            da_height_min = da_heights[k]-(delta_z/2)
            da_height_max = da_heights[k]+(delta_z/2)
            
            fah = np.where((z[foo] >= da_height_min) & (z[foo] < da_height_max))[0]
            
            t4DA[i,k,j] = np.nanmean(uas_t[i,foo[fah],j])
            rh4DA[i,k,j] = np.nanmean(uas_rh[i,foo[fah],j])
            u4DA[i,k,j] = np.nanmean(uas_u[i,foo[fah],j])
            v4DA[i,k,j] = np.nanmean(uas_v[i,foo[fah],j])
            
            # t4DA[i,:,j] = np.interp(da_heights,z[foo],uas_t[i,foo,j],left=np.nan,right=np.nan)
            # rh4DA[i,:,j] = np.interp(da_heights,z[foo],uas_rh[i,foo,j],left=np.nan,right=np.nan)        
            # u4DA[i,:,j] = np.interp(da_heights,z[foo],uas_u[i,foo,j],left=np.nan,right=np.nan)
            # v4DA[i,:,j] = np.interp(da_heights,z[foo],uas_v[i,foo,j],left=np.nan,right=np.nan)
        
        
        foo = np.where(uas_maxz[i,j] < z[:])[0]
        
        uas_t[i,foo,j] = np.nan
        uas_rh[i,foo,j] = np.nan
        uas_u[i,foo,j] = np.nan
        uas_v[i,foo,j] = np.nan
        
        foo = np.where(uas_maxz[i,j] < da_heights)[0]
    
            
        t4DA[i,foo,j] = np.nan
        rh4DA[i,foo,j] = np.nan
        u4DA[i,foo,j] = np.nan
        v4DA[i,foo,j] = np.nan

       
dew4DA = rh2dpt(t4DA,rh4DA/100)


uas = {'lat':lat, 'lon':lon, 'alt':alt, 'temp':uas_t,
       'dew':uas_dew, 'u':uas_u, 'v':uas_v, 'rh':uas_rh, 'z':z[:],
       'p':uas_pres}

uas4DA = {'lat':lat, 'lon':lon, 'alt':alt, 'temp':t4DA,
       'dew':dew4DA, 'u':u4DA, 'v':v4DA, 'rh':rh4DA, 'z':da_heights}


write_to_file(uas,output_dir, flight_stime_epoch, uas_times, first_flight, last_flight)

write_dart_obs_seq(uas4DA, output_dir, DA_times,temp_error+temp_rep_error,rh_error+rh_rep_error,wind_error+wind_rep_error)
