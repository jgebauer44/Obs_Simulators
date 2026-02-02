#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 13:01:30 2021

@author: jgebauer
"""

import glob 
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import datetime as dt
import Lidar_functions
from netCDF4 import Dataset

lidar_path = '/work/joshua.gebauer/SimObs/8May2024/obs_nc/lidar/'

lfiles = []
lfiles = lfiles + sorted(glob.glob(lidar_path + '*' + '.nc'))

file_path = '/work/joshua.gebauer/SimObs/8May2024/obs_nc/lidar_vad/'

#fname = '/Users/joshua.gebauer/LidarSim_Data/NORM_70deg_VAD.nc'

#f = Dataset(fname)

for i in range(len(lfiles)):
    
    f = Dataset(lfiles[i])
    
    base_times = f['base_time'][0]

    vad_time = np.mean(f.variables['scan1_time'] + base_times,axis=1)
    ranges = f.variables['ranges'][:]
    elevation = f.variables['scan1_el'][0]
    azimuth = f.variables['scan1_az'][:]
    scan1 = f.variables['scan1'][:]

    x = float(f.lidar_lon)
    y = float(f.lidar_lat)
    altitude = float(f.lidar_alt)

    f.close()

    VAD = Lidar_functions.process_LidarSim_scan(scan1,'VAD',elevation,azimuth,ranges,vad_time)

    #Lidar_functions.plot_VAD(VAD,'/Users/joshua.gebauer/LidarSim_Figures/CM1_550000_250000/')

    nc_file = Dataset(file_path + lfiles[i].split('/')[-1][:-3] + '.nc' ,'w',format='NETCDF4')
    nc_file.createDimension('time',None)
    nc_file.createDimension('height',len(VAD.z[2:]))

    base_time = nc_file.createVariable('base_time','i')
    base_time[:] = base_times
    base_time.string = 'Model start time'
    base_time.long_name = 'Base time in model time'
    base_time.units =  'seconds since start of model'
    base_time.ancillary_variables = 'time_offset'

    time_offset = nc_file.createVariable('time_offset','d','time')
    time_offset[:] = vad_time - base_time
    time_offset.long_name = 'Time offset from base_time'
    time_offset.units = 'seconds' 
    time_offset.ancillary_variables = "base_time"
    
    height = nc_file.createVariable('height','f','height')
    height[:] = VAD.z[2:]
    height.long_name = 'Height above ground level'
    height.units = 'm'
    height.standard_name = 'height'
    
    elevation_angle = nc_file.createVariable('elevation_angle','f','time')
    elevation_angle[:] = VAD.el
    elevation_angle.long_name = 'Beam elevation angle'
    elevation_angle.units = 'degree'
    
    nbeams = nc_file.createVariable('nbeams','i','time')
    nbeams[:] = VAD.nbeams
    nbeams.long_name = 'Number of beams (azimuth angles) used in wind vector estimations'
    nbeams.units = 'unitless'
    
    u = nc_file.createVariable('u','f',('time','height'))
    u[:,:] = VAD.u[:,2:]
    u.long_name = 'Eastward component of wind vector'
    u.units = 'm/s'
    
    u_error = nc_file.createVariable('u_error','f',('time','height'))
    u_error[:,:] = VAD.du[:,2:]
    u_error.long_name = 'Estimated error in eastward component of wind vector'
    u_error.units = 'm/s'

    v = nc_file.createVariable('v','f',('time','height'))
    v[:,:] = VAD.v[:,2:]
    v.long_name = 'Northward component of wind vector'
    v.units = 'm/s'
    
    v_error = nc_file.createVariable('v_error','f',('time','height'))
    v_error[:,:] = VAD.dv[:,2:]
    v_error.long_name = 'Estimated error in northward component of wind vector'
    v_error.units = 'm/s'

    w = nc_file.createVariable('w','f',('time','height'))
    w[:,:] = VAD.w[:,2:]
    w.long_name = 'Vertical component of wind vector'
    w.units = 'm/s'

    w_error = nc_file.createVariable('w_error','f',('time','height'))
    w_error[:,:] = VAD.dw[:,2:]
    w_error.long_name = 'Estimated error in vertical component of wind vector'
    w_error.units = 'm/s'

    wind_speed = nc_file.createVariable('wind_speed','f',('time','height'))
    wind_speed[:,:] = VAD.speed[:,2:]
    wind_speed.long_name = 'Wind speed'
    wind_speed.units = 'm/s'

    wind_direction = nc_file.createVariable('wind_direction','f',('time','height'))
    wind_direction[:,:] = VAD.wdir[:,2:]
    wind_direction.long_name = 'Wind direction'
    wind_direction.units = 'degree'

    residual = nc_file.createVariable('residual','f',('time','height'))
    residual[:,:] = VAD.residual[:,2:]
    residual.long_name = 'Fit residual'
    residual.units = 'm/s'

    correlation = nc_file.createVariable('correlation','f',('time','height'))
    correlation[:,:] = VAD.correlation[:,2:]
    correlation.long_name = 'Fit correlation coefficient'
    correlation.units = 'unitless'

    xx = nc_file.createVariable('x_position','f')
    xx[:] = x
    xx.long_name = 'Position in east-west direction'
    xx.units = 'm'

    yy = nc_file.createVariable('y_position','f')
    yy[:] = y
    yy.long_name = 'Position in north-south direction'
    yy.units = 'm'
  
    alt = nc_file.createVariable('alt','f')
    alt[:] = altitude
    alt.long_name = 'Altitude above mean sea level'
    alt.units = 'm'
    alt.standard_name = 'altitude'

    nc_file.history = 'created on ' + dt.datetime.utcnow().strftime('%Y/%m/%d %H:%M:%S UTC')

    nc_file.close()


