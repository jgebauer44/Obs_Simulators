#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 10:20:03 2023

@author: joshua.gebauer
"""

import numpy as np
from netCDF4 import Dataset
from datetime import datetime,timedelta
import glob

def calc_dewpoint_error(w, p, w_error):
    
    delta_w = 0.0001
    
    w_tmp = w/1000.
    
    w1 = w_tmp-delta_w
    w2 = w_tmp+delta_w
    
    td_deriv_w = ( w2dpt(w2,p)-w2dpt(w1,p) )/(w2-w1)
    
    td_error = (w_error/1000) * td_deriv_w
    
    return td_error

def w2e(w,p):
    e = p * w / (0.622 + w)
    return e

def w2dpt(w,p):
    
    e = w2e(w,p)
    val = np.log(e/6.112)
    dew = 243*val/(17.67 - val)
    
    return dew

def write_obs_seq(t,sigt,xt,yt,zt,ttime,w,sigw,xw,yw,zw,wtime,p,output_dir):
    
    xt_lon = np.radians(xt)
    yt_lat = np.radians(yt)
    
    xw_lon = np.radians(xw)
    yw_lat = np.radians(yw)
    
    vert_coord = 3
    truth = 1.0
    
    xt_lon = np.where(xt_lon > 0.0, xt_lon, xt_lon+(2.0*np.pi))
    xw_lon = np.where(xw_lon > 0.0, xw_lon, xw_lon+(2.0*np.pi))
    
    # We are adding temperature and mixing ratio
    data_length = len(t)+len(w)
    
    # Convert temperature to Kelvin
    t += 273.15
    
    filename = output_dir + 'profiler_obs_seq_' + ttime[0].strftime('%Y%m%d_%H%M%S')
    
    f = open(filename,"w")
    
    # Start with temperature
    
    nobs = 0
    
    for i in range(len(t)):
        
        if np.isnan(t[i]):
            pass
        else:
            nobs += 1
            
            sw_time = ttime[i] - datetime(1601,1,1,0,0,0)
            days = sw_time.days
            seconds = sw_time.seconds
            
            f.write(" OBS            %d\n" % (nobs) )
           
            f.write("   %20.14f\n" % t[i] )
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
                  (xt_lon[i], yt_lat[i], zt[i], vert_coord))

            f.write("kind\n")
            
            f.write("     %d     \n" % 14 )
            
            f.write("    %d          %d     \n" % (seconds, days) )
            
            f.write("    %20.14f  \n" % (sigt[i])**2 )
            
    # Now do specific humidity
    
    for i in range(len(w)):
        
        if np.isnan(w[i]):
            pass
        elif (w2dpt(w[i]/1000, p[i]) + 273.15) < 250:    
            pass
        else:
            nobs += 1
        
            sw_time = wtime[i] - datetime(1601,1,1,0,0,0)
            days = sw_time.days
            seconds = sw_time.seconds
            
            f.write(" OBS            %d\n" % (nobs) )
            
            dew_tmp = w2dpt(w[i]/1000, p[i]) + 273.15
            f.write("   %20.14f\n" % dew_tmp )
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
                  (xw_lon[i], yw_lat[i], zw[i], vert_coord))

            f.write("kind\n")
            
            f.write("     %d     \n" % 15 )
            
            f.write("    %d          %d     \n" % (seconds, days) )
            
            sigdew = calc_dewpoint_error(w[i], p[i], sigw[i])
            f.write("    %20.14f  \n" % (sigdew)**2 )
            
    f.close()
    
    # Now write out header information
    f = open(filename,'r')
    f_obs_seq = f.read()
    f.close()
    
    f = open(filename,'w')
    
    f.write(" obs_sequence\n")
    f.write("obs_kind_definitions\n")
    
    f.write("       %d\n" % 2)
    f.write("    %d          %s   \n" % (14, "RADIOSONDE_TEMPERATURE") )
    f.write("    %d          %s   \n" % (15, "RADIOSONDE_DEWPOINT") )
    f.write("  num_copies:            %d  num_qc:            %d\n" % (1, 1))
    f.write(" num_obs:       %d  max_num_obs:       %d\n" % (nobs, nobs) )
    f.write("observations\n")
    f.write("QC obs\n")
    f.write("  first:            %d  last:       %d\n" % (1, nobs) )
    
    f.write(f_obs_seq)
  
    f.close()
    
start_date = datetime(2024,5,8,18,0,0)
end_date = datetime(2024,5,8,20,0,0)
step = timedelta(minutes=15)

tropoe_path = '/work/joshua.gebauer/SimObs/8May2024/obs_nc/TROPoeSim_Output/irs_recenter/'

tfiles = []
tfiles = tfiles + sorted(glob.glob(tropoe_path + '*' + start_date.strftime("%Y%m%d") + '*.cdf'))
output_dir = '/work/joshua.gebauer/SimObs/8May2024/obs_seq/irs_recenter/'

curr = start_date

while curr <= end_date:
    
    t_4DA = []
    w_4DA = []
    p_4DA = [] 
    zt_4DA = []
    xt_4DA = []
    yt_4DA = []
    zw_4DA = []
    xw_4DA = []
    yw_4DA = []
    ttime_4DA = []
    wtime_4DA = []
    sigmat_4DA = []
    sigmaw_4DA = []
    
    for i in range(len(tfiles)):
        
        f = Dataset(tfiles[i])
        
        da_time = (curr - datetime(1970,1,1)).total_seconds()
        
        time = f.variables['base_time'][0] + f.variables['time_offset'][:]
        
        foo = np.where(np.isclose(da_time,time,rtol=1e-12))[0]
        
        if len(foo) == 0:
            continue
            
        
        lon = f.variables['lon'][:]
        lat = f.variables['lat'][:]
        alt = f.variables['alt'][:]
        
        t = np.squeeze(f.variables['temperature'][foo,:])
        w = np.squeeze(f.variables['waterVapor'][foo,:])
        p = np.squeeze(f.variables['pressure'][foo,:])
        z = f.variables['height'][:]
        sigmat = np.squeeze(f.variables['sigma_temperature'][foo,:])
        sigmaw = np.squeeze(f.variables['sigma_waterVapor'][foo,:])
        
        t_dof = np.squeeze(f.variables['cdfs_temperature'][foo,:])
        w_dof = np.squeeze(f.variables['cdfs_waterVapor'][foo,:])
        
        cbh = f.variables['cbh'][foo]
        lwp = f.variables['lwp'][foo]
        
        # Do temperature first
        floor_t_dof = np.floor(t_dof)
        max_dof = np.max(floor_t_dof)
        
        for j in range(int(max_dof)):
            
            foo = np.where(((j+1 == floor_t_dof) & (z < 5)))[0]
            
            if len(foo) > 0:
                if (z[foo[0]] < cbh) or (lwp < 5):
                    
                    t_4DA.append(t[foo[0]])
                    zt_4DA.append(z[foo[0]]*1000+alt)
                    xt_4DA.append(lon)
                    yt_4DA.append(lat)
                    sigmat_4DA.append(sigmat[foo[0]])
                    ttime_4DA.append(curr)
                
        # Now water vapor
        floor_w_dof = np.floor(w_dof)
        max_dof = np.max(floor_w_dof)
        
        for j in range(int(max_dof)):
            
            foo = np.where(((j+1 == floor_w_dof) & (z < 5)))[0]
            
            if len(foo) > 0:
                if (z[foo[0]] < cbh) or (lwp < 5):
                    w_4DA.append(w[foo[0]])
                    p_4DA.append(p[foo[0]])
                    zw_4DA.append(z[foo[0]]*1000+alt)
                    xw_4DA.append(lon)
                    yw_4DA.append(lat)
                    sigmaw_4DA.append(sigmaw[foo[0]])
                    wtime_4DA.append(curr)
        
    
    # Now write the data to a obs_seq file
    t_4DA = np.array(t_4DA)
    zt_4DA = np.array(zt_4DA)
    xt_4DA = np.array(xt_4DA)
    yt_4DA = np.array(yt_4DA)
    sigmat_4DA = np.array(sigmat_4DA)
    ttime_4DA = np.array(ttime_4DA)
    
    w_4DA = np.array(w_4DA)
    p_4DA = np.array(p_4DA)
    zw_4DA = np.array(zw_4DA)
    xw_4DA = np.array(xw_4DA)
    yw_4DA = np.array(yw_4DA)
    sigmaw_4DA = np.array(sigmaw_4DA)
    wtime_4DA = np.array(wtime_4DA)
    
    
    write_obs_seq(t_4DA,sigmat_4DA,xt_4DA,yt_4DA,zt_4DA,ttime_4DA,w_4DA,sigmaw_4DA,
                  xw_4DA,yw_4DA,zw_4DA,wtime_4DA,p_4DA,output_dir)
    
    curr += timedelta(minutes=15)
        
        
                
                
                
        
        
        
        
        
        
        
