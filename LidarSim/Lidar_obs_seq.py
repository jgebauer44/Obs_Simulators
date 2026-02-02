#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 11:33:13 2024

@author: joshua.gebauer
"""

import numpy as np
from netCDF4 import Dataset
from datetime import datetime,timedelta
import glob

def write_obs_seq(u,sigu,v,sigv,x,y,z,time,output_dir):
    
    x_lon = np.radians(x)
    y_lat = np.radians(y)
    
    vert_coord = 3
    truth = 1.0
    
    x_lon = np.where(x_lon > 0.0, x_lon, x_lon+(2.0*np.pi))
    
    # We are adding u and v
    data_length = len(np.where(~np.isnan(u))[0])*2
    print(time)
    filename = output_dir + 'lidar_obs_seq_' + time[0].strftime('%Y%m%d_%H%M%S')
    
    if len(u.shape) == 1:
        u = np.array([u])
        v = np.array([v])
        sigu = np.array([sigu])
        sigv = np.array([sigv])
        
    f = open(filename,"w")
    
    # Start with temperature
    
    nobs = 0
    
    for j in range(u.shape[0]):
        for i in range(u.shape[1]):
        
            if np.isnan(u[j,i]):
                pass
            else:
                nobs += 1
            
                sw_time = time[j] - datetime(1601,1,1,0,0,0)
                days = sw_time.days
                seconds = sw_time.seconds
            
                f.write(" OBS            %d\n" % (nobs) )
           
                f.write("   %20.14f\n" % u[j,i] )
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
                        (x_lon[j], y_lat[j], z[j,i], vert_coord))

                f.write("kind\n")
            
                f.write("     %d     \n" % 16 )
            
                f.write("    %d          %d     \n" % (seconds, days) )
                
                if sigu[j,i] <= 1.0:
                    f.write("    %20.14f  \n" % 1.0**2 )
                else:
                    f.write("    %20.14f  \n" % sigu[j,i]**2 )
            
    for j in range(v.shape[0]):
        for i in range(v.shape[1]):
        
            if np.isnan(v[j,i]):
                pass
            else:
                nobs += 1
            
                sw_time = time[j] - datetime(1601,1,1,0,0,0)
                days = sw_time.days
                seconds = sw_time.seconds
            
                f.write(" OBS            %d\n" % (nobs) )
           
                f.write("   %20.14f\n" % v[j,i] )
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
                        (x_lon[j], y_lat[j], z[j,i], vert_coord))

                f.write("kind\n")
            
                f.write("     %d     \n" % 17 )
            
                f.write("    %d          %d     \n" % (seconds, days) )
                
                if sigu[j,i] <= 1.0:
                    f.write("    %20.14f  \n" % 1.0**2 )
                else:
                    f.write("    %20.14f  \n" % sigv[j,i]**2 )
                    
    # Now write out header information
    f = open(filename,'r')
    f_obs_seq = f.read()
    f.close()
    
    f = open(filename,'w')
    
    f.write(" obs_sequence\n")
    f.write("obs_kind_definitions\n")
    
    f.write("       %d\n" % 2)
    f.write("    %d          %s   \n" % (16, "RADIOSONDE_U_WIND_COMPONENT") )
    f.write("    %d          %s   \n" % (17, "RADIOSONDE_V_WIND_COMPONENT") )
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

lidar_path = '/work/joshua.gebauer/SimObs/8May2024/obs_nc/lidar_vad/'

tfiles = []
tfiles = tfiles + sorted(glob.glob(lidar_path + '*' + '.nc'))
output_dir = '/work/joshua.gebauer/SimObs/8May2024/obs_seq/lidar/'
print(tfiles)
r_cutoff = 4.0
max_height = 5000
z_spacing = 500

interp_z = np.arange(0,max_height+1,z_spacing)

curr = start_date

while curr <= end_date:
    
    u_4DA = []
    v_4DA = []
    x_4DA = []
    y_4DA = []
    z_4DA = []
    ttime_4DA = []
    sigmau_4DA = []
    sigmav_4DA = []
    
    for i in range(len(tfiles)):
        
        f = Dataset(tfiles[i])
        
        da_time = (curr - datetime(1970,1,1)).total_seconds()
        
        time = f.variables['base_time'][0] + f.variables['time_offset'][:]
        
        foo = np.where(np.isclose(da_time,time,rtol=1e-12))[0]
        
        if len(foo) == 0:
            continue
            
        lon = f.variables['x_position'][:]
        lat = f.variables['y_position'][:]
        alt = f.variables['alt'][:]
        
        u = np.squeeze(f.variables['u'][foo,:])
        v = np.squeeze(f.variables['v'][foo,:])
        z = f.variables['height'][:]
        sigmau = np.squeeze(f.variables['u_error'][foo,:])
        sigmav = np.squeeze(f.variables['v_error'][foo,:])
        residual = np.squeeze(f.variables['residual'][foo,:])
        
        u_interp = np.interp(interp_z,z,u,left=np.nan, right=np.nan)
        v_interp = np.interp(interp_z,z,v,left=np.nan,right=np.nan)
        sigmau_interp = np.interp(interp_z,z,sigmau,left=np.nan,right=np.nan)
        sigmav_interp = np.interp(interp_z,z,sigmav,left=np.nan,right=np.nan)
        res_interp = np.interp(interp_z,z,residual,left=np.nan,right=np.nan)
        
        foo = np.where(res_interp > r_cutoff)
        u_interp[foo] = np.nan
        v_interp[foo] = np.nan
        sigmau_interp[foo] = np.nan
        sigmav_interp[foo] = np.nan
        
        u_4DA.append(np.copy(u_interp))
        v_4DA.append(np.copy(v_interp))
        x_4DA.append(np.copy(lon))
        y_4DA.append(np.copy(lat))
        z_4DA.append(alt+interp_z)
        ttime_4DA.append(curr)
        sigmau_4DA.append(np.copy(sigmau_interp))
        sigmav_4DA.append(np.copy(sigmav_interp))
        
    u_4DA = np.array(u_4DA)
    v_4DA = np.array(v_4DA)
    x_4DA = np.array(x_4DA)
    y_4DA = np.array(y_4DA)
    z_4DA = np.array(z_4DA)
    ttime_4DA = np.array(ttime_4DA)
    sigmau_4DA = np.array(sigmau_4DA)
    sigmav_4DA = np.array(sigmav_4DA)
    
    write_obs_seq(u_4DA,sigmau_4DA,v_4DA,sigmav_4DA,x_4DA,y_4DA,z_4DA,ttime_4DA,output_dir)
    
    curr += timedelta(minutes=15)
        
        
