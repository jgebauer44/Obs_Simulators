#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 15:18:09 2024

@author: joshua.gebauer
"""
import numpy as np
from netCDF4 import Dataset
from datetime import datetime, timedelta

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

def write_dart_obs_seq(mesonet,output_dir,temp_error,wind_error,time):
    
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
            
            f.write("    %20.14f  \n" % temp_error**2 )
    
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
            
            f.write("    %20.14f  \n" % wind_error**2)
    
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
            
            f.write("    %20.14f  \n" % wind_error**2)
    
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

###############################################################################

station_file = '/Users/joshua.gebauer/SyntheticMesonet/OK_Mesonet_12sites.txt'
obs_file = '/Users/joshua.gebauer/Test_SyntheticMesonet/20230419/OK_Mesonet/OK_Mesonet_20230419_210000_20230419_223000.nc'
output_dir = '/Users/joshua.gebauer/Test_SyntheticMesonet/20230419/12sites/'

temp_error = 1.75
rh_error = 0.20
wind_error = 1.75

times = ([datetime(2023,4,19,21,0,0),
         datetime(2023,4,19,21,15,0),
         datetime(2023,4,19,21,30,0),
         datetime(2023,4,19,21,45,0),
         datetime(2023,4,19,22,0,0),
         datetime(2023,4,19,22,15,0),
         datetime(2023,4,19,22,30,0)
         ])

stations = np.genfromtxt(station_file, delimiter= ' ',autostrip=True)

f = Dataset(obs_file)

lats = f.variables['latitude'][:]
lons = f.variables['longitude'][:]
alt = f.variables['altitude'][:]
temp = f.variables['temperature'][:]
dew = f.variables['dewpoint'][:]
rh = f.variables['relative_humidity'][:]
u = f.variables['u_wind'][:]
v = f.variables['v_wind'][:]
psfc = f.variables['surface_pressure'][:]

indices = []
for i in range(len(stations)):
    foo = np.where(lats == stations[i,0])[0]
    indices.append(foo[0])

for i in range(len(times)):
    
    mesonet = {'lat':lats[indices],'lon':lons[indices],'alt':alt[indices],'temp':temp[i,indices],
               'dew':dew[i,indices],'rh':rh[i,indices],'u':u[i,indices],'v':v[i,indices],
               'psfc':psfc[i,indices]}
    
    write_dart_obs_seq(mesonet, output_dir, temp_error, wind_error, times[i])
        
        
        


