import os
import sys
import numpy as np
from netCDF4 import Dataset
from argparse import ArgumentParser
from metpy.interpolate import inverse_distance_to_grid
import glob
import pyproj
import scipy.ndimage as ndimage
from datetime import datetime, timedelta,timezone

def write_dart_obs_seq(ref,ref0,lon,lat,z,dbz0_z,time,output_dir,cm1,ref_errors=5,dbz0_errors=7):
    
    if cm1:
        lats = np.copy(lat)
        lons = np.copy(lon)
    else:
        lats = np.radians(lat)
        lons = np.radians(lon)
        lons = np.where(lons > 0.0, lons, lons+(2.0*np.pi))
    
    vert_coord = 3
    truth = 1.0
    
    # We need to sum up the total number of good ref and dbz0_obs
    data_length = len(np.where(~np.isnan(ref))[0])+len(np.where(~np.isnan(ref0))[0])
    
    if data_length == 0:
        print('No data for assimilation at ' + time.strftime('%Y%m%d_%H%M%S'))
        return
    
    if not os.path.exists(output_dir + '/' + time.strftime('%Y%m%d_%H%M%S')):
            os.makedirs(output_dir + '/' + time.strftime('%Y%m%d_%H%M%S'))

    filename = output_dir +'/' + time.strftime('%Y%m%d_%H%M%S') + '/MRMS_obs_seq_' +  time.strftime('%Y%m%d_%H%M%S')
    
    f = open(filename,"w")
    
    # Start with normal ref
    
    nobs = 0
    for k in range(ref.shape[0]):
        for j in range(ref.shape[1]):
            for i in range(ref.shape[2]):
        
        
                if np.isnan(ref[k,j,i]):
                    pass
                else:
                    nobs += 1
            
                    sw_time = time - datetime(1601,1,1,0,0,0,tzinfo=timezone.utc)
            
                    days = sw_time.days
                    seconds = sw_time.seconds
            
                    f.write(" OBS            %d\n" % (nobs) )
            
                    f.write("   %20.14f\n" % ref[k,j,i] )
                    f.write("   %20.14f\n" % truth  )
            
                    if nobs == 1:
                        f.write(" %d %d %d\n" % (-1, nobs+1, -1) ) # First obs.
                    elif nobs == data_length:
                        f.write(" %d %d %d\n" % (nobs-1, -1, -1) ) # Last obs.
                    else:
                        f.write(" %d %d %d\n" % (nobs-1, nobs+1, -1) )
            
                    f.write("obdef\n")
                    if cm1:
                        f.write("loc3Dxyz")
                        f.write("    %20.14f          %20.14f          %20.14f\n" % 
                            (lons[j,i], lats[j,i], z[k]))
                    else:    
                        f.write("loc3d\n")
                        f.write("    %20.14f          %20.14f          %20.14f     %d\n" % 
                            (lons[j,i], lats[j,i], z[k], vert_coord))
            
                    f.write("kind\n")
                
                    f.write("     %d     \n" % 37 )
            
                    f.write("    %d          %d     \n" % (seconds, days) )
            
                    f.write("    %20.14f  \n" % ref_errors**2 )
    
    # and now do the clear air reflectivity
    for k in range(ref0.shape[0]):
        for j in range(ref0.shape[1]):
            for i in range(ref0.shape[2]):
        
                if np.isnan(ref0[k,j,i]):
                    pass
                else:
                    nobs += 1
                
                    sw_time = time - datetime(1601,1,1,0,0,0,tzinfo=timezone.utc)
                
                    days = sw_time.days
                    seconds = sw_time.seconds
                
                    f.write(" OBS            %d\n" % (nobs) )
            
                    f.write("   %20.14f\n" % (ref0[k,j,i]) )
                    f.write("   %20.14f\n" % truth  )
            
                    if nobs == 1:
                        f.write(" %d %d %d\n" % (-1, nobs+1, -1) ) # First obs.
                    elif nobs == data_length:
                        f.write(" %d %d %d\n" % (nobs-1, -1, -1) ) # Last obs.
                    else:
                        f.write(" %d %d %d\n" % (nobs-1, nobs+1, -1) )
                    
                    f.write("obdef\n")
                    if cm1:
                        f.write("loc3Dxyz")
                        f.write("    %20.14f          %20.14f          %20.14f\n" % 
                            (lons[j,i], lats[j,i], z[k]))
                    else:    
                        f.write("loc3d\n")
                        f.write("    %20.14f          %20.14f          %20.14f     %d\n" % 
                            (lons[j,i], lats[j,i], z[k], vert_coord))
            
                    f.write("kind\n")
            
                    f.write("     %d     \n" % 38 )
            
                    f.write("    %d          %d     \n" % (seconds, days) )
                
                    f.write("    %20.14f  \n" % dbz0_errors**2 )
    
    f.close()
    
    # Now write out header informations
    f = open(filename,'r')
    f_obs_seq = f.read()
    f.close()
    
    f = open(filename,'w')
    
    f.write(" obs_sequence\n")
    f.write("obs_kind_definitions\n")
    f.write("       %d\n" % 2)
    f.write("    %d          %s   \n" % (37, "RADAR_REFLECTIVITY") )
    f.write("    %d          %s   \n" % (38, "RADAR_CLEARAIR_REFLECTIVITY") )
    f.write("  num_copies:            %d  num_qc:            %d\n" % (1, 1))
    f.write(" num_obs:       %d  max_num_obs:       %d\n" % (nobs, nobs) )
    f.write("observations\n")
    f.write("QC obs\n")
    f.write("  first:            %d  last:       %d\n" % (1, nobs) )
    
    f.write(f_obs_seq)
  
    f.close()

parser = ArgumentParser()

parser.add_argument("input_file", help="Path to the simulated MRMS data")
parser.add_argument("output_dir", help="Output directory for the obs_seq files")
parser.add_argument("grid_spacing", type=int, help="Grid spacing for obs to be assimilated")
parser.add_argument("zeros_thinning",type=int, help="Thinning factor for 0dBZ obs")
parser.add_argument("zeros_levels",help='Levels for the 0dBZ obs')
parser.add_argument('ref_threshold',type=float,help='Threshold for the minimum value of reflectvity')
parser.add_argument("--cm1",action="store_true",help='Use timing and spatial information for a CM1 style model')
parser.add_argument("--set_DA_time",type=str, help="Overwrite the basetime in the obs file")

args = parser.parse_args()

input_file = args.input_file
output_dir = args.output_dir
delta_x = args.grid_spacing*1000
dbz0_thinning = args.zeros_thinning
dbz0_levels = np.array(args.zeros_levels.strip().split(',')).astype(float)
ref_min = args.ref_threshold
cm1 = args.cm1
DA_time = args.set_DA_time

f = Dataset(input_file)

if not cm1:
    time = f.variables['base_time'][0] + f.variables['time_offset'][:]
    time = np.array([datetime.fromtimestamp(ts,tz=timezone.utc) for ts in time])
elif DA_time is not None:
    DA_time_dt = (datetime.strptime(DA_time,'%Y%m%d%H%M%S')- datetime(1970,1,1)).total_seconds()
    time = (DA_time_dt) + f.variables['time_offset'][:].astype(float)
    time = np.array([datetime.fromtimestamp(ts,tz=timezone.utc) for ts in time])

ref = f.variables['reflectivity'][:]
z = f.variables['altitude'][:]
x = f.variables['x'][:]
y = f.variables['y'][:]

if not cm1:
    map = pyproj.Proj(proj=f.projection, ellps='WGS84', datum='WGS84', lat_1=f.truelat1, lat_2=f.truelat2, lat_0=f.lat0, lon_0=f.lon0)

f.close()

# Make the grid for the data
x_grid = np.arange(np.min(x),np.max(x)+1,delta_x)
y_grid = np.arange(np.min(y),np.max(y)+1,delta_x)
x_grid, y_grid = np.meshgrid(x_grid,y_grid)

if cm1:
    lons = np.copy(x_grid)
    lats = np.copy(y_grid)
else:
    lons, lats = map(x_grid, y_grid, inverse=True)

cref_grid = np.ones((len(z),x_grid.shape[0],x_grid.shape[1]))*np.nan
# Loop over all of the times
for k in range(ref.shape[0]):

    ref_grid = np.ones((len(z),x_grid.shape[0],x_grid.shape[1]))*np.nan
    ref0_grid = np.zeros((len(dbz0_levels),x_grid.shape[0],x_grid.shape[1]))

    # Do the Cressman analysis for the data on this new grid
    cr = delta_x*np.sqrt(2)
    for i in range(len(z)):
        # I need a better way to do this, but this array is only making sure that
        # the zeros don't extend outside of the radar range
        if k == 0:
            cref_grid[i] = inverse_distance_to_grid(x.ravel(),y.ravel(),ref[k,i].ravel(),x_grid,y_grid,cr)

        # We are masking out nans and super low ref values before the superob-ing
        foo = np.where(((~np.isnan(ref[k,i])) & (ref[k,i] >= 5)))
        if len(foo[0]) > 0:
            ref_grid[i] = inverse_distance_to_grid(x[foo].ravel(),y[foo].ravel(),ref[k,i][foo].ravel(),x_grid,y_grid,cr)
        else:
            ref_grid[i,:,:] = np.nan

    # We need to find where we want to assimilate the zeros at
    cref = np.nanmax(ref_grid,axis=0)
    og_mask = np.isnan(np.nanmax(cref_grid,axis=0))

    cref[np.isnan(cref)] = 0
    cref = np.where(cref < ref_min,0,cref)
    max_neighbor = (ndimage.maximum_filter(cref, size=3) > 0.1)
    zero_dbz_mask = np.where(max_neighbor, True, False)

    if dbz0_thinning > 1:
        mask2 = np.logical_not(zero_dbz_mask)
        mask2[::dbz0_thinning,::dbz0_thinning] = False
        zero_dbz_mask = np.logical_or(zero_dbz_mask,mask2)

    zero_dbz_mask = np.logical_or(zero_dbz_mask,og_mask)

    # NaN out the masked out zeros
    for i in range(len(dbz0_levels)):
        ref0_grid[i][zero_dbz_mask] = np.nan
    
    # Nan out where ref_grid is less than the threshold
    ref_grid[ref_grid < ref_min] = np.nan

    # Now write the DART file
    write_dart_obs_seq(ref_grid,ref0_grid,lons,lats,z,dbz0_levels,time[k],output_dir,cm1)