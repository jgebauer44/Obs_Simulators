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

def beam_elv(sfc_range, z):
    eradius=6371000.
    frthrde=(4.*eradius/3.)

    if sfc_range > 0.0:
        hgtdb = frthrde + z
        rngdb = sfc_range/frthrde

        elvrad = np.arctan((hgtdb*np.cos(rngdb) - frthrde)/(hgtdb * np.sin(rngdb)))

        return np.rad2deg(elvrad)

    else:

        return -999.
    
def write_dart_obs_seq(radar,x,y,lon,lat,z,time,output_dir,site_id,
                       site_lon,site_lat,site_alt,write_ref,dbz0_z,
                       vr_errors = 3,ref_errors=5,dbz0_errors=7):
    
    lats = np.radians(lat)
    lons = np.radians(lon)
    hgts = z + site_alt
    vert_coord = 3
    truth = 1.0
    
    lons = np.where(lons > 0.0, lons, lons+(2.0*np.pi))
    
    platform_lat        = np.radians(site_lat)
    platform_lon        = np.radians(site_lon)

    if platform_lon < 0:
         platform_lon = platform_lon+(2*np.pi)
    
    platform_key        = 1
    platform_vert_coord = 3

    # We need to sum up the total number of good ref and dbz0_obs
    data_length = len(np.where(~np.isnan(radar['vr_grid']))[0])
    if write_ref:
        data_length += len(np.where(~np.isnan(radar['ref_grid']))[0])+len(np.where(~np.isnan(radar['ref0_grid']))[0])
    
    if data_length == 0:
        print('No data for assimilation at ' + time.strftime('%Y%m%d_%H%M%S'))
        return
    
    if not os.path.exists(output_dir + '/' + time.strftime('%Y%m%d_%H%M%S')):
            os.makedirs(output_dir + '/' + time.strftime('%Y%m%d_%H%M%S'))

    filename = output_dir +'/' + time.strftime('%Y%m%d_%H%M%S') + '/' + site_id + '_obs_seq_' +  time.strftime('%Y%m%d_%H%M%S')
    
    f = open(filename,"w")
    
    # Start with velocity
    
    nobs = 0
    for k in range(radar['vr_grid'].shape[0]):
        for j in range(radar['vr_grid'].shape[1]):
            for i in range(radar['vr_grid'].shape[2]):
        
                if np.isnan(radar['vr_grid'][k,j,i]):
                    pass
                else:
                    nobs += 1
            
                    sw_time = time - datetime(1601,1,1,0,0,0,tzinfo=timezone.utc)
            
                    days = sw_time.days
                    seconds = sw_time.seconds
            
                    f.write(" OBS            %d\n" % (nobs) )
            
                    f.write("   %20.14f\n" % radar['vr_grid'][k,j,i] )
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
                            (lons[j,i], lats[j,i], hgts[k,j,i], vert_coord))
            
                    f.write("kind\n")

                    f.write("     %d     \n" % 36 )

                    R_xy = np.sqrt(x[j,i]**2 + y[j,i]**2)
                    elevation_angle = beam_elv(R_xy, z[k,j,i])
                    platform_dir1 = (x[j,i] / R_xy) * np.cos(np.deg2rad(elevation_angle))
                    platform_dir2 = (y[j,i] / R_xy) * np.cos(np.deg2rad(elevation_angle))
                    platform_dir3 = np.sin(np.deg2rad(elevation_angle))

                    f.write("platform\n")
                    f.write("loc3d\n")

                    f.write("    %20.14f          %20.14f        %20.14f    %d\n" % 
                            (platform_lon, platform_lat, site_alt, platform_vert_coord) )
          
                    f.write("dir3d\n")
          
                    f.write("    %20.14f          %20.14f        %20.14f\n" % (platform_dir1, platform_dir2, platform_dir3) )
                    f.write("    %20.14f     \n" % 100 )
                    f.write("    %d          \n" % platform_key )

                    f.write("    %d          %d     \n" % (seconds, days) )
            
                    f.write("    %20.14f  \n" % vr_errors**2 )

    if write_ref:
        for k in range(radar['ref_grid'].shape[0]):
            for j in range(radar['ref_grid'].shape[1]):
                for i in range(radar['ref_grid'].shape[2]):
        
                    if np.isnan(radar['ref_grid'][k,j,i]):
                        pass
                    else:
                        nobs += 1
            
                        sw_time = time - datetime(1601,1,1,0,0,0,tzinfo=timezone.utc)
            
                        days = sw_time.days
                        seconds = sw_time.seconds
            
                        f.write(" OBS            %d\n" % (nobs) )
            
                        f.write("   %20.14f\n" % radar['ref_grid'][k,j,i] )
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
                                (lons[j,i], lats[j,i], hgts[k,j,i], vert_coord))
            
                        f.write("kind\n")
                
                        f.write("     %d     \n" % 37 )
            
                        f.write("    %d          %d     \n" % (seconds, days) )
            
                        f.write("    %20.14f  \n" % ref_errors**2 )
    
        # and now do the clear air reflectivity
        for k in range(radar['ref0_grid'].shape[0]):
            for j in range(radar['ref0_grid'].shape[1]):
                for i in range(radar['ref0_grid'].shape[2]):
        
                    if np.isnan(radar['ref0_grid'][k,j,i]):
                        pass
                    else:
                        nobs += 1
                
                        sw_time = time - datetime(1601,1,1,0,0,0,tzinfo=timezone.utc)
                
                        days = sw_time.days
                        seconds = sw_time.seconds
                
                        f.write(" OBS            %d\n" % (nobs) )
            
                        f.write("   %20.14f\n" % (radar['ref0_grid'][k,j,i]) )
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
                            (lons[j,i], lats[j,i], dbz0_z[k], vert_coord))
            
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
    if write_ref:
        f.write("       %d\n" % 3)
        f.write("    %d          %s   \n" % (36, "DOPPLER_RADIAL_VELOCITY") )
        f.write("    %d          %s   \n" % (37, "RADAR_REFLECTIVITY") )
        f.write("    %d          %s   \n" % (38, "RADAR_CLEARAIR_REFLECTIVITY") )
    else:
        f.write("       %d\n" % 1)
        f.write("    %d          %s   \n" % (36, "DOPPLER_RADIAL_VELOCITY") )
    f.write("  num_copies:            %d  num_qc:            %d\n" % (1, 1))
    f.write(" num_obs:       %d  max_num_obs:       %d\n" % (nobs, nobs) )
    f.write("observations\n")
    f.write("QC obs\n")
    f.write("  first:            %d  last:       %d\n" % (1, nobs) )
    
    f.write(f_obs_seq)
  
    f.close()

parser = ArgumentParser()

parser.add_argument("input_dir", help="Path to the simulated MRMS data")
parser.add_argument("output_dir", help="Output directory for the obs_seq files")
parser.add_argument("grid_spacing", type=int, help="Grid spacing for obs to be assimilated")
parser.add_argument('ref_threshold',type=float,help='Threshold for the minimum value of reflectvity for data to used')
parser.add_argument("--write_ref", action="store_true", help="Set this to write the reflectivity obs to obs_seq")
parser.add_argument("--zeros_thinning",type=int, help="Thinning factor for 0dBZ obs")
parser.add_argument("--zeros_levels",help='Levels for the 0dBZ obs')

args = parser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir
delta_x = args.grid_spacing*1000
write_ref = args.write_ref
dbz0_thinning = args.zeros_thinning
dbz0_levels = np.array(args.zeros_levels.strip().split(',')).astype(float)
ref_min = args.ref_threshold

files = []
files = files + glob.glob(input_dir + '/*.nc')

radars = []
keys = []
for i in range(len(files)):
    f = Dataset(files[i])

    time = f.variables['base_time'][0] + f.variables['time_offset'][:]
    time = np.array([datetime.fromtimestamp(ts,tz=timezone.utc) for ts in time])

    ref = f.variables['reflectivity'][:]
    vr = f.variables['radial_velocity'][:]
    z = f.variables['altitude'][:]
    x = f.variables['x'][:]
    y = f.variables['y'][:]

    site_id = f.site_id
    site_lon = f.site_lon
    site_lat = f.site_lat
    site_alt = f.site_alt

    map = pyproj.Proj(proj=f.projection, ellps='WGS84', datum='WGS84', lat_1=f.truelat1, lat_2=f.truelat2, lat_0=site_lat, lon_0=site_lon)

    f.close()

    # Make the grid for the data
    x_grid = np.arange(np.min(x),np.max(x)+1,delta_x)
    y_grid = np.arange(np.min(y),np.max(y)+1,delta_x)
    x_grid, y_grid = np.meshgrid(x_grid,y_grid)

    lons, lats = map(x_grid, y_grid, inverse=True)
    # Loop over all of the times
    z_grid = np.ones((ref.shape[1],x_grid.shape[0],x_grid.shape[1]))*np.nan
    cref_grid = np.ones((ref.shape[1],x_grid.shape[0],x_grid.shape[1]))*np.nan
    for k in range(ref.shape[0]):

        # Saving this here to package everything up to get sent to the dart writer
        radar = {}

        ref_grid = np.ones((ref.shape[1],x_grid.shape[0],x_grid.shape[1]))*np.nan
        vr_grid = np.ones((ref.shape[1],x_grid.shape[0],x_grid.shape[1]))*np.nan

        # Do the Cressman analysis for the data on this new grid
        cr = delta_x*np.sqrt(2)
        for j in range(len(z)):

            # We are masking out nans and super low ref values before the superob-ing
            foo = np.where(((~np.isnan(ref[k,j])) & (ref[k,j] >= 5)))
            if len(foo[0]) > 0:
                ref_grid[j] = inverse_distance_to_grid(x[foo].ravel(),y[foo].ravel(),ref[k,j][foo].ravel(),x_grid,y_grid,cr)
            else:
                ref_grid[j,:,:] = np.nan

            foo = np.where(~np.isnan(vr[k,j]))
            if len(foo[0]) > 0:
                vr_grid[j] = inverse_distance_to_grid(x[foo].ravel(),y[foo].ravel(),vr[k,j][foo].ravel(),x_grid,y_grid,cr)
            else:
                vr_grid[j,:,:] = np.nan

            if k == 0:
                z_grid[j] = inverse_distance_to_grid(x.ravel(),y.ravel(),z[j].ravel(),x_grid,y_grid,cr)
                # I need a better way to do this, but this array is only making sure that
                # the zeros don't extend outside of the radar range
                if write_ref:
                    cref_grid[j] = inverse_distance_to_grid(x.ravel(),y.ravel(),ref[k,j].ravel(),x_grid,y_grid,cr)
            # If you want to assimilate reflectivity we need to do the zeros
        if write_ref:

            ref0_grid = np.ones((len(dbz0_levels),x_grid.shape[0],x_grid.shape[1]))

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

            radar['ref_grid'] = ref_grid
            radar['ref0_grid'] = ref0_grid
        
        #Nan out vr where ref_grid is less than the threshold
        foo = np.where((np.isnan(ref_grid)) | (ref_grid < ref_min))
        vr_grid[foo] = np.nan

        radar['vr_grid'] = vr_grid

         # Now write the DART file
        write_dart_obs_seq(radar,x_grid,y_grid,lons,lats,z_grid,time[k],output_dir,
                           site_id,site_lon,site_lat,site_alt,write_ref,dbz0_levels)
        


            



