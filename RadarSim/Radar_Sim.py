import os
import sys
import numpy as np
from netCDF4 import Dataset
from argparse import ArgumentParser
import glob
import pyproj
from datetime import datetime, timedelta

from collections import defaultdict
from numba import njit, prange

@njit(parallel=True)
def interp_columnwise_binary(R, Zorig, Znew):
    ''' from chatgpt. R = model field, Zorig = original model z coordinates, Znew = radar beam heights (scan_z)'''
    nz_new, ny, nx = Znew.shape
    nz = R.shape[0]
    out = np.empty((nz_new, ny, nx), dtype=R.dtype)

    for j in prange(ny):
        for i in range(nx):

            zc = Zorig[:, j, i]
            rc = R[:, j, i]

            for k in range(nz_new):
                ztgt = Znew[k, j, i]

                idx = np.searchsorted(zc, ztgt) - 1
                if idx < 0: idx = 0
                if idx >= nz-1: idx = nz-2

                z0, z1 = zc[idx], zc[idx+1]
                r0, r1 = rc[idx], rc[idx+1]

                w = (ztgt - z0) / (z1 - z0)
                out[k, j, i] = r0 + w*(r1 - r0)

    return out


def findnearest(array, value):
    array = np.asarray(array)
    return (np.abs(array-value)).argmin()

def read_namelist(filename):
    
    # This large structure will hold all of the namelist option. We are
    # predefining it so that default values will be used if they are not listed
    # in the namelist. The exception to this is if the user does not specify 
    # the model. In this scenerio the success flag will terminate the program.
    
    
    namelist = ({'success':0,
                 'model':0,              # Type of model used for the simulation. 1-WRF 5-CM1
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
                 'end_year':0,                    # Ignored is use_calendar is 0
                 'end_month':0,                   # Ignored if use_calendar is 0
                 'end_day':0,                     # Ignored if use_calendar is 0
                 'end_hour':0,                    # Ignored if use_calendar is 0
                 'end_min':0,                     # Ignored if use_calendar is 0
                 'end_sec':0,                     # Ignored if use_calendar is 0
                 'start_time':0.0,                # Start time of the profiler simulation (Ignored if used calendar is 1)
                 'end_time':86400.0,              # End time of the profiler simulation (Ignored if used calendar is 1)
                 'station_file':'None',           # File that contains the locations for the radar scans
                 'append':1,                       # = 0 - don't append to file, 1 - append to file
                 'clobber':0,                     # 0 - Don't clobber file, 1 - clobber it
                 'elevations':'0.5,0.9,1.3,1.8,2.4,3.1,4.0,5.1,6.4,8.0,10.0,12.5,15.6,19.5'
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

def theta2t(theta, w, p, p0 = 1000.):
    th = theta/ ( 1 + 0.61 * (w/1000.))
    t = th / ( (p0/p)**0.286 ) - 273.16
    
    return t

def t2density(t, p):
    
    Rv = 287.05              # Gas constant for dry air [J / (kg k)]
    dens = (p * 100.) / (Rv * (t+273.15) )
    
    return dens

def dbztowt(ref,rho):

    refl = 10.0**(0.1*ref)
    dbztowt = -2.6 * refl**0.107 * (1.2/rho)**0.4

    return dbztowt

def sfc_rng_to_beam_height(sfc_range,el):

    sfc_rng = np.copy(sfc_range)

    eradius=6371000.
    frthrde=(4.*eradius/3.)

    rngdb = sfc_rng/frthrde

    zz = (frthrde/(np.cos(rngdb[None,:,:])-(np.sin(rngdb[None,:,:])*np.tan(np.deg2rad(el[:,None,None])))))-frthrde

    return zz

def sfc_rng_to_range(sfc_range,el,z):

    eradius=6371000.
    frthrde=(4.*eradius/3.)

    rngdb = sfc_range/frthrde
    hgtdb = z +frthrde

    r = hgtdb*np.sin(rngdb[None,:,:])/np.cos(np.deg2rad(el[:,None,None]))

    return r

def read_wrf(stations,model_dir,time,prefix,latlon):

    # account for potential formatting differences
    if os.path.isfile(model_dir + '/' + prefix + time.strftime('%Y-%m-%d_%H_%M_%S')):
        file = model_dir + '/' + prefix + time.strftime('%Y-%m-%d_%H_%M_%S')
    else:
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

        truelat1 = fid.TRUELAT1
        truelat2 = fid.TRUELAT2
        lat_0 = fid.MOAD_CEN_LAT
        lon_0 = fid.STAND_LON

        # Now transform the data
        e, n = transformer.transform(fid.CEN_LON, fid.CEN_LAT)
        dx,dy = fid.DX, fid.DY
        nx, ny = fid.dimensions['west_east'].size, fid.dimensions['south_north'].size
        x0 = -(nx-1) / 2. * dx + e
        y0 = -(ny-1) / 2. * dy + n
        x_grid = np.arange(nx) * dx + x0
        y_grid = np.arange(ny) * dy + y0
        xx, yy = np.meshgrid(np.arange(nx) * dx + x0, np.arange(ny) * dy + y0)
        station_x_proj, station_y_proj = transformer.transform(np.atleast_2d(stations)[:,1], np.atleast_2d(stations)[:,0])
        station_alt = np.copy(np.atleast_2d(stations[:,2]))
        
    else:
        xx, yy = np.meshgrid(np.arange(fid.dimensions['west_east'].size) * fid.DX, np.arange(fid.dimensions['south_north'].size) * fid.DY)
        station_x_proj = np.copy(np.atleast_2d(stations)[:,0])
        station_y_proj = np.copy(np.atleast_2d(stations)[:,1])
        station_alt = np.copy(np.atleast_2d(stations)[:,2])

        truelat1 = 0
        truelat2 = 0
        lat_0 = 0
        lon_0 = 0
    
    ground = fid.variables['HGT'][0,:,:]
    ref = fid.variables['REFL_10CM'][0,:,:,:]
    zz = (fid.variables['PH'][0,:,:,:] + fid.variables['PHB'][0,:,:,:])/9.81
    zz = (zz[1:]+zz[:-1])/2

    sinalpha = fid.variables['SINALPHA'][0,:,:]
    cosalpha = fid.variables['COSALPHA'][0,:,:]
    p = (fid.variables['P'][0,:,:,:] + fid.variables['PB'][0,:,:,:])/100
    t = fid.variables['T'][0] + 300
    t = theta2t(t, np.zeros(p.shape), p)
    rho = t2density(t,p)

    w_t = dbztowt(ref,rho)

    u = (fid.variables['U'][0,:,:,1:] + fid.variables['U'][0,:,:,:-1])/2
    v = (fid.variables['V'][0,:,1:,:] + fid.variables['V'][0,:,:-1,:])/2
    w = (fid.variables['W'][0,1:,:,:] + fid.variables['W'][0,:-1,:,:])/2

    u_fixed = u*cosalpha[None,:,:] - v*sinalpha[None,:,:]
    v_fixed = v*cosalpha[None,:,:] + u*sinalpha[None,:,:]
    w_fixed = w+w_t

    lat = fid.variables['XLAT'][0]
    lon = fid.variables['XLONG'][0]
    fid.close()

    return {'station_x':station_x_proj,'station_y':station_y_proj, 'station_alt':station_alt,
            'xx':xx,'yy':yy,'zz':zz, 'x_grid':x_grid, 'y_grid':y_grid, 'lat':lat,'lon':lon, 'u':u_fixed, 'v':v_fixed, 'w':w_fixed,
            'ref':ref,'ground':ground,'truelat1':truelat1, 'truelat2':truelat2, 'lat0':lat_0, 'lon0':lon_0, 'proj':'lcc',}

def read_cm1(stations,model_dir,time,frequency,prefix):

    if int(time/frequency)+1 < 10:
        file =  model_dir + '/' + prefix + '_00000' + str(int(time/frequency)+1) +'.nc'
    elif int(time/frequency)+1 < 100:
        file = model_dir + '/' + prefix + '_0000' + str(int(time/frequency)+1)+'.nc'
    elif int(time/frequency)+1 < 1000:
        file = model_dir + '/' + prefix + '_000' + str(int(time/frequency)+1)+'.nc'
    elif int(time/frequency)+1 < 10000:
        file = model_dir + '/' + prefix + '_00' + str(int(time/frequency)+1)+'.nc'
    elif int(time/frequency)+1 < 100000:
        file = model_dir +'/' + prefix + '_0' + str(int(time/frequency)+1)+'.nc'
    else:
        file = model_dir + '/' + prefix + '_' + str(int(time/frequency)+1)+'.nc'

    f = Dataset(file,'r')

    z = f.variables['zh'][:]*1000
    x_grid = f.variables['xh'][:]*1000
    y_grid = f.variables['yh'][:]*1000

    xx, yy = np.meshgrid(x_grid,y_grid)
    zz = z[:,None,None] * np.ones(xx.shape)[None,:,:]

    ground = np.zeros(xx.shape)

    station_x_proj = np.copy(np.atleast_2d(stations)[:,0])
    station_y_proj = np.copy(np.atleast_2d(stations)[:,1])
    station_alt = np.copy(np.atleast_2d(stations)[:,2])

    truelat1 = 0
    truelat2 = 0
    lat_0 = 0
    lon_0 = 0

    rho = f.variables['rho'][0]
    ref = f.variables['dbz'][0]

    w_t = dbztowt(ref,rho)

    u = f.variables['uinterp'][0]
    v = f.variables['vinterp'][0]
    w = f.variables['winterp'][0] + w_t

    lat = np.zeros(xx.shape)
    lon = np.zeros(xx.shape)

    f.close()

    return {'station_x':station_x_proj,'station_y':station_y_proj, 'station_alt':station_alt,
            'xx':xx,'yy':yy,'zz':zz, 'x_grid':x_grid, 'y_grid':y_grid, 'lat':lat,'lon':lon, 'u':u, 'v':v, 'w':w,
            'ref':ref,'ground':ground,'truelat1':truelat1, 'truelat2':truelat2, 'lat0':lat_0, 'lon0':lon_0, 'proj':'None',}

def create_and_write_radar_obs(stations,station_id,model_dir,time,frequency,prefix,elevations,namelist,latlon, index,start_time, end_time,max_range=300):


    if isinstance(time,datetime):
        print('Starting generation of obs for time: ' + time.strftime('%Y-%m-%d_%H:%M:%S'))
    else:
        print('Starting generation of obs for time: ' + str(time))

    if namelist['model'] == 1:
        model_data = read_wrf(stations,model_dir,time,prefix,latlon)
    
    elif namelist['model'] == 5:
        model_data = read_cm1(stations,model_dir,time,frequency,prefix)

    a_e = 4*6371 *1000/3 
    beam_height = np.sqrt(max_range**2 + (a_e)**2 + 2*(a_e)*max_range*np.sin(np.deg2rad(elevations[0])))-a_e
    sfc_rng = a_e*np.arcsin(max_range*np.cos(np.deg2rad(elevations[0]))/(a_e+beam_height))*1000
    
    
    ##################
    # enclose the approximate domain to do a quick check for radar location - saves ~1-9 s per excluded radar!
    buffer = max_range + 100
    buf_y1, buf_y2 = np.min(model_data['yy'])-buffer, np.max(model_data['yy'])+buffer
    buf_x1, buf_x2 = np.min(model_data['xx'])-buffer, np.max(model_data['xx'])+buffer
    
    ##################
    
    
    for k in range(np.atleast_2d(stations).shape[0]):
        
        print('Generating data for ' + np.atleast_1d(station_id)[k])
        
        # do a very rough and dirty check for radar locations. It's faster than a more formal check of the surface range 
        rx  = model_data['station_x'][k]
        ry  = model_data['station_y'][k]
        
        if (rx > buf_x1) & (rx < buf_x2) & (ry > buf_y1) & (ry < buf_y2):
           
            # Calculate the surface range from the radar (xrel = 'x coordinates relative to radar')
            xrel = model_data['xx']-model_data['station_x'][k]
            yrel = model_data['yy']-model_data['station_y'][k]
            zrel = model_data['zz']-model_data['station_alt'].flatten()[k]
            grid_sfc_rng = np.sqrt(xrel**2 + yrel**2)
        
            # helps with indexing later
            xrel_1d = model_data['x_grid']-model_data['station_x'][k]
            yrel_1d = model_data['y_grid']-model_data['station_y'][k]
            
        
            az_temp = np.rad2deg(np.arctan2(xrel,yrel))
            az_temp[az_temp < 0.0] += 360.0
        
            # # We only want to work with the points that are inside max range
            foo = np.where(grid_sfc_rng <= sfc_rng)
        
            if len(foo[0]) == 0:
                print('    radar not in model domain, skipping...')
                continue
            
            range_mask = (grid_sfc_rng > sfc_rng)
            
            # radar relative coordinates, masked where max range is exceeded
            masked_xrel = np.where(range_mask, np.nan, xrel)
            masked_yrel = np.where(range_mask, np.nan, yrel)
        
            # find indices to subset data around radar 
            # note that +1 was added to the max index already
            y1, y2 = findnearest(np.nanmin(masked_yrel), yrel_1d), findnearest(np.nanmax(masked_yrel), yrel_1d)+1
            x1, x2 = findnearest(np.nanmin(masked_xrel), xrel_1d), findnearest(np.nanmax(masked_xrel), xrel_1d)+1

            # create subset 
            ref_sub = model_data['ref'][:, y1:y2, x1:x2]
            u_sub   = model_data['u'][:, y1:y2, x1:x2]
            v_sub   = model_data['v'][:, y1:y2, x1:x2]
            w_sub   = model_data['w'][:, y1:y2, x1:x2]
        
            x_sub   = xrel[y1:y2, x1:x2]
            y_sub   = yrel[y1:y2, x1:x2]
            z_sub   = zrel[:,y1:y2, x1:x2]
        
            lat_sub = model_data['lat'][y1:y2, x1:x2]
            lon_sub = model_data['lon'][y1:y2, x1:x2]
            ground_temp = model_data['ground'][y1:y2, x1:x2]
        
            sfc_rng_temp = grid_sfc_rng[y1:y2, x1:x2]
        
            az_sub = np.rad2deg(np.arctan2(x_sub, y_sub))
            az_sub[az_sub < 0.0] += 360.0
        
        
            # Now for each grid box, calculate the height of the beam for each elevation
            # First find the heights for each elevation
            scan_z = sfc_rng_to_beam_height(sfc_rng_temp,elevations)
        
            # Now find the range for each elevation
            scan_r = sfc_rng_to_range(sfc_rng_temp,elevations,scan_z)
        
        
            # using jit and numba, perform the interpolation
            # it loops over all x and y grid points, and does an interpolation in each vertical column
            # it can't handle masked arrays, so we need to fill the invalid data with nans
            myref = interp_columnwise_binary(ref_sub.filled(np.nan), z_sub.filled(np.nan), scan_z)
            myu   = interp_columnwise_binary(u_sub.filled(np.nan),   z_sub.filled(np.nan), scan_z)
            myv   = interp_columnwise_binary(v_sub.filled(np.nan),   z_sub.filled(np.nan), scan_z)
            myw   = interp_columnwise_binary(w_sub.filled(np.nan),   z_sub.filled(np.nan), scan_z)
            
            myvel = (myu*x_sub + myv*y_sub + myw*scan_z)/ np.sqrt(x_sub**2 + y_sub**2 + scan_z**2)
        
        
            # now apply final masks 
            ground_mask = (np.stack([ground_temp]*len(elevations)) >= (scan_z+model_data['station_alt'].flatten()[k]))
            range_mask  = (scan_r > max_range*1000)
            ref_mask    = myref < 5
        
            myref[ground_mask | range_mask] = np.nan
            myvel[ground_mask | range_mask | ref_mask] = np.nan

                 
            radar = {'station_id': np.atleast_1d(station_id)[k], 'radar_lat':np.atleast_2d(stations)[k,0], 'radar_lon':np.atleast_2d(stations)[k,1],'radar_alt':np.atleast_2d(stations)[k,2],
                     'ref':[myref.copy()], 'vel':[myvel.copy()],
                     'lat':[lat_sub], 'lon':[lon_sub], 'x':[x_sub], 'y':[y_sub],'z':[scan_z],'el':elevations,'az':[az_sub],
                     'truelat1':np.copy(model_data['truelat1']), 'truelat2':np.copy(model_data['truelat2']),
                     'lat0':np.copy(model_data['lat0']), 'lon0':np.copy(model_data['lon0']), 'proj':'lcc'}



            write_to_file(radar,output_dir, namelist, time, index, start_time, end_time)

            del radar
            
    return 1#, radar


def write_to_file(radar,output_dir, namelist, model_time, snum, start_time, end_time):
    
    
    if namelist['use_calendar'] == 1:
        start_str = start_time.strftime('%Y%m%d_%H%M%S')
        end_str = end_time.strftime('%Y%m%d_%H%M%S')
    else:
        start_str = str(int(start_time))
        end_str = str(int(end_time))
    
    # We need to do this for each radar site
    for i in range(len(np.atleast_1d(radar['station_id']))):

        outfile_path = output_dir + '/' + namelist['outfile_root'] + '_' + np.atleast_1d(radar['station_id'])[i] + '_' + start_str + '_' + end_str + '.nc'

        # We don't want to append the file needs to be created the first time  
        if ((namelist['append'] == 0) & (not os.path.exists(outfile_path))):
            fid = Dataset(outfile_path,'w')
        
            tdim = fid.createDimension('time',None)
            xdim = fid.createDimension('x',radar['x'][i].shape[1])
            ydim = fid.createDimension('y',radar['y'][i].shape[0])
            edim = fid.createDimension('el',len(radar['el']))
        
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
        
            lat = fid.createVariable('latitude','f4',('y','x',))
            lat.long_name = 'Latitude'
            lat.units = 'degree'
        
            lon = fid.createVariable('longitude','f4',('y','x',))
            lon.long_name = 'Longitude'
            lon.units = 'degree'

            yy = fid.createVariable('y','f4',('y','x',))
            yy.long_name = 'Northing'
            yy.units = 'm'
        
            xx = fid.createVariable('x','f4',('y','x',))
            xx.long_name = 'Easting'
            xx.units = 'm'
        
            alt = fid.createVariable('altitude','f4',('el','y','x',))
            alt.long_name = 'Altitude above radar'
            alt.units = 'm'

            elev = fid.createVariable('elevation','f4',('el',))
            elev.long_name = 'Elevation'
            elev.units = 'deg'

            azim = fid.createVariable('azimuth','f4',('y','x',))
            azim.long_name = 'Azimuth'
            azim.units = 'deg'

            ref = fid.createVariable('reflectivity','f4',('time','el','y','x',))
            ref.long_name = 'Reflectivity'
            ref.units = 'dBZ'

            vel = fid.createVariable('radial_velocity','f4',('time','el','y','x',))
            vel.long_name = 'Radial Velocity'
            vel.units = 'm/s'
        
            if namelist['model'] == 1:
                fid.model = 'WRF'
        
            if namelist['use_calendar'] == 0:
            
                fid.start_time = start_time
                fid.end_time = end_time
        
            else:
                fid.start_time = start_time.strftime('%Y%m%d_%H%M%S')
                fid.end_time = end_time.strftime('%Y%m%d_%H%M%S')
        
            lat[:] = radar['lat'][i]
            lon[:] = radar['lon'][i]
            alt[:] = radar['z'][i]
            xx[:] = radar['x'][i]
            yy[:] = radar['y'][i]

            fid.projection = radar['proj']
            fid.truelat1 = radar['truelat1']
            fid.truelat2 = radar['truelat2']
            fid.lat0 = radar['lat0']
            fid.lon0 = radar['lon0']
            fid.site_id = np.atleast_1d(radar['station_id'])[i]
            fid.site_lon = np.atleast_1d(radar['radar_lon'])[i]
            fid.site_lat = np.atleast_1d(radar['radar_lat'])[i]
            fid.site_alt = np.atleast_1d(radar['radar_alt'])[i]

        # Append the data to the file
    
        fid = Dataset(outfile_path,'a')
    
        bt = fid.variables['base_time']
        time = fid.variables['time_offset']
        alt= fid.variables['altitude']
        xx = fid.variables['x']
        yy = fid.variables['y']
        ref = fid.variables['reflectivity']
        vel = fid.variables['radial_velocity']
        elev = fid.variables['elevation']
        azim = fid.variables['azimuth']
    
        if namelist['use_calendar'] == 0:
            time[snum] = model_time-start_time
        else:
            time[snum] = (model_time - datetime(1970,1,1)).total_seconds() - bt[0]
    
        ref[snum,:,:,:] = radar['ref'][i][:,:,:]
        vel[snum,:,:,:] = radar['vel'][i][:,:,:]
    
        fid.sim_number = snum
    
        fid.close()
##############################################################################          
#Create parser for command line arguments
parser = ArgumentParser()

parser.add_argument("namelist_file", help="Name of the namelist file (string)")
parser.add_argument("--output_dir", help="Path to output directory")

args = parser.parse_args()

namelist_file = args.namelist_file
output_dir = args.output_dir

if output_dir is None:
    output_dir = os.getcwd() + '/'

print("-----------------------------------------------------------------------")
print("Starting Radar_Sim")
print("Output directory set to " + output_dir)

# Read the namelist file
namelist = read_namelist(namelist_file)
namelist['output_dir'] = output_dir
if namelist['success'] != 1:
    print('>>> Radar_Sim FAILED and ABORTED <<<')
    print("-----------------------------------------------------------------------")
    sys.exit()

# Read in the radar scan file
print('Reading in radar station file')
try:
    stations = np.genfromtxt(namelist['station_file'],autostrip=True,usecols=[1,2,3])
    station_id = np.genfromtxt(namelist['station_file'],dtype=str,autostrip=True,usecols=[0])
except:
    print('ERROR: Something went wrong reading radar stations')
    print('>>> Radar_Sim FAILED and ABORTED <<<')
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


# Get the levels for the MRMS_Ref Grid
elevations = namelist['elevations'].split(',')
elevations = np.array(elevations).astype(float)


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
                print(('>>> Radar_Sim FAILED and ABORTED'))
                print('--------------------------------------------------------------------')
                print(' ')
                sys.exit()
        
        else:
           if start_time.strftime('%Y%m%d_%H%M%S') != out.start_time:
               print('Append mode was selected, but the start time is not the same.')
               print(('>>> Radar_Sim FAILED and ABORTED'))
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
    
    success = create_and_write_radar_obs(stations,station_id,namelist['model_dir'],model_time[index],namelist['model_frequency'],
                 namelist['model_prefix'], elevations, namelist, namelist['coordinate_type'], index, start_time, end_time)
    
    # if success != 1:
    #     print('Something went wrong collecting the radar obs for ', model_time[index])
    #     print('Skipping model time')
    #     continue
    
    # write_to_file(radar,output_dir, namelist, model_time[index], index, start_time, end_time)
