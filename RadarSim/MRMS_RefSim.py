import os
import sys
import numpy as np
from netCDF4 import Dataset
from argparse import ArgumentParser
import glob
import pyproj
from datetime import datetime, timedelta

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
                 'station_file':'None',           # File that contains the locations for the radar scans
                 'append':1,                       # = 0 - don't append to file, 1 - append to file
                 'clobber':0,                     # 0 - Don't clobber file, 1 - clobber it
                 'levels':'500,1000,1500,2000,2500,3000,3500,4000,5000,6000,7000,8000,9000,10000'                     # Levels for MRMS style reflectivity
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

def create_mrms_ref(stations,model_dir,time,prefix,levels,mrms_mask,latlon=1):

    print('Starting generation of obs for time: ' + time.strftime('%Y-%m-%d_%H:%M:%S'))
    file = model_dir + '/' + prefix + time.strftime('%Y-%m-%d_%H:%M:%S')

    try:
        fid = Dataset(file,'r')
    except:
        print('Could not open ' + file)
        return -999., 0, mrms_mask
    
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
        station_x_proj, station_y_proj = transformer.transform(stations[:,1], stations[:,0])
        station_alt = np.copy(stations[:,2])
        
    else:
        xx, yy = np.meshgrid(np.arange(fid.dimensions['west_east'].size) * fid.DX, np.arange(fid.dimensions['south_north'].size) * fid.DY)
        station_x_proj = np.copy(stations[:,0])
        station_y_proj = np.copy(stations[:,1])
        station_alt = np.copy(stations[:,3])

        truelat1 = 0
        truelat2 = 0
        lat_0 = 0
        lon_0 = 0
    
    ground = fid.variables['HGT'][0,:,:]
    ref = fid.variables['REFL_10CM'][0,:,:,:]
    zz = (fid.variables['PH'][0,:,:,:] + fid.variables['PHB'][0,:,:,:])/9.81
    zz = (zz[1:]+zz[:-1])/2

    lat = fid.variables['XLAT'][0]
    lon = fid.variables['XLONG'][0]
    fid.close()

    # We only have to do this once
    if mrms_mask is None:
        mrms_mask = get_mrms_mask(station_x_proj, station_y_proj, station_alt, levels, ground,xx,yy)
    
    # Initialize the mrms array
    mrms_ref = np.ones((len(levels),ref.shape[1],ref.shape[2]))

    # Looooong loop, I am going to have to come back to this
    # to find a faster way to do this
    for i in range(ref.shape[1]):
        for j in range(ref.shape[2]):
            mrms_ref[:,i,j] = np.interp(levels,zz[:,i,j],ref[:,i,j],left = np.nan,right=np.nan)
    
    # Apply the mask to the data to remove where we don't want it
    mrms_ref[mrms_mask] = np.nan

    ref_dict = {'lat':lat,'lon':lon,'x':xx,'y':yy,'z':levels,
                'mrms_ref':mrms_ref, 'truelat1':truelat1, 'truelat2':truelat2,
                'lat0':lat_0, 'lon0':lon_0, 'proj':'lcc'}
    return 1, ref_dict, mrms_mask

def beam_elv(sfc_range,z):

    sfc_rng = np.copy(sfc_range)
    zz = np.copy(z)

    if len(sfc_rng.shape) == 2:
        sfc_rng = np.stack([sfc_rng]*19,axis=0)
    
    if len(zz.shape) == 1:
        zz = np.tile(zz,[sfc_rng.shape[1],sfc_rng.shape[2],1]).T

    eradius=6371000.
    frthrde=(4.*eradius/3.)

    elvrad = np.ones(sfc_rng.shape)*np.nan
    foo = np.where(sfc_rng > 0.0)

    hgtdb = frthrde + zz
    rngdb = sfc_rng/frthrde

    elvrad[foo] = np.arctan((hgtdb[foo]*np.cos(rngdb[foo]) - frthrde)/(hgtdb[foo] * np.sin(rngdb[foo])))

    return np.rad2deg(elvrad)

def get_mrms_mask(x,y,alt,levels,ground,xx,yy,min_el=0.5,max_el=19.5,max_range=300):

    a_e = 4*6371 *1000/3

    # Initialize the array for the mask
    mask_counts = np.zeros((len(levels),xx.shape[0],xx.shape[1]))
    
    # Mask out levels that are below the ground
    for i in range(len(levels)):
        foo = np.where(ground >= levels[i])
        mask_counts[i][foo] = np.nan

    # Make a range array for calculation
    r = np.arange(0,max_range,1)
    beam_height = np.sqrt(r**2 + (a_e)**2 + 2*(a_e)*r*np.sin(np.deg2rad(min_el)))-a_e
    sfc_rng = a_e*np.arcsin(r*np.cos(min_el)/(a_e+beam_height))*1000

    for i in range(len(x)):

        # Calculate the surface range from the radar
        xx_temp = xx-x[i]
        yy_temp = yy-y[i]
        zz_temp = levels-alt[i]

        grid_sfc_rng = np.stack([np.sqrt(xx_temp**2 + yy_temp**2)]*len(levels),axis=0)

        # Find the elevation for the points around the radar 
        el = beam_elv(grid_sfc_rng,zz_temp)
        
        foo = np.where(((el >= min_el) & (el <= max_el) & (grid_sfc_rng <= sfc_rng[-1])))
    
        mask_counts[foo] += 1

    # Now find where there is any coverage
    mask = np.where(mask_counts > 0,False,True)

    return mask

def write_to_file(radar,output_dir, namelist, model_time, snum, start_time, end_time):
    
    
    if namelist['use_calendar'] == 1:
        start_str = start_time.strftime('%Y%m%d_%H%M%S')
        end_str = end_time.strftime('%Y%m%d_%H%M%S')
    else:
        start_str = str(start_time)
        end_str = str(end_time)
    
    outfile_path = output_dir + '/' + namelist['outfile_root'] + '_' + start_str + '_' + end_str + '.nc'
    
    # We don't want to append the file needs to be created the first time  
    if ((namelist['append'] == 0) & (not os.path.exists(output_dir + namelist['outfile_root'] + '_' + start_str + '_' + end_str + '.nc'))):
        fid = Dataset(output_dir + '/' + namelist['outfile_root'] + '_' + start_str + '_' + end_str + '.nc','w')
        
        tdim = fid.createDimension('time',None)
        xdim = fid.createDimension('x',radar['x'].shape[1])
        ydim = fid.createDimension('y',radar['y'].shape[0])
        hdim = fid.createDimension('height',len(radar['z']))
        
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
        
        lat = fid.createVariable('latitude','f4',('y','x'))
        lat.long_name = 'Latitude'
        lat.units = 'degree'
        
        lon = fid.createVariable('longitude','f4',('y','x',))
        lon.long_name = 'Longitude'
        lon.units = 'degree'

        yy = fid.createVariable('y','f4',('y','x'))
        yy.long_name = 'Northing'
        yy.units = 'm'
        
        xx = fid.createVariable('x','f4',('y','x',))
        xx.long_name = 'Easting'
        xx.units = 'm'
        
        alt = fid.createVariable('altitude','f4',('height',))
        alt.long_name = 'Altitude above sea level'
        alt.units = 'm'
        
        ref = fid.createVariable('reflectivity','f4',('time','height','y','x'))
        ref.long_name = 'MRMS reflectivity'
        ref.units = 'dBZ'
        
        if namelist['model'] == 1:
            fid.model = 'WRF'
        
        if namelist['use_calendar'] == 0:
            
            fid.start_time = start_time
            fid.end_time = end_time
        
        else:
            fid.start_time = start_time.strftime('%Y%m%d_%H%M%S')
            fid.end_time = end_time.strftime('%Y%m%d_%H%M%S')
        
        lat[:] = radar['lat']
        lon[:] = radar['lon']
        alt[:] = radar['z']
        xx[:] = radar['x']
        yy[:] = radar['y']

        fid.projection = radar['proj']
        fid.truelat1 = radar['truelat1']
        fid.truelat2 = radar['truelat2']
        fid.lat0 = radar['lat0']
        fid.lon0 = radar['lon0']
    # Append the data to the file
    
    fid = Dataset(outfile_path,'a')
    
    bt = fid.variables['base_time']
    time = fid.variables['time_offset']
    alt= fid.variables['altitude']
    xx = fid.variables['x']
    yy = fid.variables['y']
    ref = fid.variables['reflectivity']
    
    
    if namelist['use_calendar'] == 0:
        time[snum] = model_time-start_time
    else:
        time[snum] = (model_time - datetime(1970,1,1)).total_seconds() - bt[0]
    
    ref[snum,:,:,:] = radar['mrms_ref'][:,:,:]
    
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
print("Starting MRMS_RefSim")
print("Output directory set to " + output_dir)

# Read the namelist file
namelist = read_namelist(namelist_file)
namelist['output_dir'] = output_dir
if namelist['success'] != 1:
    print('>>> MRMS_RefSim FAILED and ABORTED <<<')
    print("-----------------------------------------------------------------------")
    sys.exit()

# Read in the radar scan file
print('Reading in radar station file')
try:
    stations = np.genfromtxt(namelist['station_file'],autostrip=True,usecols=[1,2,3])
except:
    print('ERROR: Something went wrong reading radar stations')
    print('>>> MRMS_RefSim FAILED and ABORTED <<<')
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
levels = namelist['levels'].split(',')
levels = np.array(levels).astype(float)


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
                print(('>>> MRMS_RefSim FAILED and ABORTED'))
                print('--------------------------------------------------------------------')
                print(' ')
                sys.exit()
        
        else:
           if start_time.strftime('%Y%m%d_%H%M%S') != out.start_time:
               print('Append mode was selected, but the start time is not the same.')
               print(('>>> MRMS_RefSim FAILED and ABORTED'))
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

mrms_mask = None
for index in range(len(snum)):
    
    print('Performing simulation number ' + str(snum[index]) +  ' at time ' + str(model_time[index]))
    
    if index <= last_snum:
        '        ...but was already processed. Continuing.'
        continue
    
    if namelist['model'] == 1:
        success, radar, mrms_mask = create_mrms_ref(stations,namelist['model_dir'],model_time[index],
                                     namelist['model_prefix'], levels, mrms_mask, namelist['coordinate_type'])
        
        if success != 1:
            print('Something went wrong collecting the radar obs for ', model_time[index])
            print('Skipping model time')
            continue
    else:
        print('Error: Model type ' + namelist['model_type'] + ' is not a valid option')
        
        
    write_to_file(radar,output_dir, namelist, model_time[index], index, start_time, end_time)
