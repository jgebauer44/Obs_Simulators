import numpy as np
import pandas as pd
import os

df = pd.read_csv('/data/geoinfo-Jan8.csv')

meso_lats = np.array(df['nlat'])
meso_lons = np.array(df['elon'])
meso_elev = np.array(df['elev'])
meso_ids = np.array(df['stid'])
meso_names = np.array(df['name'])
meso_datd = np.array(df['datd'])


def namelist_write(site, site_id, lat, lon):
	with open(f'namelist_site.input', 'w') as f:
		f.write('model                = 1         # 5-CM1\n')
		f.write('model_frequency      = 900       # Frequency of model output\n')
		f.write('model_dir            = /data/NatureRun_20230419/              # Director with the model data\n')
		f.write('model_prefix         = wrfwof_d02_    # Prefix for the model data files\n')
		f.write('outfile_root         = MWR_{}_{}\n'.format(site_id, site))
		f.write('append               = 0         # Flag for whether we are appending to existing file\n')
		f.write('clobber              = 0         # Flag for whether we are clobbering existing file\n')
		f.write('#\n')
		f.write('profiler_x           = {}    # x-position for profiler in meters\n'.format(lon))
		f.write('profiler_y           = {}    # y-position for profiler in meters\n'.format(lat))
		f.write('use_calendar         = 1\n')
		f.write('start_year           = 2023\n')
		f.write('start_month          = 4\n')
		f.write('start_day            = 19\n')
		f.write('start_hour           = 21\n')
		f.write('start_min            = 0\n')
		f.write('start_sec            = 0\n')
		f.write('end_year             = 2023\n')
		f.write('end_month            = 4\n')
		f.write('end_day              = 19\n')
		f.write('end_hour             = 22\n')
		f.write('end_min              = 30\n')
		f.write('end_sec              = 0\n')
		f.write('start_time           = 600       # Start time of the simulation\n')
		f.write('end_time             = 7800      # End time of the simulation\n')
		f.write('profiler_type        = 2         # 1-IRS, 2-MWR\n')
		f.write('aeri_noise_file      = /home/tropoe/vip/src/tropoe/AERI_avg_noise_20220301_20220430.nc\n')
		f.write('coordinate_type      = 1\n')
		f.write('#\n')
		f.write('mp_scheme            = 1         # 1-NSSL, 2-Morrison\n')
		f.write('ndcnst               = 250       # Number of cloud drops for microphysic scheme\n')
		f.write('#\n')
		f.write('spectral_bands       = 612-618,624-660,674-713,713-722,538-588,860.1-864.0,872.2-877.5,898.2-905.4\n')

		f.write('mwr_freq             = 22.24,23.04,23.84,25.44,26.24,27.84,31.4,51.26,52.28,53.86,54.94,56.66,57.3,58\n')
		f.write('mwr_noise            = 0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.8,0.8,0.3,0.3,0.3,0.25,0.25\n')
		f.write('mwr_elev             = 9.8,17.9,23.9,29.9,42,90,138,149.9,155.9,161.9,169.8\n')
		f.close()


i = 0
while i < len(meso_lats):
	if meso_datd[i] >= 20990000:
        	print(meso_names[i])
        	site_name = meso_names[i].replace(" ","")
        	namelist_write(site_name, meso_ids[i], meso_lats[i], meso_lons[i])
        	os.system('python RadianceSim.py 20230419 namelist_site.input --output_dir /data/RadianceSim_Output/MWR')
        	i+=1
	else:
        	i+=1

