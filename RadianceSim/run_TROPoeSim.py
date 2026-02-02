import numpy as np
import os
import glob
from pathlib import Path

stations = np.genfromtxt('/home/tropoe/vip/src/tropoe/profilers.txt',autostrip=True)

def namelist_write(lat, lon):
        with open(f'pyVIP_site.txt', 'w') as f:

                f.write('# The input parameter file for the AERIoe retrieval\n')
                f.write('#\n')
                f.write('# Note that lines that start with a "#" are ignored and the default is used\n')
                f.write('# The keys on the left of the "=" are important, and will be matched by the same\n')
                f.write('# strings in the VIP structure.  The values on the right of the = are the values\n')
                f.write('#\n')
                f.write('# This section is for AERI options\n')
                f.write('tag                       = sim   # String for temporary files / directories (default "tag")\n')
                f.write('aeri                      = 1        # 1 - ARM, 2 - dmv2cdf, 3 - dmv2ncdf, 4 - ARM, eng in summary, -1 - No AERI (default 0)\n')
                f.write('aeri_path                 = /data/work/joshua.gebauer/SimObs/8May2024/obs_nc/irs/IRS_{}_{}_20240508_180000.nc     # Path to AERI ch1 radiance files (default "None")\n'.format(lat,lon))
                f.write('aeri_noise_file           = /home/tropoe/vip/src/tropoe/AERI_avg_noise_20220301_20220430.nc        # Path to AERI summary files (default "None")\n')
                f.write('append                    = 0\n')
                f.write('#psfc_max                  = 1030.    # Default maximum surface pressure [mb] (default 1030.)\n')
                f.write('#\n')
                f.write('# This section is for mwr options\n')
                f.write('mwr                        = 0        # 0-none, 1-Tb fields are individual, 2-Tb field is 2-d array (default 0)\n')
                f.write('mwr_path                   = /data/work/joshua.gebauer/SimObs/8May2024/obs_nc/mwr/MWR_{}_{}_20240508_180000.nc     # Path to the MWR data (default None)\n'.format(lat,lon))
                f.write('#\n')
                f.write('# This section is for mwrscan options\n')
                f.write('mwrscan                    = 0        # 0-none, 1-Tb fields are individual, 2-Tb field is 2-d array (default 0)\n')
                f.write('mwrscan_path               = /data/RadianceSim_Output/MWRscan/MWR_NORM_20230419_210000.nc     # Path to the MWRscan data (default None)\n')
                f.write('#\n')
                f.write('# This section controls the cloud base height options\n')
                f.write('cbh_default_ht            = 2.0       # Default CBH height [km AGL], if no CBH data found (default 2.0)\n')
                f.write('#\n')
                f.write('# This section defines the output directory and files\n')
                f.write('output_rootname           = IRS_{}_{}      # String with the rootname of the output file (default None)\n'.format(lat,lon))
                f.write('output_path               = /data/work/joshua.gebauer/SimObs/8May2024/obs_nc/TROPoeSim_Output/irs_recenter/     # Path where the output file will be placed (default None)\n')
                f.write('output_clobber            = 0         # 0-do not clobber preexisting output, 1-clobber them, 2-append to last file of dataset (default 0)\n')
                f.write('#output_file_keep_small    = 0        # 0 - all fields written, 1-keep output file small by not including Sop, Akern... (default 0)\n')
                f.write('#\n')
                f.write('#\n')
                f.write('# This section controls the retrieval options\n')
                f.write('#max_itrations             = 10       # The maximum number of iterations to use (default 10)\n')
                f.write('#first_guess               = 1        # 1- use prior as FG, 2-use lapse rate and 60% RH as FG, 3-use previous sample as FG (default 1)\n')
                f.write('superadiabatic_maxht      = 0.100    # The maximum height a superadiabatic layer at the surface can have [km AGL] (default 0.300)\n')
                f.write('#\n')
                f.write('spectral_bands            = 612-618,624-660,674-713,713-722,538-588,860.1-864.0,872.2-877.5,898.2-905.4 # No spaces here, no default\n')
                f.write('#\n')
                f.write('#\n')
                f.write('globatt_Site = WRF Simulation\n')
                f.write('globatt_Instrument = RadianceSim\n')
                f.write('globatt_Dataset_contact = Joshua Gebauer,CIWRO/NOAA, joshua.gebauer@noaa.gov\n')
                f.write('globatt_Processing_comment = Data were processed using radiosonde prior information\n')
                f.write('globatt_LBLRTM_comment = Data were processed using LBLRTM.v12.2 in the TROPoe Docker container\n')
                f.close()
i = 0
while i < len(stations):
    print(str(i+1) + ':' + str(stations[i,0]) + ' ' + str(stations[i,1]))
    namelist_write(stations[i,0], stations[i,1])
    matching_files = glob.glob('/data/work/joshua.gebauer/SimObs/8May2024/obs_nc/TROPoeSim_Output/irs_recenter/IRS_' + str(stations[i,0]) +'_' + str(stations[i,1]) + '*.cdf')
    if len(matching_files) == 0:
        os.system('python TROPoeSim.py 20240508 pyVIP_site.txt /home/tropoe/vip/src/tropoe/prior.MIDLAT.nc')
    else:
        print('File exists! Skipping')
    i+=1