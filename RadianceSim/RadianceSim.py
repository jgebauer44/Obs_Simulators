#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import glob
from netCDF4 import Dataset
from argparse import ArgumentParser
from datetime import datetime, timedelta
from subprocess import Popen, PIPE


import Namelist_Functions
import VIP_Databases_functions
import Model_Reads
import Calc_Radiances
import Other_functions
import Output_Functions

##############################################################################          
#Create parser for command line arguments

parser = ArgumentParser()

parser.add_argument("date", type=str, help="Date to run the code [YYYYMMDD]")
parser.add_argument("namelist_file", help="Name of the namelist file (string)")
parser.add_argument("--output_dir", help="Path to output directory")
parser.add_argument("--debug", action="store_true", help="Set this to turn on the debug mode")

args = parser.parse_args()

date = args.date
namelist_file = args.namelist_file
output_dir = args.output_dir
debug = args.debug

yy = int(date[0:4])
mm = int(date[4:6])
dd = int(date[6:8])
ymd = yy*10000 + mm*100 + dd
date = datetime(yy,mm,dd)

verbose = 1
 
if output_dir is None:
    output_dir = os.getcwd() + '/'
    
if debug is None:
    debug = False

# We need the background shell to be the C-shell, as we will be spawning out
# a variety of commands that make this assumption. So we will do a
# quick check to find out if the C-shell exists on this system, and if so,
# set the SHELL to this.

process = Popen('which csh', stdout = PIPE, stderr = PIPE, shell=True)
stdout, stderr = process.communicate()

if stdout.decode() == '':
    print('Error: Unable to find the C-shell command on this system')
    print(('>>> RadianceSim FAILED and ABORTED'))
    sys.exit()
else:
    SHELL = stdout[:-1].decode()
    
print("-----------------------------------------------------------------------")
print("Starting RadianceSim")
print("Output directory set to " + output_dir)

# Read the namelist file
namelist = Namelist_Functions.read_namelist(namelist_file)
namelist['output_dir'] = output_dir
if namelist['success'] != 1:
    print('>>> RadianceSim FAILED and ABORTED <<<')
    print("-----------------------------------------------------------------------")
    print(' ')
    sys.exit()
    

# Make sure the all of the stuff for LBLRTM is set up properly and get the 
# tmp directory uniquekey
process = Popen('echo $$', stdout = PIPE, stderr = PIPE, shell=True, executable = SHELL)
stdout, stderr = process.communicate()

uniquekey = namelist['tag'] + '.' + stdout[:-1].decode()

if not os.path.exists(namelist['lbl_home'] + '/bin/lblrun'):
    print('Error: Unable to find the script "lblrun" in the "lbl_home"/bin directory')
    print('This is a critical component of the LBLRTM configuration - aborting')
    print(('>>> RadianceSim FAILED and ABORTED'))
    print('--------------------------------------------------------------------')
    print(' ')
    sys.exit()

# Make sure that the output directory exists
if not os.path.exists(output_dir):
    print('Error: The output directory does not exist')
    print(('>>> RadianceSim FAILED and ABORTED'))
    print('--------------------------------------------------------------------')
    print(' ')
    sys.exit()

# Look at the name of the LBLtmpDir; if it starts with a "$"
# then assume that first part is an environment variable and
# decode the path accordingly

if namelist['lbl_temp_dir'][0] == '$':
    envpath = namelist['lbl_temp_dir'].split('/')
    tmpdir = os.getenv(envpath[0])
    if not tmpdir:
        print('Error: The LBLRTM temporary directory is being set to an environment variable that does not exist')
        print(('>>> RadianceSim FAILED and ABORTED <<<'))
        print('--------------------------------------------------------------------')
        print(' ')
        sys.exit()
    for i in range(1,len(envpath)):
        tmpdir = tmpdir + '/' + envpath[i]
    namelist['lbl_temp_dir'] = tmpdir
    
# Create the temporary working directory
lbltmpdir = namelist['lbl_temp_dir'] + '/' + uniquekey
print('Setting the temporary directory for RT model runs to: ' + lbltmpdir)

try:
    os.makedirs(lbltmpdir)
except:
    print('Error making the temporary directory')
    print(('>>> RadianceSim FAILED and ABORTED <<<'))
    print('--------------------------------------------------------------------')
    print(' ')
    sys.exit()
    

# Read in the SSP databases that are needed to account for clouds.
sspl, flag = VIP_Databases_functions.read_scat_databases(namelist['lcloud_ssp'])
if flag == 1:
    print('Error: Problem reading SSP file for liquid cloud properties')
    print(('>>> RadianceSim FAILED and ABORTED'))
    print('--------------------------------------------------------------------')
    print(' ')
    sys.exit()

sspi, flag = VIP_Databases_functions.read_scat_databases(namelist['icloud_ssp'])
if flag == 1:
    print('Error: Problem reading SSP file for ice cloud properties')
    print(('>>> RadianceSim FAILED and ABORTED'))
    print('--------------------------------------------------------------------')
    print(' ')
    sys.exit()

# Determine the minimum and maximum Reff values in these, but I need to have the
# minimum value be just a touch larger than the actual minimum value in the database
# hence the 1.01 multipliers

minLReff = np.nanmin(sspl['data'][2,:])*1.01
maxLReff = np.nanmax(sspl['data'][2,:])
miniReff = np.nanmin(sspi['data'][2,:])*1.01
maxiReff = np.nanmax(sspi['data'][2,:])

# Print out info on LBLRTM
print(' ')
print(('Working with the LBLRTM version ' + namelist['lbl_version']))
print(('  in the directory ' + namelist['lbl_home']))
print(('  and the TAPE3 file ' + namelist['lbl_tape3']))
print(' ')

# Make sure that the specified TAPE3 file exists
if not os.path.exists(namelist['lbl_home'] + '/hitran/' + namelist['lbl_tape3']):
    print('Error: unable to find the specified TAPE3 file in the LBL_HOME hitran directory')
    print(('>>> RadianceSim FAILED and ABORTED'))
    print('--------------------------------------------------------------------')
    print(' ')
    sys.exit()

# Define some paths and constants
lbldir = lbltmpdir + '/lblout'       # Name of the lbl output directory
lbltp5 = lbltmpdir + '/lbltp5'       # Name of the tape5 file
lbltp3 = namelist['lbl_tape3']           # Name of the tape3 file
lbllog = lbltmpdir + '/lbllog'       # Name of the lbl log file
lbltmp = lbltmpdir + '/lbltmp'       # Temporary directory for LBLRUN (will be LBL_RUN_ROOT)
monortm_config = 'monortm_config.txt'
monortm_zfreq = 'monortm_zfreqs.txt'    # For MWR-zenith calculations
monortm_tfile = 'monortm_sonde.cdf'

# Make two commands: one for MWR-zenith and one for MWR-scan
monortm_zexec = ('cd ' + lbltmpdir + ' ; setenv monortm_config ' + monortm_config +
                ' ; setenv monortm_freqs ' + monortm_zfreq + ' ; ' + namelist['monortm_wrapper'])

create_monortm_config = 1           # Set this flag to create a custom config file for MonoRTM
create_monortm_zfreq = 1            # Set this flag to create a custom freq-zenith file for MonoRTM

# Read in the standard atmosphere information
stdatmos = VIP_Databases_functions.read_stdatmos(namelist['path_std_atmos'], namelist['lbl_std_atmos'],verbose)
if stdatmos['status'] == 0:
    print('Error: Unable to find/read the standard atmosphere file')
    print(('>>> RadianceSim FAILED and ABORTED'))
    print('--------------------------------------------------------------------')
    print(' ')
    sys.exit()

# Define the spectral region(s) that will be simulated
bands = namelist['spectral_bands']
foo = np.where(bands[0,:] >= 0)[0]
if len(foo) <= 0:
    print('Error: the spectral bands do not have any properly defined values')
    print(('>>> RadianceSim FAILED and ABORTED'))
    print('--------------------------------------------------------------------')
    print(' ')
    sys.exit()
bands = bands[:,foo]

# Get the mwr frequencies and noise
parts = namelist['mwr_freq'].split(',')
mwr_freq = np.array(parts).astype(float)

parts = namelist['mwr_noise'].split(',')
mwr_noise = np.array(parts).astype(float)

parts = namelist['mwr_elev'].split(',')
mwr_elev = np.array(parts).astype(float)

# After all of that we are finally ready to start the simulation
# First we are going to going to find out how many simulations we are doing
# based on start_time, end_time, and model_frequency.

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


last_snum = -1
# Check to see if this is an append run. 
if namelist['append'] == 1:
    
    # Make sure the files exists
    if namelist['user_calendar'] == 1:
        outname = output_dir + namelist['outfile_root'] + '_' + str(ymd) +'_' + model_time[0].strftime('%H%M%S') + '.nc' 
        if os.path.exists(output_dir + namelist['outfile_root'] + '_' + str(ymd) +'_' + model_time[0].strftime('%H%M%S') + '.nc' ):
            out = Dataset(output_dir + namelist['outfile_root'] + '_' + str(ymd) +'.nc' )
    else:
        outname = output_dir + namelist['outfile_root'] + '_' + str(ymd) + '_' + str(int(model_time[0])) +'.nc'

        
    if os.path.exists(outname):
        out = Dataset(outname) 
        # Make sure that the start time is the same
        
        if namelist['use_calendar'] == 0:
            if namelist['start_time'] != int(out.start_time):
                print('Append mode was selected, but the start time is not the same.')
                print(('>>> RadianceSim FAILED and ABORTED'))
                print('--------------------------------------------------------------------')
                print(' ')
                sys.exit()
        
        else:
           if start_time.strftime('%Y%m%d_%H%M%S') != out.start_time:
               print('Append mode was selected, but the start time is not the same.')
               print(('>>> RadianceSim FAILED and ABORTED'))
               print('--------------------------------------------------------------------')
               print(' ')
               sys.exit()
        
        # Make sure the user is writing to the same profiler type
        if namelist['profiler_type'] != out.profiler_code:
            print('Append mode was selected, but the simulated profiler is not the same.')
            print(('>>> RadianceSim FAILED and ABORTED'))
            print('--------------------------------------------------------------------')
            print(' ')
            sys.exit() 
        
        # Get the last scan number in the file
        last_snum = int(out.sim_number)
        out.close()
    
    else:
        print('Append mode was selcted, but ' + output_dir + 
              namelist['outfile_root'] + '_' + str(ymd) + '.nc does not exist.')
        print('A new output file will be created!')
        namelist['append'] = 0

# Now check if we need to clobber a file
else:
    if os.path.exists(output_dir + namelist['outfile_root'] + '_' + str(ymd) +'.nc'):
        if namelist['clobber'] == 1:
            print(output_dir + namelist['outfile_root'] + '_' + str(ymd) +'.nc'
                  + ' exists and will be clobbered!')
            os.remove(output_dir + namelist['outfile_root'] + '_' + str(ymd) +'.nc')
            
        else:
            print('ERROR:' + output_dir + namelist['outfile_root'] + '_' + str(ymd) +'.nc' + 
                  ' exists and clobber is set to 0.')
            print('       Must abort to prevent file from being overwritten!')
            print(('>>> RadianceSim FAILED and ABORTED'))
            print('--------------------------------------------------------------------')
            print(' ')
            sys.exit()



# Read in the AERI noise file
f = Dataset(namelist['aeri_noise_file'],'r')
awnum = f.variables['wnum'][:]
anoise = f.variables['noise'][:]
f.close()


w0idx, nY = Other_functions.find_wnum_idx(awnum,bands)
if nY < 0:
    print('Error: Problem selecting the proper wavenumbers for simulation')
    print(('>>> RadianceSim FAILED and ABORTED'))
    print('--------------------------------------------------------------------')
    print(' ')
    sys.exit()

wnum = awnum[w0idx]
noise = anoise[w0idx]

# Select nice round numbers to use as the wavenumber limits for
# the LBLRTM calc, but remember that I need to pad by 50 cm-1 for FSCAN
lblwnum1 = int((np.min(wnum)-60)/100) * 100
lblwnum2 = (int((np.max(wnum)+60)/100)+1)*100
   
# Now we are really ready to simulate some radiances we are going to loop over
# all of the model times. If we are in append mode, we will skip those sims

for index in range(len(snum)):
    
    print('Performing simulation number ' + str(snum[index]) +  ' at time ' + str(model_time[index]))
    
    if index < last_snum:
        '        ...but was already processed. Continuing.'
        continue
    
    
    model, flag = Model_Reads.read_model_data(model_time[index],namelist,sspl,sspi)
    
    if flag == 0:
        print ("Error while reading the model data. Must abort")
        print(('>>> RadianceSim FAILED and ABORTED'))
        print('--------------------------------------------------------------------')
        print(' ')
        sys.exit()
    
    # Do I use a hardcoded value for CO2, or use my Dave Turners simple model to predict
    #the concentration? Unit is ppm

    if namelist['co2_profile'][0] < 0:
       namelist['co2_profile'][0] = Other_functions.predict_co2_concentration(yy, mm, dd)

    # Calculate the trace gas profiles and 
    # Quick test to make sure that the trace gas models make sense. If not, abort
    co2 = Other_functions.trace_gas_prof(0, np.insert(model['z']-model['alt'],0,0), namelist['co2_profile'])
    nfooco2 = len(np.where(co2 < 0)[0])
    ch4 = Other_functions.trace_gas_prof(0, np.insert(model['z']-model['alt'],0,0), namelist['ch4_profile'])
    nfooch4 = len(np.where(ch4 < 0)[0])
    n2o = Other_functions.trace_gas_prof(0, np.insert(model['z']-model['alt'],0,0), namelist['n2o_profile'])
    nfoon2o = len(np.where(n2o < 0)[0])
    if ((nfooco2 > 0) | (nfooch4 > 0) | (nfoon2o > 0)):
        print('Error: The CO2, CH4, and/or N2O parameters are incorrect giving negative values - aborting')
        print(('>>> RadianceSim FAILED and ABORTED'))
        print('--------------------------------------------------------------------')
        print(' ')
        sys.exit()
        
    
    # Add extra layers from standard atmosphere to profiler to simulate radiance from stratosphere
    rt_extra_layers = Other_functions.compute_extra_layers(np.max(model['z']))
    
    # We have the model data so now it is time to do the radiative transfer
    # This is based on the radiative transfer for TROPoe.

    if namelist['profiler_type'] == 1:
        flag, rad, wnumc, totaltime = Calc_Radiances.calc_ir_radiances(model,
                                        namelist['lbl_home'], lbldir, lbltmp,
                                        namelist['lbl_std_atmos'], lbltp5, lbltp3,
                                        co2, ch4, n2o, lblwnum1, lblwnum2,
                                        awnum, rt_extra_layers, stdatmos,
                                        sspl, sspi)
        
        if flag == 0:
            print(' -- Skipping this sample due to problem with radiance calc')
            continue
        
        # Select the wavenumber indices to use
        w1idx, junk = Other_functions.find_wnum_idx(wnumc,bands)
        if len(w1idx) != len(w0idx):
            print('Problemn with wnum indices')
            print(('>>> RadianceSim FAILED and ABORTED'))
            print('--------------------------------------------------------------------')
            print(' ')
            sys.exit()
        
        wnumc = wnumc[w1idx]
        rad = rad[w1idx]
        
        # Finally add noise to the radiance to simulate the noise in IR
        # spectrometers we are making the somewhat nasty assumption that the
        # errors in the channels are uncorrelated
        rng = np.random.default_rng()
        normal = rng.normal(0,noise)
        rad += normal
        
        
    elif namelist['profiler_type'] >= 2:
        # Create the MonoRTM configuration file if we need to
        if create_monortm_config == 1:
            lun = open(lbltmpdir + '/' + monortm_config, 'w')
            lun.write(namelist['monortm_exec'] + '\n')
            lun.write(namelist['monortm_spec'] + '\n')
            lun.write('0\n')          # The verbose flag
            lun.write('{:0d}\n'.format(namelist['lbl_std_atmos']))
            lun.write('1\n')          # The 'output layer optical depths' flag
            for gg in range(6):       # The 6 continuum multipliers
                lun.write('1.0\n')
            lun.write('{:7.3f}\n'.format(np.max(model['z'])-0.01))
            lun.write('{:0d}\n'.format(len(model['z'])+len(rt_extra_layers)+1))
            
            # Write out the surface level first
            lun.write('{:7.3f}\n'.format(0.0))
            for gg in range(len(model['z'])):
                lun.write('{:7.3f}\n'.format(model['z'][gg]-model['alt']))
            for gg in range(len(rt_extra_layers)):
                lun.write('{:7.3f}\n'.format(rt_extra_layers[gg]-model['alt']))
            lun.close()

            # Turn the flag off, as we only need to create these files once
            #create_monortm_config = 0
        
        
        if create_monortm_zfreq == 1:
                # Create the MonoRTM freuency file
                lun = open(lbltmpdir + '/' + monortm_zfreq, 'w')
                lun.write('\n')
                lun.write('{:0d}\n'.format(len(mwr_freq)))
                for gg in range(len(mwr_freq)):
                    lun.write('{:7.3f}\n'.format(mwr_freq[gg]))
                lun.close()

                # Turn the flag off, as we only need to create these files once
                create_monortm_zfreq = 0
        
        if namelist['profiler_type'] == 2:
            flag, mwr_bt, total_time = Calc_Radiances.calc_mwr_brightness_temp(model,mwr_freq,lbltmpdir,
                                               monortm_tfile, monortm_zexec,
                                               stdatmos)
        
            if flag == 0:
                print('-- Skipping this sample due to issue with MonoRTM (likely bad input profile')
                continue
        
            # Finally add noise to the radiance to simulate the noise in MWR
            # We are making the somewhat nasty assumption that error in channels is
            # uncorrelated even though we know they are not
        
            rng = np.random.default_rng()
            normal = rng.normal(0,mwr_noise)
            mwr_bt += normal
        
        elif namelist['profiler_type'] == 3:
            flag, mwrscan_bt = Calc_Radiances.calc_mwrscan_brightness_temp(model,mwr_freq,mwr_elev, 
                                               lbltmpdir,monortm_tfile, monortm_zexec,
                                               stdatmos)
            
            if flag == 0:
                print('-- Skipping this sample due to issue with MonoRTM (likely bad input profile)')
                continue
            
            idx = np.arange(len(mwr_freq))
            
            # Finally add noise to the radiance to simulate the noise in MWR
            # We are making the somewhat nasty assumption that error in channels is
            # uncorrelated even though we know they are not
        
            rng = np.random.default_rng()
            for ii in range(len(mwr_elev)):
                normal = rng.normal(0,mwr_noise)
                mwrscan_bt[ii*len(mwr_freq)+idx] += normal
            
        else:
            print('Profiler type ' + str(namelist['profiler_type']) + ' is not defined')
            print(('>>> RadianceSim FAILED and ABORTED'))
            print('--------------------------------------------------------------------')
            print(' ')
            sys.exit()
    
    
    # We did it! Now lets write the data to an output file
    
    if namelist['profiler_type'] == 1:
        success = Output_Functions.write_output(namelist,output_dir, rad, wnumc,
                                                noise,model, model_time,
                                                snum[index], snum[0], ymd)
        
        if success != 1:
            print('Problem writing data to output file')
            print(('>>> RadianceSim FAILED and ABORTED'))
            print('--------------------------------------------------------------------')
            print(' ')
            sys.exit()
        
    elif namelist['profiler_type'] == 2:
        success = Output_Functions.write_output(namelist,output_dir, mwr_bt, mwr_freq,
                                                mwr_noise,model, model_time,
                                                snum[index], snum[0], ymd)
        
        if success != 1:
            print('Problem writing data to output file')
            print(('>>> RadianceSim FAILED and ABORTED'))
            print('--------------------------------------------------------------------')
            print(' ')
            sys.exit()
        
    elif namelist['profiler_type'] == 3:
        tmp_data = {'elev':np.repeat(mwr_elev,len(mwr_freq)), 'freq':np.tile(mwr_freq,len(mwr_elev))}
        success = Output_Functions.write_output(namelist,output_dir, mwrscan_bt, tmp_data,
                                                np.tile(mwr_noise,len(mwr_elev)),model, model_time,
                                                snum[index], snum[0], ymd)
    
    
        if success != 1:
            print('Problem writing data to output file')
            print(('>>> RadianceSim FAILED and ABORTED'))
            print('--------------------------------------------------------------------')
            print(' ')
            sys.exit()
    
    
    





