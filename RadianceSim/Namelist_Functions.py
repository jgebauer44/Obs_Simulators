import os
import numpy as np


##############################################################################
# This rountine reads in the parameters from the namelist
##############################################################################

def read_namelist(filename):
    
    # This large structure will hold all of the namelist option. We are
    # predefining it so that default values will be used if they are not listed
    # in the namelist. The exception to this is if the user does not specify 
    # the model. In this scenerio the success flag will terminate the program.
    
    maxbands = 200
    
    namelist = ({'success':0,
                 'model':0,              # Type of model used for the simulation. 1-NSSL 5-CM1 (only one that works right now)
                 'model_frequency':0.,    # Frequency of the model output used for the simulation
                 'model_dir':'None',        # Directory with the model data
                 'model_prefix':'None',     # Prefix for the model data files
                 'outfile_root':'None',
                 'append':0,
                 'clobber':0,
                 'coordinate_type':2,             # 1-Lat/Lon, 2-x,y,z (only works with 2 for now)
                 'profiler_x':5000.0,             # x position of simulated profiler in default units (ignored if coordinate_type is 1)
                 'profiler_y':5000.0,             # y position of simulated profiler in default units  (ignored if coordinate_type is 1)
                 'profiler_alt':300.0,            # height of the simulated profiler (m above sea level)
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
                 'profiler_type':1,               # Type of profiler to simulate, 1-AERI/ASSIST, 2-MWR zenith
                 'aeri_noise_file':'None',        # Path to the file that contains the information about the aeri noise
                 
                 'lbl_home':'/home/tropoe/vip/src/lblrtm_v12.1/lblrtm',   # String with the LBL_HOME path (environment variable)
                 'lbl_version':'v12.1',            # String with  the version information on LBLRTM
                 'lbl_temp_dir':'/tmp',            # Temporary working directory for the retrieval
                 'lbl_std_atmos':6,                 # Standard atmosphere to use in LBLRTM and MonoRTM calcs
                 'path_std_atmos':'/home/tropoe/vip/src/input/idl_code/std_atmosphere.idl',  # The path to the IDL save file with the standard atmosphere
                 'lbl_tape3':'tape3.data',        # The TAPE3 file to use in the lblrtm calculation. Needs to be in the the directory lbl_home/hitran/
                 'monortm_version':'v5.0',         # String with the version information on MonoRTM
                 'monortm_wrapper':'/home/tropoe/vip/src/monortm_v5.0/wrapper/monortm_v5', # Turner wrapper to run MonoRTM
                 'monortm_exec':'/home/tropoe/vip/src/monortm_v5.0/monortm/monortm_v5.0_linux_intel_sgl',     # AERs MonoRTM executable
                 'monortm_spec':'/home/tropoe/vip/src/monortm_v5.0/monolnfl_v1.0/TAPE3.spectral_lines.dat.0_55.v5.0_veryfast', #MonoRTM spectral database
                 'tag':'RadianceSim',              # String for temporary file directories
                 
                 'mp_scheme':5,                    # Microphysics scheme. 1-NSSL, 2-Morrison
                 'ndcnst':250,                     # Number of cloud droplets per cm-1 in Morrison scheme
                 'co2_profile':[-1.0,5,-5],        # Mean CO2 concentration, -1 in first element means predict CO2 based on date
                 'ch4_profile':[1.793,0,-5],       # Mean CH4 concentration [ppm]
                 'n2o_profile':[0.310,0,-5],       # Mean N2O concentration [ppm]
                 'lcloud_ssp':'/home/tropoe/vip/src/input/ssp_db_files/ssp_db.mie_wat.gamma_sigma_0p100',      # SSP file for liquid cloud properties
                 'icloud_ssp':'/home/tropoe/vip/src/input/ssp_db_files/ssp_db.mie_ice.gamma_sigma_0p100',      # SSP file for ice cloud properties
                 
                 'mwr_freq':'23.8,31.4',           # Comma separated string of frequencies [GHz] of MWR Tb fields
                 'mwr_noise':'0.3,0.3',            # Comma separated string  of noise levels [K]
                 'mwr_elev':'90',
                 
                 'spectral_bands':'None'           # An array of spectral band to retrieve (e.g. 612-618,624-660,674-713,713-722,538-588,860.1-864.0,872.2-877.5,898.2-905.4)'
                 })


    if os.path.exists(filename):
        print('Reading the namelist: ' + filename)
        
        try:
            inputt = np.genfromtxt(filename, dtype=str, comments ='#',delimiter='=', autostrip=True)
        except:
            print ('ERROR: There was a problem reading the namelist')
            return namelist
    
    else:
        print('ERROR: The namelist file ' + namelist + ' does not exist')
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
                 
                if key == 'spectral_bands':
                    bands = np.zeros((2,maxbands))-1
                    tmp = inputt[foo,1][0].split(',')
                     
                    if len(tmp) >= maxbands:
                        print('Error: There were more spectral bands defined than maximum allowed (maxbands = ' + str(maxbands) + ')')
                        return namelist
                
                    for j in range(len(tmp)):
                        feh = tmp[j].split('-')
                        if len(feh) != 2:
                            print('Error: Unable to properly decompose the spectral_bands key')
                            return namelist
                    
                        bands[0,j] = float(feh[0])
                        bands[1,j] = float(feh[1])
                
                    namelist['spectral_bands'] = bands
                
                elif ((key == 'co2_profile') or
                      (key == 'ch4_profile') or
                      (key == 'n20_profile')):
                    
                    feh = inputt[foo,1][0].split(',')
                    if len(feh) != len(namelist[key]):
                        print('Error: The key ' + key + ' in namelist file must be a ' +
                              str(len(namelist[key])) + ' element array')
                        return namelist
                    
                    namelist[key][0] = float(feh[0])
                    namelist[key][1] = float(feh[1])
                    namelist[key][2] = float(feh[2])
                
                else:
                    namelist[key] = type(namelist[key])(inputt[foo,1][0])
            else:
                nfound -= 1
    
    # Need to trap condition where spectral_bands was not set (like a MWR retrieval)
    # and reset it to a standard set of bands
    if (type(namelist['spectral_bands']) == str):
        blo = [612, 624, 674, 713, 538, 860.1, 872.2, 898.2]
        bhi = [618, 660, 713, 722, 588, 864.0, 877.5, 905.4]
        namelist['spectral_bands'] = np.array([blo,bhi])
    
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