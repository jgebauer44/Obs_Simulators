import os
import shutil
import numpy as np
import scipy.io

################################################################################
# This file contains the following functions:
# read_vip_file()
# check_vip_file()
# abort()
# read_scat_databases()
# read_stdatmos()
################################################################################


# This is the stucture with all of the input that we need.
# The code below will be searching the VIP input file for the same keys
# that are listed in this structure. If a line is found with the same
# name , then the code will determine the type for the value
# (e.g., is the value for that key a float, string, integer, etc) and then
# will cast the value provided in the VIP file as that.
# There are three exceptions to this
#   1) "Success" has the status of this routine -- it is not set by VIP file
#   2) "aeri_calib_pres" is a 2-elements floating point array
#   3) "vip_filename" captures the name of the input VIP file itself
# The code will ouput how many of the keys in this structure were found.
# Note that not all of them have to be in the VIP file; if a key in this
# structure is not found in the VIP file then it maintains it default value

maxbands = 200  # The maximum number of bands to enable for the retrieval

full_vip = ({
    'success': {'value': 0, 'comment': "Interal success flag. Not for outside use", 'default': False},
    'tres': {'value': 0, 'comment': 'Temporal resolution [min], 0 implies native AERI temporal resolution', 'default': True},
    'avg_instant': {'value': 1, 'comment': 'A flag to specify if this is an average (0) over the tres period, or instantaneous (1) sample (i.e. do not average the data)', 'default': True},
    'tag': {'value': 'tropoe', 'comment': 'String for temporary files / directories\n', 'default': True},

    'aeri': {'value': 0, 'comment': '0-Do not use AERI data, 1 - Use AERI data', 'default': True},
    'aeri_path': {'value': "None", 'comment': 'Path to the AERI ch1 radiance files', 'default': True},
    'aeri_noise_file' : {'value':"None", 'comment': 'Path to the AERI uncertainty data', 'default':True},
    'irs_min_noise_flag':{'value': 1, 'comment': 'If non-zero, then the irs_min_noise_spectrum will be used as a floor; otherwise, will use input IRS noise spectrum from instrument', 'default': True},
    'irs_min_noise_wnum':{'value':'500,522,546,575,600,631,747,1439,1770,1884,2217,3000', 'comment': 'Wavenumber array [cm-1] for the minimum noise spectrum', 'default': True},
    'irs_min_noise_spec':{'value': '65.309,14.056,3.283,1.333,0.813,0.557,0.304,0.581,0.822,0.025,0.023,0.044', 'comment': 'Noise array [RU] for the minimum noise spectrum', 'default': True},
    
    'station_lat': {'value': -999., 'comment': 'Station latitude [degN]; if negative get value from AERI data file', 'default': True},
    'station_lon': {'value': -999., 'comment': 'Station longitude [degE]; if negative get value from AERI data file', 'default': True},
    'station_alt': {'value': -999., 'comment': 'Station altitude [m MSL]; if negative get value from AERI data file', 'default': True},
    'station_psfc_min': {'value': 800., 'comment': 'Default minimum surface pressure [mb]', 'default': True},
    'station_psfc_max': {'value': 1030., 'comment': 'Default maximum surface pressure [mb]\n', 'default': True},

    'ext_wv_prof_type': {'value': 0, 'comment': 'External WV profile source: 0-none; 1-sonde; 2-ARM Raman lidar (rlprofmr), 3-NCAR WV DIAL, 4-Model sounding', 'default': True},
    'ext_wv_prof_path': {'value': 'None', 'comment': 'Path to the external profile of WV data', 'default': True},
    'ext_wv_prof_minht': {'value': 0.0, 'comment': 'Minimum height to use the data from the external WV profiler [km AGL]', 'default': True},
    'ext_wv_prof_maxht': {'value': 10.0, 'comment': 'Maximum height to use the data from the external WV profiler [km AGL]', 'default': True},
    'ext_wv_noise_mult_val': {'value': [1.0, 1, 1.0], 'comment': '3-element comma delimited list with the multipliers to apply the noise profile of the external water vapor profile (must be > 1)', 'default': True},
    'ext_wv_noise_mult_hts': {'value': [0.0, 3, 20], 'comment': '3-element comma delimited list with the corresponding heights for the noise multipliers [km AGL]', 'default': True},
    'ext_wv_add_rel_error': {'value': 0.0, 'comment': 'When using the RLID, I may want to include a relative error contribution to the uncertainty to account for calibration. This is a correlated error component, and thus effects the off-diagonal elements of the observation covariance matrix. Units are [%]', 'default': True},
    'ext_wv_time_delta': {'value': 1.0, 'comment': 'Maximum amount of time from endpoints of external WV dataset to extrapolate [hours]\n', 'default': True},

    'ext_temp_prof_type': {'value': 0, 'comment': 'External temperature profile source: 0-none; 1-sonde; 2-ARM Raman lidar (rlproftemp); 4-Model sounding', 'default': True},
    'ext_temp_prof_path': {'value': 'None', 'comment': 'Path to external profile of temp data', 'default': True},
    'ext_temp_prof_minht': {'value': 0.0, 'comment': 'Minimum height to use the data from the external temp profiler [km AGL]', 'default': True},
    'ext_temp_prof_maxht': {'value': 10.0, 'comment': 'Maximum height to use the data from the external temp profiler [km AGL]', 'default': True},
    'ext_temp_noise_mult': {'value': 1.0, 'comment': 'Multiplicative value to apply the noise profile of the external temperature profile (must be >= 0)', 'default': True},
    'ext_temp_noise_adder_val': {'value': [0.0, 0, 0], 'comment': '3-element comma delimited list of additive values to apply the noise profile of the external temperature profile (must be >= 0)', 'default': True},
    'ext_temp_noise_adder_hts': {'value': [0.0, 3, 20], 'comment': '3-element comma delimited list with the corresponding heights for the additive value [km AGL]', 'default': True},
    'ext_temp_time_delta': {'value': 1.0, 'comment': 'Maximum amount of time from endpoints of external temp dataset to extrapolate [hours]\n', 'default': True},

    'mod_wv_prof_type': {'value': 0, 'comment': 'NWP model WV profile source: 0-none; 4-Model sounding', 'default': True},
    'mod_wv_prof_path': {'value': 'None', 'comment': 'Path to the model profile of WV data', 'default': True},
    'mod_wv_prof_minht': {'value': 0.0, 'comment': 'Minimum height to use the data from the model WV profiler [km AGL]', 'default': True},
    'mod_wv_prof_maxht': {'value': 10.0, 'comment': 'Maximum height to use the data from the model WV profiler [km AGL]', 'default': True},
    'mod_wv_noise_mult_val': {'value': [1.0, 1, 1.0], 'comment': '3-element comma delimited list with the multipliers to apply the noise profile of the model water vapor profile (must be > 1)', 'default': True},
    'mod_wv_noise_mult_hts': {'value': [0.0, 3, 20], 'comment': '3-element comma delimited list with the corresponding heights for the noise multipliers [km AGL]', 'default': True},
    'mod_wv_time_delta': {'value': 1.0, 'comment': 'Maximum amount of time from endpoints of model WV dataset to extrapolate [hours]\n', 'default': True},

    'mod_temp_prof_type': {'value': 0, 'comment': 'NWP model temperature profile source: 0-none; 4-Model sounding', 'default': True},
    'mod_temp_prof_path': {'value': 'None', 'comment': 'Path to the model profile of temp data', 'default': True},
    'mod_temp_prof_minht': {'value': 0.0, 'comment': 'Minimum height to use the data from the model temp profiler [km AGL]', 'default': True},
    'mod_temp_prof_maxht': {'value': 10.0, 'comment': 'Maximum height to use the data from the model temp profiler [km AGL]', 'default': True},
    'mod_temp_noise_mult': {'value': 1.0, 'comment': 'Multiplicative value to apply the noise profile of the model temperature profile (must be >= 0)', 'default': True},
    'mod_temp_noise_adder_val': {'value': [0.0, 0, 0], 'comment': '3-element comma delimited list of additive values to apply the noise profile of the model temperature profile (must be >= 0)', 'default': True},
    'mod_temp_noise_adder_hts': {'value': [0.0, 3, 20], 'comment': '3-element comma delimited list with the corresponding heights for the additive value [km AGL]', 'default': True},
    'mod_temp_time_delta': {'value': 1.0, 'comment': 'Maximum amount of time from endpoints of model WV dataset to extrapolate [hours]\n', 'default': True},

    'ext_sfc_temp_type': {'value': 0, 'comment': 'External surface temperature met data type: 0-none, 1-ARM met data [degC], 2-NCAR ISFS data [degC], 3-CLAMPS MWR met data [degC]', 'default': True},
    'ext_sfc_temp_npts': {'value': 1, 'comment': 'Number of surface temperature met points to use in the retrieval.  Minimum=1, maximum=1000.  Larger number increases the weight of the observation', 'default': True},
    'ext_sfc_temp_rep_error': {'value': 0.0, 'comment': 'Representativeness error for the surface temperature measurement [degC], which is added to the typical assumed uncertainty of 0.5 degC', 'default': True},
    'ext_sfc_wv_type': {'value': 0, 'comment': 'External surface water vapor met data type: 0-none, 1-ARM met data [g/kg], 2-NCAR ISFS data [g/kg], 3-CLAMPS MWR met data [g/kg]', 'default': True},
    'ext_sfc_wv_npts': {'value': 1, 'comment': 'Number of surface water vapor met points to use in the retrieval.  Minimum=1, maximum=1000.  Larger number increases the weight of the observation', 'default': True},
    'ext_sfc_wv_mult_error': {'value': 1.0, 'comment': 'Multiplier for the error in the surface water vapor measurement.  This is applied BEFORE the \'rep_error\' value', 'default': True},
    'ext_sfc_wv_rep_error': {'value': 0.0, 'comment': 'Representativeness error for the surface water vapor measurement [g/kg], which is added to the typical assumed uncertainty of 0.5 degC and 3%RH', 'default': True},
    'ext_sfc_path': {'value': 'None', 'comment': 'Path to the external surface met data', 'default': True},
    'ext_sfc_time_delta': {'value': 0.2, 'comment': 'Maximum amount of time from endpoints of external surface met dataset to extrapolate [hours]', 'default': True},
    'ext_sfc_relative_height': {'value': 0, 'comment': 'Relative height of the met station to the AERI zenith port [m]; note if met station is below AERI port then the value should be negative', 'default': True},
    'ext_sfc_p_type': {'value': 0, 'comment': '0 - Use the internal AERI pressure sensor for psfc; 1-ARM met data, 2-NCAR ISFS data, 3-CLAMPS MWR met data\n', 'default': True},

    'co2_sfc_type': {'value': 0, 'comment': 'External CO2 surface data type: 0-none, 1-DDT QC PGS data', 'default': False},
    'co2_sfc_npts': {'value': 1, 'comment': 'Number of surface CO2 in-situ points to use in the retrieval.  Minimum=1, maximum=1000.  Larger number increases the weight of the observation', 'default': False},
    'co2_sfc_rep_error': {'value': 0.0, 'comment': 'Representativeness error for the CO2 surface measurement [ppm], which is added to the uncertainty of the obs in the input file', 'default': False},
    'co2_sfc_path': {'value': 'None', 'comment': 'Path to the external surface CO2 data', 'default': False},
    'co2_sfc_relative_height': {'value': 0, 'comment': 'Relative height of the CO2 surface measurement to the AERI zenith port [m]; note if in-situ obs is below AERI port then the value should be negative', 'default': False},
    'co2_sfc_time_delta': {'value': 1.5, 'comment': 'Maximum amount of time from endpoints of external CO2 in-situ dataset to extrapolate [hours] \n', 'default': False},

    'mwr': {'value': 0, 'comment': '0 - none, 1 - Tb fields are individual time series, 2 - Tb field is 2-d array', 'default': True},
    'mwr_path': {'value': 'None', 'comment': 'Path to the MWR data', 'default': True},
    'mwr_rootname': {'value': 'mwr', 'comment': 'Rootname of the MWR data files', 'default': True},
    'mwr_n_tb_fields': {'value': 0, 'comment': 'Number of fields to read in', 'default': True},
    'mwr_tb_freqs': {'value': '23.8,31.4', 'comment': 'Comma separated list of frequency [GHz] of MWR Tb fields', 'default': True},
    'mwr_tb_noise': {'value': '0.3,0.3', 'comment': 'Comma separated list of noise levels [K] in the MWR Tb fields', 'default': True},
    'mwr_tb_bias': {'value': '0.0,0.0', 'comment': 'Comma separated list of bias [K] in the MWR Tb fields; this value is ADDED to the MWR observations', 'default': True},

    'mwrscan': {'value': 0, 'comment': '0 - none, 1 - Use MWR scan data', 'default': True},
    'mwrscan_path': {'value': 'None', 'comment': 'Path to the MWRscan data', 'default': True},
    'mwrscan_rootname': {'value': 'mwr', 'comment': 'Rootname of the MWRscan data files', 'default': True},
    'mwrscan_elev_field': {'value': 'elev', 'comment': 'Name of the scene mirror elevation field; this field must exist', 'default': True},
    'mwrscan_n_tb_fields': {'value': 0, 'comment': 'Number of fields to read in', 'default': True},
    'mwrscan_n_elevations': {'value': 2, 'comment': 'The number of elevations to use in retrieval (put zenith obs in \'mwr_type\')', 'default': True},
    'mwrscan_elevations': {'value': '20,160', 'comment': 'The elevation angles to use in deg, where 90 is zenith.  The code will look for these obs within the averaging interval', 'default': True},
    'mwrscan_tb_field_names': {'value': 'tbsky23,tbsky31', 'comment': 'Comma separated list of field names for the Tb fields', 'default': True},
    'mwrscan_tb_freqs': {'value': '23.8,31.4', 'comment': 'Comma separated list of frequency [GHz] of MWR', 'default': True},
    'mwrscan_tb_noise': {'value': '0.3,0.3', 'comment': 'Comma separated list of noise levels [K] in the MWR Tb fields', 'default': True},
    'mwrscan_tb_bias': {'value': '0.0,0.0', 'comment': 'Comma separated list of bias [K] in the MWR Tb fields; this value is ADDED to the MWR observations', 'default': True},

    'rass_prof_type': {'value': 0, 'comment': '0 - none, 5 - RASS Tv field has units C (no other values work)', 'default': True},
    'rass_prof_path': {'value': 'None', 'comment': 'Path to the RASS data', 'default': True},
    'rass_prof_minht': {'value': 0.0, 'comment': 'Minimum height to use the data from the RASS [km AGL]', 'default': True},
    'rass_prof_maxht': {'value': 5.0, 'comment': 'Maximum height to use the data from the RASS [km AGL]', 'default': True},
    'rass_noise_adder_val': {'value': [0.0, 0.0, 0.0], 'comment': '3-element comma delimited list of additive values to apply the noise profile of the RASS temperature profile (must be >= 0).', 'default': True},
    'rass_noise_adder_hts': {'value': [0.0, 3, 20], 'comment': '3-element comma delimited list with the corresponding heights for the additive value [km AGL]', 'default': True},
    'rass_time_delta': {'value': 2.5, 'comment': 'The maximum amount of time [hours] that the RASS sample must be to the sampling time to be used. \n', 'default': True},

    'cbh_default_ht': {'value': 2.0, 'comment': 'Default CBH height [km AGL], if no CBH data found \n', 'default': True},

    'output_rootname': {'value': 'None', 'comment': 'String with the rootname of the output file', 'default': True},
    'output_path': {'value': 'None', 'comment': 'Path where the output file will be placed', 'default': True},
    'output_clobber': {'value': 0, 'comment': '0 - do not clobber preexisting output files, 1 - clobber them, 2 - append to the last file of this day', 'default': True},
    'output_file_keep_small': {'value': 0, 'comment': '0 - all fields written; 1 - keep output file small by not including Sop, Akern, others \n', 'default': True},

    'lbl_home': {'value': '/home/tropoe/vip/src/lblrtm_v12.1/lblrtm', 'comment': 'String with the LBL_HOME path (environment variable)', 'default': False},
    'lbl_version': {'value': 'v12.1', 'comment': 'String with the version information on LBLRTM', 'default': False},
    'lbl_temp_dir': {'value': '/tmp', 'comment': 'Temporary working directory for the retrieval', 'default': False},
    'lbl_std_atmos': {'value': 6, 'comment': 'Standard atmosphere to use in LBLRTM and MonoRTM calcs', 'default': False},
    'path_std_atmos': {'value': '/home/tropoe/vip/src/input/idl_code/std_atmosphere.idl', 'comment': 'The path to the IDL save file with the standard atmosphere info in it', 'default': False},
    'lbl_tape3': {'value': 'tape3.data', 'comment': 'The TAPE3 file to use in the lblrtm calculation.  Needs to be in the directory lbl_home/hitran/', 'default': False},
    'monortm_version': {'value': 'v5.0', 'comment': 'String with the version information on MonoRTM', 'default': False},
    'monortm_wrapper': {'value': '/home/tropoe/vip/src/monortm_v5.0/wrapper/monortm_v5', 'comment': 'Turner wrapper to run MonoRTM', 'default': False},
    'monortm_exec': {'value': '/home/tropoe/vip/src/monortm_v5.0/monortm/monortm_v5.0_linux_gnu_sgl', 'comment': 'AERs MonoRTM executable', 'default': False},
    'monortm_spec': {'value': '/home/tropoe/vip/src/monortm_v5.0/monolnfl_v1.0/TAPE3.spectral_lines.dat.0_55.v5.0_veryfast', 'comment': 'MonoRTM spectral database\n', 'default': False},

    'lblrtm_jac_option': {'value': 4, 'comment': '1 - LBLRTM Finite Diffs, 2 - 3calc method, 3 - deltaOD method, 4 - interpol method', 'default': True},
    'lblrtm_forward_threshold': {'value': 0., 'comment': 'The upper LWP threshold [g/m2] to use LBLRTM vs. radxfer in forward calculation', 'default': True},
    'lblrtm_jac_interpol_npts_wnum': {'value': 10, 'comment': 'The number of points per wnum to use in the compute_jacobian_interpol() function', 'default': True},
    'monortm_jac_option': {'value': 2, 'comment': '1 - MonoRTM Finite Diffs, 2 - 3calc method', 'default': True},
    'jac_max_ht': {'value': 8.0, 'comment': 'Maximum height to compute the Jacobian [km AGL]', 'default': True},
    'max_iterations': {'value': 10, 'comment': 'The maximum number of iterations to use', 'default': True},
    'first_guess': {'value': 1, 'comment': '1 - use prior as FG, 2 - use lapse rate and 60% RH profile as FG, 3 - use previous sample as FG', 'default': True},
    'superadiabatic_maxht': {'value': 0.300, 'comment': 'The maximum height a superadiabatic layer at the surface can have [km AGL]', 'default': True},
    'spectral_bands': {'value': "None", 'comment': 'An array of spectral bands to use (e.g. 612-618,624-660,674-713,713-722,538-588,860.1-864.0,872.2-877.5,898.2-905.4)', 'default': True},
    'retrieve_temp': {'value': 1, 'comment': '0 - do not retrieve temp, 1 - do retrieve temp (default)', 'default': True},
    'retrieve_wvmr': {'value': 1, 'comment': '0 - do not retrieve wvmr, 1 - do retrieve wvmr (default)', 'default': True},
    'retrieve_co2': {'value': 0, 'comment': '0 - do not retrieve co2, 1 - do retrieve co2 (exponential model), 2 - do retrieve co2 (step model)', 'default': False},
    'fix_co2_shape': {'value': 0, 'comment': '(This option only works with retrieve_co2=1): 0 - retrieve all three coefs, 1 - shape coef is f(PBLH) and fixed', 'default': False},
    'retrieve_ch4': {'value': 0, 'comment': '0 - do not retrieve ch4, 1 - do retrieve ch4 (exponential model), 2 - do retrieve co2 (step model)', 'default': False},
    'fix_ch4_shape': {'value': 0, 'comment': '(This option only works with retrieve_ch4=1): 0 - retrieve all three coefs, 1 - shape coef is f(PBLH) and fixed', 'default': False},
    'retrieve_n2o': {'value': 0, 'comment': '0 - do not retrieve n2o, 1 - do retrieve n2o (exponential model), 2 - do retrieve co2 (step model)', 'default': False},
    'fix_n2o_shape': {'value': 0, 'comment': '(This option only works with retrieve_n2o=1): 0 - retrieve all three coefs, 1 - shape coef is f(PBLH) and fixed', 'default': False},
    'retrieve_lcloud': {'value': 1, 'comment': '0 - do not retrieve liquid clouds, 1 - retrieve liquid cloud properties', 'default': True},
    'retrieve_icloud': {'value': 0, 'comment': '0 - do not retrieve   ice  clouds, 1 - retrieve   ice  cloud properties', 'default': True},
    'lcloud_ssp': {'value': '/home/tropoe/vip/src/input/ssp_db_files/ssp_db.mie_wat.gamma_sigma_0p100', 'comment': 'SSP file for liquid cloud properties', 'default': False},
    'icloud_ssp': {'value': '/home/tropoe/vip/src/input/ssp_db_files/ssp_db.mie_ice.gamma_sigma_0p100', 'comment': 'SSP file for   ice  cloud properties', 'default': False},
    'qc_rms_value': {'value': 10.0, 'comment': 'The RMS value between ((obs minus calc)/obs_uncert) spectra, with values less than this being \'good\'.  In short, if the solution is within n-sigma of the observation (where \'n\' is given by this value, then the retrieval is good', 'default': True},
    'prior_t_ival': {'value': 1.0, 'comment': 'The prior inflation factor (>= 1) to apply at the surface for temperature', 'default': True},
    'prior_t_iht': {'value': 1.0, 'comment': 'The height [km AGL] where the inflation factor goes to 1 (linear) for temperature', 'default': True},
    'prior_q_ival': {'value': 1.0, 'comment': 'The prior inflation factor (>= 1) to apply at the surface for water vapor mixing ratio', 'default': True},
    'prior_q_iht': {'value': 1.0, 'comment': 'The height [km AGL] where the inflation factor goes to 1 (linear) for water vapor mixing ratio', 'default': True},
    'prior_tq_cov_val': {'value': 1.0, 'comment': 'A multiplicative value (0 < val <= 1) that is used to decrease the covariance in the prior between temperature and WVMR at all heights', 'default': False},
    'prior_chimney_ht': {'value': 0.0, 'comment': 'The height of any \'chimney\' [km AGL]; prior data below this height are totally decorrelated', 'default': False},
    'prior_co2_mn': {'value': [-1.0, 5, -5], 'comment': 'Mean co2 concentration [ppm] (see \'retrieve_co2\' above)', 'default': False},
    'prior_co2_sd': {'value': [2.0, 15, 3], 'comment': '1-sigma uncertainty in co2 [ppm]', 'default': False},
    'prior_ch4_mn': {'value': [1.793, 0, -5], 'comment': 'Mean ch4 concentration [ppm] (se}e \'retrieve_ch4\' above)', 'default': False},
    'prior_ch4_sd': {'value': [0.0538, 0.0015, 3], 'comment': '1-sigma uncertainty in ch4 [ppm]', 'default': False},
    'prior_n2o_mn': {'value': [0.310, 0, -5], 'comment': 'Mean n2o concentration [ppm] (see \'retrieve_n2o\' above)', 'default': False},
    'prior_n2o_sd': {'value': [0.0093, 0, 3], 'comment': '1-sigma uncertainty in n2o [ppm]', 'default': False},
    'prior_lwp_mn': {'value':  0.0, 'comment': 'Mean LWP [g/m2]', 'default': True},
    'prior_lwp_sd': {'value': 50.0, 'comment': '1-sigma uncertainty in LWP [g/m2]', 'default': True},
    'prior_lReff_mn': {'value': 8.0, 'comment': 'Mean liquid Reff [microns]', 'default': True},
    'prior_lReff_sd': {'value': 4.0, 'comment': '1-sigma uncertainty in liquid Reff [microns]', 'default': True},
    'prior_itau_mn': {'value': 0.0, 'comment': 'Mean ice cloud optical depth (geometric limit)', 'default': True},
    'prior_itau_sd': {'value': 5.0, 'comment': '1-sigma uncertainty in ice cloud optical depth', 'default': True},
    'prior_iReff_mn': {'value': 25.0, 'comment': 'Mean ice cloud Reff [microns]', 'default': True},
    'prior_iReff_sd': {'value': 8.0, 'comment': '1-sigma uncertainty in ice cloud} Reff [Reff]', 'default': True},
    'min_PBL_height': {'value': 0.3, 'comment': 'The minimum height of the planetary boundary l}ayer (used for trace gases) [km AGL]', 'default': False},
    'max_PBL_height': {'value': 5.0, 'comment': 'The maximum height of the planetary boundary layer (used for trace gases) [km AGL]', 'default': False},
    'nudge_PBL_height': {'value': 0.5, 'comment': 'The temperature offset (nudge) added to the surface temp to find PBL height [C]', 'default': False},
    'recenter_prior': {'value':1, 'comment': '0 - do not recenter prior, 1 - recenter prior', 'default': True}
}
)

################################################################################
#This routine reads in the controlling parameters from the VIP file.
################################################################################

def read_vip_file(filename,globatt,verbose,debug,dostop):

    # Read in the all the values from the full vip dict and create a smaller vip file
    vip = {}
    for key in full_vip.keys():
        vip[key] = full_vip[key]['value']

    if os.path.exists(filename):

        if verbose >= 1:
            print('Reading the VIP file: ' + filename)

        try:
            # inputt = np.genfromtxt(filename, dtype=str,comments='#', usecols = (0,1,2))
            inputt = np.genfromtxt(filename, dtype=str, comments='#', delimiter='=', autostrip=True)
        except Exception as e:
            print('There was an problem reading the VIP file')
    else:
        print('The VIP file ' + filename + ' does not exist')
        return vip

    if len(inputt) == 0:
        print('There were no valid lines found in the VIP file')
        return vip

    # Look for obsolete tags, and abort if they are found (forcing user to update their VIP file)
    obsolete_tags = ['AERI_LAT','AERI_LON','AERI_ALT','PSFC_MIN','PSFC_MAX']
    obsolete_idx  = np.zeros_like(obsolete_tags, dtype=int)
    vip_keys = [k.upper() for k in vip.keys()]
    for i in range(len(obsolete_tags)):
        foo = np.where(obsolete_tags[i].upper() == vip_keys)[0]
        if(len(foo) > 0):
            obsolete_idx[i] = 1
    foo = np.where(obsolete_idx > 0)[0]
    if(len(foo) > 0):
        print('Error: there were obsolete tags in the VIP file:')
        for i in range(len(foo)):
            print('     '+obsolete_tags[foo[i]])
            return vip

    # Look for these tags

    nfound = 1
    for key in vip.keys():
        if key != 'success':
            nfound += 1
            if key == 'vip_filename':
                vip['vip_filename'] = filename
            else:
                foo = np.where(key == inputt[:,0])[0]
                if len(foo) > 1:
                    print('Error: There were multiple lines with the same key in VIP file: ' + key)
                    return vip

                elif len(foo) == 1:
                    if verbose == 3:
                        print('Loading the key ' + key)
                    if key == 'spectral_bands':
                        bands = np.zeros((2, maxbands))-1
                        tmp = inputt[foo,1][0].split(',')

                        if len(tmp) >= maxbands:
                            print('Error: There were more spectral bands defined than maximum allowed (maxbands = ' + str(maxbands) + ')')
                            return vip

                        for j in range(len(tmp)):
                            feh = tmp[j].split('-')
                            if len(feh) != 2:
                                print('Error: Unable to properly decompose the spectral_bands key')
                                if dostop:
                                    wait = input('Stopping inside to debug this bad boy. Press enter to continue')
                                return vip
                            bands[0,j] = float(feh[0])
                            bands[1,j] = float(feh[1])
                        vip['spectral_bands'] = bands

                    elif ((key == 'ext_wv_noise_mult_val') |
                          (key == 'ext_wv_noise_mult_hts') |
                          (key == 'ext_temp_noise_adder_val') |
                          (key == 'ext_temp_noise_adder_hts') |
                          (key == 'mod_wv_noise_mult_val') |
                          (key == 'mod_wv_noise_mult_hts') |
                          (key == 'mod_temp_noise_adder_val') |
                          (key == 'mod_temp_noise_adder_hts') |
                          (key == 'rass_noise_adder_val') |
                          (key == 'rass_noise_adder_hts') |
                          (key == 'prior_co2_mn') |
                          (key == 'prior_co2_sd') |
                          (key == 'prior_ch4_mn') |
                          (key == 'prior_ch4_sd') |
                          (key == 'prior_n2o_mn') |
                          (key == 'prior_n2o_sd') ):

                        feh = inputt[foo,1][0].split(',')
                        if len(feh) != len(vip[key]):
                            print('Error: The key ' + key + ' in VIP file must be a ' + str(len(vip[key])) + ' element array')
                            if dostop:
                                wait = input('Stopping inside to debug this bad boy. Press enter to continue')
                            return vip

                        vip[key][0] = float(feh[0])
                        vip[key][1] = float(feh[1])
                        vip[key][2] = float(feh[2])

                    else:
                        vip[key] = type(vip[key])(inputt[foo,1][0])
                else:
                    if verbose == 3:
                        print('UNABLE to find the key ' + key)
                    nfound -= 1

    if verbose == 3:
        print(vip)
    if verbose >= 2:
        print('There were ' + str(nfound) + ' entries found out of ' + str(len(list(vip.keys()))))

    # Now look for any global attributes that might have been entered in the file

    matching = [s for s in inputt[:,0] if "globatt" in s]

    if verbose >= 2:
        print('There were ' + str(len(matching)) + ' global attributes found')

    for i in range(len(matching)):
        foo = np.where(matching[i] == inputt[:,0])[0]
        globatt[matching[i][8:]] = inputt[foo,1][0]

    # Need to trap condition where spectral_bands was not set (and thus is the default string "None")
    # and then reset it to a standard set of bands used for IR sounding
    if(type(vip['spectral_bands']) == str):
        blo = [612., 624, 674, 713, 538, 860.1, 872.2, 898.2]
        bhi = [618., 660, 713, 722, 588, 864.0, 877.5, 905.4]
        vip['spectral_bands'] = np.array([blo,bhi])

    vip['success'] = 1

    if dostop:
        wait = input('Stopping inside to debug this bad boy. Press enter to continue')
    return vip

################################################################################
# This code performs some QC on the entries in the VIP structure to ensure
# that they are within a valid range.  Not every entry is checked...
################################################################################

def check_vip(vip):
    flag = 0           # Default is that everything is ok

    if ((vip['output_clobber'] < 0) | (vip['output_clobber'] > 2)):
        print('Error: The output_clobber flag can only be set to 0, 1, or 2')
        flag = 1

    if vip['jac_max_ht'] <= 1.0:
        print('Error: The maximum height to compute the Jacobian is too small; please increase')
        flag = 1

    if ((vip['max_iterations'] < 0) | (vip['max_iterations'] > 25)):
        print('Error: The maximum number of iterations must be between 0 and 25')
        flag = 1

    if ((vip['lbl_std_atmos'] < 1) | (vip['lbl_std_atmos'] > 6)):
        print('Error: The LBLRTM standard atmosphere must be an integer between 1 and 6')
        flag = 1
    
    if ((vip['retrieve_lcloud'] == 0) & (vip['prior_lwp_mn'] > 0)):
        print('WARNING: retrieve_lcloud set to 0, but prior_lwp_mn is non-zero!')

    if ((vip['retrieve_icloud'] == 0) & (vip['prior_itau_mn'] > 0)):
        print('WARNING: retrieve_icloud set to 0, but prior_itau_mn is non-zero!')
        
    return flag

################################################################################
# This routine is called when AERIoe aborts
################################################################################

def abort(lbltmpdir, date):

    if os.path.exists(lbltmpdir):
        shutil.rmtree(lbltmpdir)

    print('>>> AERI retrieval on ' + str(date) + ' FAILED and ABORTED <<<')
    print('--------------------------------------------------------------------')
    print(' ')

################################################################################
#   This routine reads in the scattering properties from databases that were
#   created by Mie code (newmie_iteration) and by Ping Yang's computations (that
#   database was compiled by write_database.pro in IceScatProperties), and other
#   similar scattering property databases.  Rectangular matrices are returned,
#   which can then be processed by the routine get_scat_properties.pro to extract
#   out the properties depending on wavelength and effective radius.  Note that
#   there are two Mie databases: one for ice particles and one for water particles.
################################################################################

def read_scat_databases(dbname):

    #See if the database actually exists

    if not os.path.exists(dbname):
        print('ERROR: Unable to find the scattering database')
        return [], 1

    #Number of header lines (before the phase function angles)
    nheader = 5

    #Number of columns of data (not including the phase function)
    ncols = 13

    #Open and read the single scattering property database
    print('Reading: ' + dbname)
    f = open(dbname, 'r')
    f.readline()
    f.readline()
    nlines = int(f.readline().split()[0])
    nphase = int(f.readline().split()[0])

    if nlines <= 0:
        print('ERROR: There were no datalines found in this database -- this should not occur')
        f.close()
        return [], 1

    if nphase <= 0:
        print('ERROR: The scattering phase function was not defined in this database')
        f.close()
        return [], 1
    f.close()

    pangle = np.genfromtxt(dbname ,skip_header = nheader, max_rows = 1)

    data = np.genfromtxt(dbname ,skip_header = nheader+2).T

    fields = (['wavelength [um]','wavenumber [cm-1]','effective radius',
              'extinction cross section', 'scattering cross section',
              'absorption cross section', 'single scatter albedo',
              'asymmetry parameter', 'Extinction efficiency',
              'Absorption efficiency', 'Scattering efficiency', 'Volume',
              'Projected area','Rest of the elements are phase function'])

    database = ({'dbname':dbname, 'ncols':ncols, 'nphase':nphase, 'pangle':pangle,
                'data':data, 'columns':fields})

    return database, 0

################################################################################
# This function reads in the standard atmosphere profiles
################################################################################

def read_stdatmos(filename, stdatmos, verbose):

    if not os.path.exists(filename):
        print('Error: Unable to find the IDL save file with the standard atmosphere information in it')
        return {'status':0}

    temp = scipy.io.readsav(filename, python_dict = True)
    idx = stdatmos-1
    if ((idx < 0) | (idx >= len(temp['name']))):
        print('Error: the standard atmosphere specified is out-of-range')
        return {'status':0}

    if verbose >= 1:
        print('Using standard atmosphere: ' + temp['name'][idx].decode())

    return {'status':1, 'z':temp['z'][idx,:], 'p':temp['p'][idx,:], 't':temp['t'][idx,:], 'w':temp['w'][idx,:], 'pwv':temp['pwv'][idx], 'name':temp['name'][idx]}
