import os
import sys
import numpy as np
import shutil
import scipy.io
import copy
import warnings
from netCDF4 import Dataset
from datetime import datetime
from time import gmtime, strftime
from subprocess import Popen, PIPE
from argparse import ArgumentParser

import Other_functions
import VIP_Databases_functions
import Calcs_Conversions
import Model_Reads
import Jacobian_Functions
import Output_Functions

# Create parser for command line arguments
parser = ArgumentParser()

parser.add_argument("date", type=int, help="Date to run the code [YYYYMMDD]")
parser.add_argument("vip_filename", help="Name if the VIP file (string)")
parser.add_argument("prior_filename", help="Name of the prior input dataset (string)")
parser.add_argument("--shour", type=float, help="Start hour (decimal, 0-24)")
parser.add_argument("--ehour", type=float, help="End hour (decimal, 0-24) [If ehour<0 process up to last AERI sample]")
parser.add_argument("--verbose",type=int, choices=[0,1,2,3], help="The verbosity of the output (0-very quiet, 3-noisy)")
parser.add_argument("--doplot", action="store_true", help="If set, then create real-time display of retrievals")
parser.add_argument("--debug", action="store_true", help="Set this to turn on the debug mode")
parser.add_argument("--dostop",action="store_true", help="Set this to stop at the end before exiting")

args = parser.parse_args()

date = args.date
vip_filename = args.vip_filename
prior_filename = args.prior_filename
shour = args.shour
ehour = args.ehour
verbose = args.verbose
doplot = args.doplot
debug = args.debug
dostop = args.dostop

#Check to see if any of these are set; if not, fall back to default values

if shour is None:
    shour = 0.
if ehour is None:
    ehour = -1.
if verbose is None:
    verbose = 1
if debug is None:
    debug = False
if dostop is None:
    dostop = False

# Initialize
success = True

# We need the background shell to be the C-shell, as we will be spawning out
# a variety of commands that make this assumption. So we will do a
# quick check to find out if the C-shell exists on this system, and if so,
# set the SHELL to this.

if verbose == 3:
    print(' ')
    print(('The current shell is', os.getenv('SHELL')))
else:
    warnings.filterwarnings("ignore", category=UserWarning)

process = Popen('which csh', stdout = PIPE, stderr = PIPE, shell=True)
stdout, stderr = process.communicate()

if stdout.decode() == '':
    print('Error: Unable to find the C-shell command on this system')
    print(('>>> TROPoe retrieval on ' + str(date) + ' FAILED and ABORTED <<<'))
    sys.exit()
else:
    SHELL = stdout[:-1].decode()

if verbose == 3:
    print(('The shell for all commands is', SHELL))

#Capture the version of this file
globatt = {'algorithm_code': 'TROPoe Retrieval Code',
           'algorithm_author': 'Dave Turner, Earth System Research Laboratory / NOAA dave.turner@noaa.gov',
           'algorithm_comment1': 'TROPoe is a physical-iterative algorithm that retrieves thermodynamic profiles from ' +
                                 'a wide range of ground-based remote sensors.  It was primarily designed to use either ' +
                                 'infrared spectrometers or microwave radiometers as the primary instrument, and include ' +
                                 'observations from other sources to improve the quality of the retrieved profiles',
           'algorithm_comment2': 'Original code was written in IDL and is described by the "AERIoe" papers listed below',
           'algorithm_comment3': 'Code was ported to python by Joshua Gebauer with contributions ' +
                                 'from Tyler Bell (both at the University of Oklahoma)',
           'algorithm_reference1': 'DD Turner and U Loehnert, Information Content and ' +
                    'Uncertanties in Thermodynamic Profiles and Liquid Cloud Properties ' +
                    'Retrieved from the Ground-Based Atmospheric Emitted Radiance ' +
                    'Interferometer (AERI), J Appl Met Clim, vol 53, pp 752-771, 2014 ' +
                    'doi:10.1175/JAMC-D-13-0126.1',
           'algorithm_reference2': 'DD Turner and WG Blumberg, Improvements to the AERIoe ' +
                    'thermodynamic profile retrieval algorithm. IEEE Selected Topics ' +
                    'Appl. Earth Obs. Remote Sens., 12, 1339-1354, doi:10.1109/JSTARS.2018.2874968',
           'algorithm_reference3': 'DD Turner and U Loehnert, Ground-based temperature and humidity profiling: ' +
                    'Combining active and passive remote sensors, Atmos. Meas. Tech., vol 14, pp 3033-3048, ' +
                    'doi:10.5194/amt-14-3033-2021', 
           'forward_model_reference1': 'The forward radiative transfer models are from Atmospheric and Environmental ' +
                    'Research Inc (AER); an overview is provided by Clough et al., Atmospheric radiative transfer ' +
                    'modeling: A summary of the AER codes, JQSRT, vol 91, pp 233-244, 2005, doi:10.1016/j.jqsrt.2004.05.058', 
           'forward_model_reference2': 'The infrared model is LBLRTM; papers describing this model include ' + 
                    'doi:10.1029/2018JD029508, doi:10.1175/amsmonographs-d-15-0041.1, and doi:10.1098/rsta.2011.0295', 
           'forward_model_reference3': 'The microwave model is MonoRTM; papers describing this model include ' + 
                    'doi:10.1109/TGRS.2010.2091416 and doi:10.1109/TGRS.2008.2002435', 
           'datafile_created_on_date': strftime("%Y-%m-%d %H:%M:%S", gmtime()),
           'datafile_created_on_machine': os.uname()[-1]}


# Start the retrieval
print(' ')
print('------------------------------------------------------------------------')
print(('>>> Starting TROPoe retrieval for ' + str(date) + ' (from ' + str(shour) + ' to ' + str(ehour) + ' UTC) <<<'))

#Find the VIP file and read it

vip = VIP_Databases_functions.read_vip_file(vip_filename, globatt = globatt, debug = debug, verbose = verbose, dostop = dostop)

if vip['success'] != 1:
    print(('>>> TROPoe retrieval on ' + str(date) + ' FAILED and ABORTED <<<'))
    print('--------------------------------------------------------------------')
    print(' ')
    sys.exit()

process = Popen('echo $$', stdout = PIPE, stderr = PIPE, shell=True, executable = SHELL)
stdout, stderr = process.communicate()

uniquekey = vip['tag'] + '.' + stdout[:-1].decode()

if debug:
    print('DDT: Saving the VIP and globatt structure into "vip.npy" -- for debugging')
    np.save('vip.npy', vip)

# Make sure that Paul van Delst's script "lblrun" is in the $LBL_HOME/bin
# directory, as this is used often. The assumption is that if it is there,
# that the rest of the LBLRTM distribution is set up to use it properly.

if not os.path.exists(vip['lbl_home'] + '/bin/lblrun'):
    print('Error: Unable to find the script "lblrun" in the "lbl_home"/bin directory')
    print('This is a critical component of the LBLRTM configuration - aborting')
    print(('>>> TROPoe retrieval on ' + str(date) + ' FAILED and ABORTED <<<'))
    print('--------------------------------------------------------------------')
    print(' ')
    sys.exit()

# Check if the prior data exists
if not os.path.exists(prior_filename):
    print(('Error: Unable to find the prior data file: ' + prior_filename))
    print(('>>> TROPoe retrieval on ' + str(date) + ' FAILED and ABORTED <<<'))
    print('--------------------------------------------------------------------')
    print(' ')
    sys.exit()

print(('Using the prior file: ' + prior_filename))


# Make sure that the output directory exists
if not os.path.exists(vip['output_path']):
    print('Error: The output directory does not exist')
    print(('>>> TROPoe retrieval on ' + str(date) + ' FAILED and ABORTED <<<'))
    print('--------------------------------------------------------------------')
    print(' ')
    sys.exit()

# Look at the name of the LBLtmpDir; if it starts with a "$"
# then assume that first part is an environment variable and
# decode the path accordingly

if vip['lbl_temp_dir'][0] == '$':
    envpath = vip['lbl_temp_dir'].split('/')
    tmpdir = os.getenv(envpath[0])
    if not tmpdir:
        print('Error: The LBLRTM temporary directory is being set to an environment variable that does not exist')
        print(('>>> TROPoe retrieval on ' + str(date) + ' FAILED and ABORTED <<<'))
        print('--------------------------------------------------------------------')
        print(' ')
        sys.exit()
    for i in range(1,len(envpath)):
        tmpdir = tmpdir + '/' + envpath[i]
    vip['lbl_temp_dir'] = tmpdir

# Create the temporary working directory
lbltmpdir = vip['lbl_temp_dir'] + '/' + uniquekey
print(('Setting the temporary directory for RT model runs to: ' + lbltmpdir))

#Address this in Python 3 version
try:
    os.makedirs(lbltmpdir)
except:
    print('Error making the temporary directory')
    print(('>>> TROPoe retrieval on ' + str(date) + ' FAILED and ABORTED <<<'))
    print('--------------------------------------------------------------------')
    print(' ')

#Now we are ready to start the main retrieval.
notqcov = 0    # I will leave this here for now, but later will add this to the vip file. If "1" it assumes no covariance between T & Q
cvgmult = 0.25 # I will leave this here for now, but later will add this to the vip file. It is a multiplier to apply to the convergence test (0.1 - 1.0)

success = 0

# House keeping stuff
starttime = datetime.now()
endtime = starttime
print(' ')

# Read in the SSP databases
sspl, flag = VIP_Databases_functions.read_scat_databases(vip['lcloud_ssp'])
if flag == 1:
    print('Error: Problem reading SSP file for liquid cloud properties')
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()


sspi, flag = VIP_Databases_functions.read_scat_databases(vip['icloud_ssp'])
if flag == 1:
    print('Error: Problem reading SSP file for ice cloud properties')
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()

# Determine the minimum and maximum Reff values in these, but I need to have the
# minimum value be just a touch larger than the actual minimum value in the database
# hence the 1.01 multipliers

minLReff = np.nanmin(sspl['data'][2,:])*1.01
maxLReff = np.nanmax(sspl['data'][2,:])
miniReff = np.nanmin(sspi['data'][2,:])*1.01
maxiReff = np.nanmax(sspi['data'][2,:])

# Perform some more baseline checking of the keywords
if ((cvgmult  < 0.1) | (cvgmult > 1)):
    print('Error: cvgmult is too small or too large')
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()

# Perform some baseline checking of parameters in the VIP structure
# to make sure that the values are within the valid range
if VIP_Databases_functions.check_vip(vip) == 1:
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()

# Set switches associated with retrieving various variables and computing Jacobians
# This is a mess. Original was terribly inefficient. Maybe look to handle this better in future


if ((vip['retrieve_co2'] >= 1) & (vip['retrieve_co2'] <= 2)):
    doco2 = vip['retrieve_co2']
else:
    doco2 = 0
if ((vip['retrieve_ch4'] >= 1) & (vip['retrieve_ch4'] <= 2)):
    doch4 = vip['retrieve_ch4']
else:
    doch4 = 0
if ((vip['retrieve_n2o'] >= 1) & (vip['retrieve_n2o'] <= 2)):
    don2o = vip['retrieve_n2o']
else:
    don2o = 0
if vip['retrieve_lcloud'] >= 1:
    dolcloud = 1
    fixlcloud = 0           # Jacobian flag (for some reason 0 is on)
else:
    dolcloud = 0
    fixlcloud = 1           # Jacobian flag (for some reason 1 is off)
if vip['retrieve_icloud'] >= 1:
    doicloud = 1
    fixicloud = 0          # Jacobian flag (for some reason 0 is on)
else:
    doicloud = 0
    fixicloud = 1          # Jacobian flag (for some reason 1 is off)
if vip['retrieve_temp'] >= 1:
    dotemp = 1
    fixtemp = 0
else:
    dotemp = 0
    fixtemp = 1
if vip['retrieve_wvmr'] >= 1:
    dowvmr = 1
    fixwvmr = 0
else:
    dowvmr = 0
    fixwvmr = 1

modeflag = [dotemp, dowvmr, dolcloud, dolcloud, doicloud, doicloud, doco2, doch4, don2o]

# Select the LBLRTM version to use
print(' ')
print(('Working with the LBLRTM version ' + vip['lbl_version']))
print(('  in the directory ' + vip['lbl_home']))
print(('  and the TAPE3 file ' + vip['lbl_tape3']))
print(' ')

# Quick check: make sure the LBLRTM path is properly set
if not os.path.exists(vip['lbl_home'] + '/bin/lblrtm'):
    print('Error: lblhome is not properly set')
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()

# Make sure that the specified TAPE3 file exists
if not os.path.exists(vip['lbl_home'] + '/hitran/' + vip['lbl_tape3']):
    print('Error: unable to find the specified TAPE3 file in the LBL_HOME hitran directory')
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()

# Define some paths and constants
lbldir = lbltmpdir + '/lblout'       # Name of the lbl output directory
lbltp5 = lbltmpdir + '/lbltp5'       # Name of the tape5 file
lbltp3 = vip['lbl_tape3']           # Name of the tape3 file
lbllog = lbltmpdir + '/lbllog'       # Name of the lbl log file
lbltmp = lbltmpdir + '/lbltmp'       # Temporary directory for LBLRUN (will be LBL_RUN_ROOT)
monortm_config = 'monortm_config.txt'
monortm_zfreq = 'monortm_zfreqs.txt'    # For MWR-zenith calculations
monortm_sfreq = 'monortm_sfreqs.txt'    # For MWR-scan calculations
monortm_tfile = 'monortm_sonde.cdf'

# Make two commands: one for MWR-zenith and one for MWR-scan
monortm_zexec = ('cd ' + lbltmpdir + ' ; setenv monortm_config ' + monortm_config +
                ' ; setenv monortm_freqs ' + monortm_zfreq + ' ; ' + vip['monortm_wrapper'])

monortm_sexec = ('cd ' + lbltmpdir + ' ; setenv monortm_config ' + monortm_config +
                ' ; setenv monortm_freqs ' + monortm_sfreq + ' ; ' + vip['monortm_wrapper'])

# This should be included in the VIP file. Right now it is always set.
create_monortm_config = 1           # Set this flag to create a custom config file for MonoRTM
create_monortm_zfreq = 1            # Set this flag to create a custom freq-zenith file for MonoRTM
create_monortm_sfreq = 1            # Set this flag to create a custom freq-scan file for MonoRTM

#Load the standard atmosphere

stdatmos = VIP_Databases_functions.read_stdatmos(vip['path_std_atmos'], vip['lbl_std_atmos'], verbose)
if stdatmos['status'] == 0:
    print('Error: Unable to find/read the standard atmosphere file')
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()

# Define the spectral regions(s) to use in the retrieval
bands = vip['spectral_bands']
foo = np.where(bands[0,:] >= 0)[0]
if len(foo) <= 0:
    print('Error: the spectral bands do not have any properly defined values')
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()
bands = bands[:,foo]
    
# Echo to the user the type of retrieval being performed
print(' ')
tmp = 'Retrieving: '
if dotemp == 1:
    tmp = tmp + 'T '
if dowvmr == 1:
    tmp = tmp + 'Q '
if dolcloud == 1:
    tmp = tmp + 'Liq_Cloud '
if doicloud == 1:
    tmp = tmp + 'Ice_Cloud '
if doco2 == 1:
    tmp = tmp + 'CO2 '
if doch4 == 1:
    tmp = tmp + 'CH4 '
if don2o == 1:
    tmp = tmp + 'N2O '
print(tmp)
print(' ')

# Read in the a priori covariance matrix of T/Q for this study
nsonde_prior = -1
try:
    fid = Dataset(prior_filename,'r')
except:
    print('Error: Unable to open the XaSa file')
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()

z = fid.variables['height'][:]
Pa = fid.variables['mean_pressure'][:]
Xa = fid.variables['mean_prior'][:]
Sa = fid.variables['covariance_prior'][:]
try:
    nsonde_prior = int(fid.Nsonde.split()[0])
except AttributeError:
    nsonde_prior = int(fid.Nprofiles.split()[0])

comment_prior = str(fid.Comment)
minT = float(fid.QC_limits_T.split()[5])
maxT = float(fid.QC_limits_T.split()[7])
if verbose == 3:
    print(('QC limits for T are ' + str(minT) + ' and ' + str(maxT)))
minQ = float(fid.QC_limits_q.split()[7])
maxQ = float( fid.QC_limits_q.split()[9])
if verbose == 3:
    print(('QC limits for Q are ' + str(minQ) + ' and ' + str(maxQ)))
fid.close()

if verbose >= 1:
    print(('Retrieved profiles will have ' + str(len(z)) + ' levels (from prior)'))
if verbose >= 2:
    print(('There were ' + str(nsonde_prior) + ' radiosondes used in the calculation of the prior'))

# Inflate the lowest levels of the prior covariance matrix, if desired

Sa, status = Other_functions.inflate_prior_covariance(Sa, z, vip['prior_t_ival'], vip['prior_t_iht'],
             vip['prior_q_ival'], vip['prior_q_iht'], vip['prior_tq_cov_val'],
             vip['prior_chimney_ht'], verbose)

if status == 0:
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()
    
# Read in the data
fail, aeri, mwr, mwrscan = Model_Reads.read_all_data(vip,date)
 
if fail == 1:
    print('Error reading in data: aborting')
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()
 
location = {'lat':aeri['lat'], 'lon':aeri['lon'],'alt':np.round(aeri['alt']*1000)}

if vip['station_alt'] >= 0:
    if verbose >= 2:
        print('Overriding alt with info from VIP file')
    location['alt'] = int(vip['station_alt'])

# Very simple check to make sure station altitude makes sense [m MSL]
if(location['alt'] < 0):
    print('    Error: the station altitude must be > 0 [m MSL]')
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()
    
# This is where external profile reads would go....


# This is where surface data reads would go....

# Do I use a hardcoded value for CO2, or use my simple model to predict
# the concentration? Unit is ppm

if vip['prior_co2_mn'][0] < 0:
   vip['prior_co2_mn'][0] = Other_functions.predict_co2_concentration(int(str(date)[0:4]), int(str(date)[4:6]), int(str(date)[6:8]))

# Quick test to make sure that the trace gas models make sense. If not, abort
tmpco2 = Other_functions.trace_gas_prof(vip['retrieve_co2'], z, vip['prior_co2_mn'])
nfooco2 = len(np.where(tmpco2 < 0)[0])
tmpch4 = Other_functions.trace_gas_prof(vip['retrieve_ch4'], z, vip['prior_ch4_mn'])
nfooch4 = len(np.where(tmpch4 < 0)[0])
tmpn2o = Other_functions.trace_gas_prof(vip['retrieve_n2o'], z, vip['prior_n2o_mn'])
nfoon2o = len(np.where(tmpn2o < 0)[0])
if ((nfooco2 > 0) | (nfooch4 > 0) | (nfoon2o > 0)):
    print('Error: The CO2, CH4, and/or N2O parameters are incorrect giving negative values - aborting')
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()

# Recenter the prior if desired
recenter_input_value = -1
if vip['recenter_prior'] > 0:
    if vip['aeri'] == 1:
        recenter_input_value = np.mean(aeri['wsfc'])
    elif vip['mwr'] == 1:
        recenter_input_value = np.mean(mwr['wsfc'])
    elif vip['mwrscan'] == 1:
        recenter_input_value = np.mean(mwrscan['wsfc'])
            
    # Determine which method to scale the temperature
    if ((vip['recenter_prior'] == 1) | (vip['recenter_prior'] == 3) | (vip['recenter_prior'] == 5)):
        changeTmethod = 0
        if ((vip['recenter_prior'] == 1) | (vip['recenter_prior'] == 5)):
            changeTmethod = 1
        elif vip['recenter_prior'] == 3:
            changeTmethod = 2

    # Quick check to make sure the recenter value is ok before trying to recenter the prior
    if recenter_input_value > 0:

        # Recenter the prior, using the inputs determined above
        successflag, newXa, newSa, comments = Model_Reads.recenter_prior(z, Pa, Xa, Sa, 
                    recenter_input_value, sfc_or_pwv=1, changeTmethod=changeTmethod)

        # Now replace the variables, if successful
        #   and update the global attributes to note that prior recentering was performed
        if successflag == 1:
            Xa = newXa
            Sa = newSa
            globatt.update(comments)
# Splice these trace gases and clouds into the Xa and Sa matrices.
# I am assuming no correlations between the TGs and clouds and the T/Q profiles
# The state vector is going to be defined to be:
# X = [T(z), q(z), LWP, ReL, tauI, ReI, CO2(3), CH4(3), N2O(3)
minsd = 1e-5        # Smallest value that doesn't result in problems in the matrix inversions below
diag = np.diag([np.max([vip['prior_lwp_sd'],minsd])**2,           # LWP : index 0
               np.max([vip['prior_lReff_sd'],minsd])**2,         # liquid Reff : index 1
               np.max([vip['prior_itau_sd'],minsd])**2,          # ice optical depth : index 2
               np.max([vip['prior_iReff_sd'],minsd])**2,         # ice Reff : index 3
               np.max([vip['prior_co2_sd'][0],minsd])**2,        #
               np.max([vip['prior_co2_sd'][1],minsd])**2,        # CO2 : indices 4,5,6
               np.max([vip['prior_co2_sd'][2],minsd])**2,        #
               np.max([vip['prior_ch4_sd'][0],minsd])**2,        #
               np.max([vip['prior_ch4_sd'][1],minsd])**2,        # CH4 : indices 7,8,9
               np.max([vip['prior_ch4_sd'][2],minsd])**2,        #
               np.max([vip['prior_n2o_sd'][0],minsd])**2,        #
               np.max([vip['prior_n2o_sd'][1],minsd])**2,        # N2O : indices 10,11,12
               np.max([vip['prior_n2o_sd'][2],minsd])**2])       #

zero = np.zeros((len(diag[0,:]),len(Xa)))
Sa = np.append(np.append(Sa,zero,axis=0),np.append(zero.T,diag,axis=0),axis=1)
Xa = np.append(Xa, [vip['prior_lwp_mn'], vip['prior_lReff_mn'],
               vip['prior_itau_mn'], vip['prior_iReff_mn'],
               vip['prior_co2_mn'][0], vip['prior_co2_mn'][1], vip['prior_co2_mn'][2],
               vip['prior_ch4_mn'][0], vip['prior_ch4_mn'][1], vip['prior_ch4_mn'][2],
               vip['prior_n2o_mn'][0], vip['prior_n2o_mn'][1], vip['prior_n2o_mn'][2]])
sig_Xa = np.sqrt(np.diag(Sa))

# Put all of the prior information into a structure for later
prior = {'comment':comment_prior, 'filename':prior_filename, 'nsonde':nsonde_prior,
         'Xa':np.copy(Xa), 'Sa':np.copy(Sa)}

# Make a directory for LBL_RUN_ROOT
if os.path.exists(lbltmp):
    shutil.rmtree(lbltmp)

os.mkdir(lbltmp)

# If neither liquid or ice clouds are enabled, then indicate that
# all retrievals are done as clear sky
if ((vip['retrieve_lcloud'] == 0) & (vip['retrieve_icloud'] == 0)):
    if verbose >= 2:
        print('All cloud retrievals disabled -- assuming clear sky')
    Xa[2*len(z)] = 0  # Zero LWP
    Xa[2*len(z)+2] = 0 # Zero ice optical depth
    aeri['cbhflag'][:] = 0
    # Note that I am leaving the CBH values untouched...

# Now loop over the observations and perform the retrievals
xret = []                  #Initialize. Will overwrite this if a succesful ret
read_deltaod = 0           # Initialize.
already_saved = 0          # Flag to say if saved already or not...
fsample = 0                # Counter for number of spectra processed
precompute_prior_jacobian = {'status':0}    # This will allow us to store the forward calculations from the prior, makes code faster

# Read in the AERI prior file to get the typical wavenumbers
f = Dataset(vip['aeri_noise_file'],'r')
awnum = f.variables['wnum'][:]
f.close()

# Quick check to make sure that the spectral bands being selected are actually
# valid for this interferometer (this ensures spectral range matches calculation below)
if(vip['aeri'] >= 1):
    minv = np.min(awnum)
    maxv = np.max(awnum)
    foo = np.where(bands < minv)
    if(len(foo) > 0):
        bands[foo] = minv+0.1
    foo = np.where(bands > maxv)
    if(len(foo) > 0):
        bands[foo] = maxv-0.1

# If clobber == 2, then we will try to append. But this requires that
# I perform a check to make sure that we are appending to a file that was
# created by a version of the code that makes sense. I only need to make this
# test once, hence this flag
if vip['output_clobber'] == 2:
    check_clobber = 1
else:
    check_clobber = 0

# This defines the extra vertical layers that will be added to both
# the infrared and microwave radiative transfer calculations
rt_extra_layers = Other_functions.compute_extra_layers(np.max(z))

version = ''
noutfilename = ''

################################################################################
# This is the main loop for the retrieval!
################################################################################
for i in range(len(aeri['secs'])):                        # { loop_i
    
    # Make sure that is isn't raining
    if aeri['precip'][i] > 0.001:
        print(f"  Sample {i:2d} at {aeri['secs'][i]:.2f} -- raining, no retrieval performed")
        continue
    
    else:
        print(f"  Sample {i:2d} at {aeri['secs'][i]:.2f} UTC is being processed (cbh is {aeri['cbh'][i]:.3f})")
    
    
    if ((vip['station_psfc_min'] > aeri['psfc'][i]) | (aeri['psfc'][i] > vip['station_psfc_max'])):
        print('Error: Surface pressure is not within range set in VIP -- skipping sample')
        continue
    
    # I need a flag for the observations, so I can select the proper forward model.
    # The values are:
    #                1 -- AERI
    #                2 -- MWR zenith data
    #                3 -- external temperature profiler (sonde, lidar, NWP, etc) (*)
    #                4 -- external water vapor profiler (sonde, lidar, NWP, etc) (*)
    #                5 -- surface temperature measurement
    #                6 -- surface water vapor measurement
    #                7 -- external NWP model profile (*)
    #                8 -- external NWP model water vapor profile (*)
    #                9 -- in-situ surface CO2 obs
    #               10 -- MWR non-zenith data
    #               11 -- RASS virtural temperature data
    #               12, ... -- IASI, CrIS, S-HIS, others...
    #     (*) Note that the NWP input could come in two ways, but I wanted a way
    #         to bring in lidar and NWP data, while retaining backwards compatibility
    
    vector_built = False
    if vip['aeri'] == 1:
        wnum = np.copy(aeri['wnum'])
        Y = np.copy(aeri['radmn'][i])
        sigY = np.copy(aeri['noise'][i])
        flagY = np.ones(len(wnum))
        dimY = np.copy(wnum)
        vector_built = True
    
    if vip['mwr'] == 1:
        if vector_built:
            Y = np.append(Y,mwr['bt'][i])
            sigY = np.append(sigY,mwr['noise'][i])
            flagY = np.append(flagY, np.ones(len(mwr['bt'][i]))*2)
            dimY = np.append(dimY, mwr['freq'])
        else:
            Y = np.copy(mwr['bt'][i])
            sigY = np.copy(mwr['noise'][i])
            flagY = np.ones(len(mwr['bt'][i]))*2
            dimY = np.copy(mwr['freq'])
            vector_built = True
    
    if vip['mwrscan'] == 1:
        if vector_built:
            Y = np.append(Y,mwrscan['bt'][i])
            sigY = np.append(sigY,mwrscan['noise'][i])
            flagY = np.append(flagY, np.ones(len(mwrscan['bt'][i]))*3)
            dimY = np.append(dimY,(mwrscan['freq']*1000 + mwrscan['elev']/1000))
        else:
            Y = np.copy(mwrscan['bt'][i])
            sigY = np.copy(mwrscan['noise'][i])
            flagY = np.ones(len(mwrscan['bt'][i]))*3
            dimY = mwrscan['freq']*1000 + mwrscan['elev']/1000
            
    # Add in surface data to obs vector
    
    Y = np.append(Y,aeri['tsfc'][i])
    sigY = np.append(sigY,0.5)
    flagY = np.append(flagY, 5)
    dimY = np.append(dimY,0)
    
    Y = np.append(Y,aeri['wsfc'][i])
    sigY = np.append(sigY,1.5)
    flagY = np.append(flagY, 6)
    dimY = np.append(dimY,0)
    
    # Other data types will be added here
    
    
    nY = len(Y)
    Sy = np.diag(sigY**2)
    
    # Quick check: All of the 1-sigma uncertainties from the observations
    # should have been positive. If not then abort as extra logic needs
    # to be added above...
    foo = np.where((sigY <= 0) & (Y < -900))[0]
    if len(foo) > 0:
        tmp = np.copy(flagY[foo])
        feh = np.unique(tmp)
        if len(feh) <= 0:
            print('This should not happen. Major error in quick check of 1-sigma uncertainties')
            VIP_Databases_functions.abort(lbltmpdir,date)
            sys.exit()
        else:
            print(('Warning: There were missing values in these obs: ' + str(feh)))
        sigY[foo] *= -1           # Presumably, the missing values had -999 for their uncertainties
            
    
    foo = np.where((sigY <= 0) & (Y > -900))[0]
    if len(foo) > 0:
        tmp = np.copy(flagY[foo])
        feh = np.unique(tmp)
        if len(feh) <= 0:
            print('This should not happen. Major error in quick check of 1-sigma uncertainties')
            VIP_Databases_functions.abort(lbltmpdir,date)
            sys.exit()
        else:
            print(('Error: There were negative 1-sigma uncertainties in obs: ' + str(feh) + ' Skipping sample'))
            continue
    
    # Compute the estimate of the forward model uncertainty (Sf).
    # This is computed at Sf = kb # Sb # transpose(Kb), where
    # B is the vector of model parameters and Sb is its covariance matrix
    # Note: right now, I will assume a perfect forward model

    Sf = np.diag(np.zeros(len(np.diag(Sy))))          # This is a matrix of zeros

    # Build the observational covariance matrix, which is a combination of
    # the uncertainties in the forward model and the observations
    
    Sm = Sy + Sf
    SmInv = np.linalg.pinv(Sm)
    
    # Get the other input variables that the forward model will need
    nX = len(z)*2                 # For both T and Q
    
    # Start building the first guess vector
    #    T & Q, LWP, ReL, TauI, ReI, co2(3), ch4(3), n2o(3)
    X0 = np.copy(Xa)     # Start with the prior, and overwrite portions of it if desired
    first_guess = 'prior'
    if vip['first_guess'] == 1:
        # Use the prior as the first guess
        if verbose >= 3:
            print('Using prior as first guess')
    elif vip['first_guess'] == 2:
        # Build a first guess from the AERI-estimated surface temperture,
        # an assumed lapse rate, and a 60% RH as first guess
        if verbose >= 3:
            print('Using Tsfc with lapse rate and 60& RH as first guess')
        first_guess = 'Tsfc with lapse rate and 60% RH'
        lapserate = -7.0        # C / km
        constRH = 60.           # percent RH
        t = aeri['tsfc'][i] + z*lapserate
        p = Calcs_Conversions.inv_hypsometric(z, t+273.16, aeri['atmos_pres'][i])  # [mb]
        q = Calcs_Conversions.rh2w(t, np.ones(len(z))*constRH/100., p)
        X0 = np.concatenate([t, q, Xa[nX:nX+12]])    # T, Q, LWP, ReL, TauI, ReI, co2(3), ch4(3), n2o(3)
    elif vip['first_guess'] == 3:
        print('Sorry, first_guess = 3 is bugged in real TROPoe so not included here')
        VIP_Databases_functions.abort(lbltmpdir,date)
        sys.exit()
    
    else:
        print('Error: Undefined first guess option')
        VIP_Databases_functions.abort(lbltmpdir,date)
        sys.exit()
    
    # Build the first guess vector
    itern = 0
    converged = 0
    Xn = np.copy(X0)
    Fxnm1 = np.array([-999.])
    
    # If we are to append to the file, then I need to find the last valid
    # sample in the file, so I only process after that point...
    if ((vip['output_clobber'] == 2) & (check_clobber == 1)):
        xret, fsample, noutfilename = Output_Functions.create_xret(xret, fsample, vip, aeri, Xa, Sa, z, bands, dimY, flagY)
        check_clobber = 0
        if fsample < 0:
            VIP_Databases_functions.abort(lbltmpdir,date)
            sys.exit()
        if fsample == 0:
            vip['output_clobber'] = 0
            xret = []
        if ((verbose >= 1) & (fsample > 0)):
            print(('Will append output to the file ' + noutfilename))
    
    # If we are in 'append' mode, then skip any AERi samples that are
    # before the last time in the xret structure. Generally, the current
    # AERI sample will always be before the last one in the xret structure,
    # except in the cases where we just started the code in append mode. If
    # that happens, then the xret structure will be (partially) populated
    # by the create_xret routine above, which gets the time from the
    # existing netCDF file. And the way we are writing the data that is all
    # that needed to be retrieved...

    if vip['output_clobber'] == 2:
        if aeri['secs'][i] <= xret[fsample-1]['secs']:
            print('  ....but was already processed (append mode)')
            continue
        
    cbh = aeri['cbh'][i]
    cbhflag = aeri['cbhflag'][i]
    
    # Define the gamma factors needed to keep the retrieval sane
    # MWR-only retrievals are more linear and thus the gfactor can be more agressive

    if vip['aeri'] == 0:
        gfactor = np.array([100.,10.,3.,1.])
    else:
        gfactor = np.array([1000.,300.,100.,30.,10.,3.,1.])
    if len(gfactor) < vip['max_iterations']:
        gfactor = np.append(gfactor, np.ones(vip['max_iterations']-len(gfactor)+3))
    
    # Select nice round numbers to use as the wavenumber limits for
    # the LBLRTM calc, but remember that I need to pad by 50 cm-1 for FSCAN
    if vip['aeri'] == 1:
        lblwnum1 = int((np.min(wnum)-60)/100) * 100
        lblwnum2 = (int((np.max(wnum)+60)/100)+1)*100
    continue_next_sample = 0          # A flag used to watch for bad jacobian calcs
    
    while ((itern <= vip['max_iterations']) & (converged == 0)):        # { While loop over iter

        if verbose >= 3:
            print((' Making the forward calculation for iteration ' + str(itern)))

        if os.path.exists(lbltp5):
            shutil.rmtree(lbltp5)

        if os.path.exists(lbldir):
            shutil.rmtree(lbldir)

        if os.path.exists(lbllog):
            shutil.rmtree(lbllog)

        # Update the pressure profile using the current estimate of temperature
        p = Calcs_Conversions.inv_hypsometric(z, Xn[0:int(nX/2)]+273.16, aeri['psfc'][i])
        
        # If the trace gas profile shape is mandated to be a function of the PBL height,
        # then set that here. First, compute the current estimate of the PBL height,
        # then the coefficient and overwrite any current shape coefficient

        if itern == 0:
            pblh = Other_functions.compute_pblh(z, Xn[0:int(nX/2)], p, np.sqrt(np.diag(Sa[0:int(nX/2), 0:int(nX/2)])),
                                   minht=vip['min_PBL_height'], maxht=vip['max_PBL_height'], nudge=vip['nudge_PBL_height'])
        else:
            pblh = Other_functions.compute_pblh(z, Xn[0:int(nX / 2)], p, np.sqrt(np.diag(Sop[0:int(nX/2), 0:int(nX/2)])),
                                   minht=vip['min_PBL_height'], maxht=vip['max_PBL_height'], nudge=vip['nudge_PBL_height'])
            
        coef = Other_functions.get_a2_pblh(pblh)           # Get the shape coef for this PBL height
        if ((vip['retrieve_co2'] == 1) & (vip['fix_co2_shape'] == 1)):
            Xn[nX+4+2] = coef
        if ((vip['retrieve_ch4'] == 1) & (vip['fix_ch4_shape'] == 1)):
            Xn[nX+4+3+2] = coef
        if ((vip['retrieve_n2o'] == 1) & (vip['fix_n2o_shape'] == 1)):
            Xn[nX+4+6+2] = coef

        # Compute its inverse of the prior  
        SaInv = np.linalg.pinv(Sa)
        built_FXn = False
        # This function makes the forward calculation and computes the Jacobian
        # for the AERI component of the forward model
        foo = np.where(flagY == 1)[0]
        
        if len(foo) > 0:
            if((precompute_prior_jacobian['status'] == 1) & (itern == 0)):
                       # Load the forward calculation stuff from the precompute prior data
                   if(verbose >= 1):
                       print('    Preloading forward calculation and jacobian from prior structure')
                   FXn   = precompute_prior_jacobian['FX0']
                   Kij   = precompute_prior_jacobian['Kij0']
                   flag  = precompute_prior_jacobian['flag0']
                   wnumc = precompute_prior_jacobian['wnumc0']
            else:
                # Otherwise, run the forward model and compute the Jacobian
                # Height is currently fixed to sea-level
                flag, Kij, FXn, wnumc, version_compute_jacobian, totaltime  = \
                       Jacobian_Functions.compute_jacobian_interpol(Xn, p, z,
                       vip['lbl_home'], lbldir, lbltmp, vip['lbl_std_atmos'], lbltp5, lbltp3,
                       cbh, sspl, sspi, lblwnum1, lblwnum2,
                       fixtemp, fixwvmr, doco2, doch4, don2o, fixlcloud, fixicloud,
                       vip['fix_co2_shape'], vip['fix_ch4_shape'], vip['fix_n2o_shape'],
                       vip['jac_max_ht'], awnum, vip['lblrtm_forward_threshold'],
                       location['alt'], rt_extra_layers, stdatmos, vip['lblrtm_jac_interpol_npts_wnum'], 
                       verbose, debug, doapodize=True)
                       
            if(precompute_prior_jacobian['status'] == 0):
                precompute_prior_jacobian = {'status':1, 'X0':np.copy(Xn), 'FX0':np.copy(FXn), 'Kij0':np.copy(Kij), 
                'flag0':np.copy(flag), 'wnumc0':np.copy(wnumc)}
        
            # If the Jacobian did not compute properly (i.e., an error occurred)
            # then we need to abort
            if flag == 0:
                print(' -- Skipping this sample due to issue with LBLRTM Jacobian (likely bad input profile)')
                continue_next_sample = 1
                break
        
            # Select the wavenumber indices to use
            w1idx, junk = Other_functions.find_wnum_idx(wnumc, bands)
            if len(w1idx) != len(wnum):
                print('Problem with wnum indices1')
                VIP_Databases_functions.abort(lbltmpdir,date)
                sys.exit()
            wnumc = wnumc[w1idx]
            FXn = FXn[w1idx]
            Kij = Kij[w1idx,:]
        
            # Are there missing values from the AERI? If so,then we want to make the
            # forward model calculation have the same value and put no sensitivity
            # in the Jacobian there so that the retrieval is unaffected (this is really for
            # aeri_type = -1)

            foo = np.where(flagY == 1)[0]
            bar = np.where(Y[foo] < -900)[0]
            if len(bar) > 0:
                FXn[bar] = np.copy(Y[foo[bar]])
                for gg in range(len(bar)):
                    Kij[bar[gg],:] = 0.
            
            built_FXn = True
        # Now start processing the other observation types that might be in the obs vector

        # Perform the forward model calculation and compute the Jacobian for the
        # MWR-zenith portion of the observation vector
        foo = np.where(flagY == 2)[0]
        if len(foo) > 0:
            if create_monortm_config == 1:
                # Create the MonoRTM configuration file
                lun = open(lbltmpdir + '/' + monortm_config, 'w')
                lun.write(vip['monortm_exec'] + '\n')
                lun.write(vip['monortm_spec'] + '\n')
                lun.write('0\n')          # The verbose flag
                lun.write('{:0d}\n'.format(vip['lbl_std_atmos']))
                lun.write('1\n')          # The 'output layer optical depths' flag
                for gg in range(6):       # The 6 continuum multipliers
                    lun.write('1.0\n')
                lun.write('{:7.3f}\n'.format(np.max(z)-0.01))
                lun.write('{:0d}\n'.format(len(z)+len(rt_extra_layers)))
                for gg in range(len(z)):
                    lun.write('{:7.3f}\n'.format(z[gg]))
                for gg in range(len(rt_extra_layers)):
                    lun.write('{:7.3f}\n'.format(rt_extra_layers[gg]))
                lun.close()

                # Turn the flag off, as we only need to create these files once
                create_monortm_config = 0



            if create_monortm_zfreq == 1:
                # Create the MonoRTM freuency file
                lun = open(lbltmpdir + '/' + monortm_zfreq, 'w')
                lun.write('\n')
                lun.write('{:0d}\n'.format(len(mwr['freq'])))
                for gg in range(len(mwr['freq'])):
                    lun.write('{:7.3f}\n'.format(mwr['freq'][gg]))
                lun.close()

                # Turn the flag off, as we only need to create these files once
                create_monortm_zfreq = 0
            
            # Run the forward model and compute the Jacobian
            if vip['monortm_jac_option'] == 1:
                flag, KK, FF, m_comp_time = Jacobian_Functions.compute_jacobian_microwave_finitediff(Xn, p, z,
                        mwr['freq'], cbh, vip, lbltmpdir, monortm_tfile, monortm_zexec,
                        fixtemp, fixwvmr, fixlcloud, vip['jac_max_ht'], stdatmos, location['alt'], verbose)

            elif vip['monortm_jac_option'] == 2:
                flag, KK, FF, m_comp_time = Jacobian_Functions.compute_jacobian_microwave_3method(Xn, p, z,
                        mwr['freq'], cbh, vip, lbltmpdir, monortm_tfile, monortm_zexec,
                        fixtemp, fixwvmr, fixlcloud, vip['jac_max_ht'], stdatmos, location['alt'], verbose)

            else:
                print('Error: Undefined option for monortm_jac_option')
                VIP_Databases_functions.abort(lbltmpdir,date)
                sys.exit()
            
            # Now the size of the forward calculation should be the correct size to match
            # the number of MWR observations in the Y vector
            if len(foo) != len(FF):
                print('Problem computing the Jacobian for the microwave radiometer')
                VIP_Databases_functions.abort(lbltmpdir,date)
                sys.exit()
            
            # Are there misssing values from the MWR? If so, then we want to make
            # the forward model calculation have the same value and put no sensitivity
            # in the Jacobian there so that the retrieval is unaffected

            bar = np.where(Y[foo] < -900)[0]
            if len(bar) > 0:
                FF[bar] = Y[foo[bar]]
                KK[bar,:] = 0.
            
            if built_FXn:
                Kij = np.append(Kij, KK, axis = 0)
                FXn = np.append(FXn,FF)
            else:
                Kij = np.copy(KK)
                FXn = np.copy(FF)
                
        # Perform the forward model calculation and compute the Jacobian for the
        # MWR-zenith portion of the observation vector
        foo = np.where(flagY == 3)[0]
        if len(foo) > 0:
            if create_monortm_config == 1:
                # Create the MonoRTM configuration file
                lun = open(lbltmpdir + '/' + monortm_config, 'w')
                lun.write(vip['monortm_exec'] + '\n')
                lun.write(vip['monortm_spec'] + '\n')
                lun.write('0\n')          # The verbose flag
                lun.write('{:0d}\n'.format(vip['lbl_std_atmos']))
                lun.write('1\n')          # The 'output layer optical depths' flag
                for gg in range(6):       # The 6 continuum multipliers
                    lun.write('1.0\n')
                lun.write('{:7.3f}\n'.format(np.max(z)-0.01))
                lun.write('{:0d}\n'.format(len(z)+len(rt_extra_layers)))
                for gg in range(len(z)):
                    lun.write('{:7.3f}\n'.format(z[gg]))
                for gg in range(len(rt_extra_layers)):
                    lun.write('{:7.3f}\n'.format(rt_extra_layers[gg]))
                lun.close()

                # Turn the flag off, as we only need to create these files once
                create_monortm_config = 0



            if create_monortm_zfreq == 1:
                # Create the MonoRTM freuency file
                lun = open(lbltmpdir + '/' + monortm_zfreq, 'w')
                lun.write('\n')
                tmp = np.unique(mwrscan['freq'])
                print(tmp)
                lun.write('{:0d}\n'.format(len(tmp)))
                for gg in range(len(tmp)):
                    lun.write('{:7.3f}\n'.format(tmp[gg]))
                lun.close()

                # Turn the flag off, as we only need to create these files once
                create_monortm_zfreq = 0
            
            # Run the forward model and compute the Jacobian
            flag, KK, FF, m_comp_time = Jacobian_Functions.compute_jacobian_microwavescan_3method(Xn, p, z,
                    mwrscan, cbh, vip, lbltmpdir, monortm_tfile, monortm_zexec,
                    fixtemp, fixwvmr, fixlcloud, vip['jac_max_ht'], stdatmos, location['alt'], verbose)
            
            if flag == 0:
                print(' -- Skipping this sample due to issure with MonoRTM Jacobian (likely bad input profile)')
                continue_next_sample = 1
                break

            # Now the size fo the forward calculation should be the correct size to match the
            # the number of MWR-scan observation in the Y vector
            if len(foo) != len(FF):
                print('Problem computing the Jacobian for the microwave radiometer scan')
                VIP_Databases_functions.abort(lbltmpdir,date)
                sys.exit()

            # Are there misssing values from the MWR? If so, then we want to make
            # the forward model calculation have the same value and put no sensitivity
            # in the Jacobian there so that the retrieval is unaffected

            bar = np.where(Y[foo] < -900)[0]
            if len(bar) > 0:
                FF[bar] = Y[foo[bar]]
                KK[bar,:] = 0.
            
            if built_FXn:
                Kij = np.append(Kij, KK, axis = 0)
                FXn = np.append(FXn,FF)
            else:
                Kij = np.copy(KK)
                FXn = np.copy(FF)
                
            
        # Perform the forward model calculation and compute the Jacobian for the
        # external water vapor profiler portion ofthe observation vector
        foo = np.where((flagY == 5) | (flagY == 6))[0]
        if len(foo) > 0:
            
            units = np.array(['C','g/kg'])
            

            flag, KK, FF = Jacobian_Functions.compute_jacobian_external_sfc_met(Xn, p, z,
                             0, units, vip['prior_chimney_ht'])

            if flag == 0:
                print('Problem computing the Jacobian for the external surface met data. Have to abort')
                VIP_Databases_functions.abort(lbltmpdir,date)
                sys.exit()

            if len(foo) != len(FF):
                print('Problem computing the Jacobian for the external surface met')
                VIP_Databases_functions.abort(lbltmpdir,date)
                sys.exit()

            # Are there misssing values in the surface met data? If so, then we want to make
            # the forward model calculation have the same value and put no sensitivity
            # in the Jacobian there so that the retrieval is unaffected

            bar = np.where(Y[foo] < -900)[0]
            if len(bar) > 0:
                FF[bar] = Y[foo[bar]]
                KK[bar,:] = 0.
            Kij = np.append(Kij, KK, axis = 0)
            FXn = np.append(FXn,FF)
            
        # Other forward model stuff would go here
        
        ########
        # Done computing forward calculation and Jacobians. Now the retrieval math
        ########

        # Compute the L-curve values to determine an appropriate point for gamma
        use_L_curve = 0
        if use_L_curve > 0:
            # Starting here, I am following the Carissimo et al. logic hete
            if (itern == 0) & verbose >= 2:
                print('Using the L-curve method to optimize gamma')
            ggamma = np.arange(100)*10/99 - 5     # Values from -5 to +5
            ggamma = 10.**ggamma        # this is the range of values I want: 10^(-5) to 10^(+5)

            gfac = Other_functions.lcurve(ggamma, flagY, Y, FXn, Kij, Xn, Xa, Sa, Sm, z)
        else:
            gfac = gfactor[itern]
            
        
        B      = (gfac * SaInv) + Kij.T.dot(SmInv).dot(Kij)
        Binv   = np.linalg.pinv(B)
        Gain   = Binv.dot(Kij.T).dot(SmInv)
        Xnp1   = Xa[:,None] + Gain.dot(Y[:,None] - FXn[:,None] + Kij.dot((Xn-Xa)[:,None]))
        Sop    = Binv.dot(gfac*gfac*SaInv + Kij.T.dot(SmInv).dot(Kij)).dot(Binv)
        SopInv = np.linalg.pinv(Sop)
        Akern  = (Binv.dot(Kij.T).dot(SmInv).dot(Kij)).T
        
        # If we are trying to fix the shape of the TG profiles as a function of the
        # PBLH, then we need to make a special tweak here. The gain matrix for the
        # factor(s) will be zero, which would make the next iteration have the shape
        # factor in the prior. But I don't want to be changing the prior with each iteration,
        # as that will impact the "append" option (if we are using that). So we need this
        # stub of code to do the same thing

        if ((vip['retrieve_co2'] == 1) & (vip['fix_co2_shape'] == 1)):
            Xnp1[nX+4+2] = np.copy(Xn[nX+4+2])
        if ((vip['retrieve_ch4'] == 1) & (vip['fix_ch4_shape'] == 1)):
            Xnp1[nX+4+5] = np.copy(Xn[nX+4+5])
        if ((vip['retrieve_n2o'] == 1) & (vip['fix_n2o_shape'] == 1)):
            Xnp1[nX+4+8] = np.copy(Xn[nX+4+8])

        # Look for NaN values in this updated state vector. They should not
        # exist, but if they do, then let's stop the code here to allow
        # me to look at it. Not optimal solution for operation code
        # though, as it really should output a flagged result or abort/
        foo = np.where(np.isnan(Xnp1))[0]          # DDT
        if len(foo) > 0:
            print('Stopping for NaN issue 1')
            VIP_Databases_functions.abort(lbltmpdir,date)
            sys.exit()
        
        # Compute some information content numbers. The DFS will be computed
        # as the [total, temp, WVMR, LWP, ReffL, TauI, ReffI, co2, ch4, n2o]
        tmp = np.diag(Akern)
        dfs = np.array([np.sum(tmp), np.sum(tmp[0:int(nX/2)]), np.sum(tmp[int(nX/2):nX]), tmp[nX],
                    tmp[nX+1], tmp[nX+2], tmp[nX+3], tmp[nX+4], tmp[nX+5], tmp[nX+6],
                    tmp[nX+7], tmp[nX+8], tmp[nX+9], tmp[nX+10], tmp[nX+11], tmp[nX+12]])

        sic = 0.5 * np.log(scipy.linalg.det(Sa.dot(SopInv)))

        vres,cdfs = Other_functions.compute_vres_from_akern(Akern, z, do_cdfs=True)
        # Compute the N-form and M-form convergence criteria (X and Y spaces, resp)
        if itern == 0:
        # Set the initial RMS and di2 values to large numbers

            old_rmsa = 1e20          # RMS for all observations
            old_rmsr = 1e20          # RMS for only the AERI and MWR radiance obs
            old_di2m = 1e20          # di-squared number
        
        di2n = ((Xn[:,None]-Xnp1).T.dot(SopInv).dot(Xn[:,None]-Xnp1))[0,0]
        if len(Fxnm1) == nY:
            di2m = ((FXn[:,None] - Fxnm1[:,None]).T.dot(
                scipy.linalg.pinv2(Kij.dot(Sop).dot(Kij.T)+Sm)).dot(
                FXn[:,None] - Fxnm1[:,None]))[0,0]
        else:
            di2m = 9.0e9

        # Perform the RH_limit test (i.e., make sure thew WVMR Is not too large
        # such that RH > 100%)
        if ((itern == 0) & (verbose >= 3)):
            print('Testing for RH > 100%')
        rh = Calcs_Conversions.w2rh(np.squeeze(Xnp1[int(nX/2):nX]), p, np.squeeze(Xnp1[0:int(nX/2)]),0) * 100   # units are %RH
        feh = np.where(rh > 100)[0]
        if len(feh) > 0:
            if verbose >= 3:
                print('RH is above 100% somewhere in this profile -- setting it to 100%')
            rh[feh] = 100.
            Xnp1[int(nX/2):nX,0] = Calcs_Conversions.rh2w(np.squeeze(Xnp1[0:int(nX/2)]), rh/100., p)

        # Perform the monotonically ascending potential temperature test (i.e
        # make sure that theta never decreases with height)
        if ((itern == 0) & (verbose >= 3)):
            print('Testing for decreasing theata with height')

        # Multiply WVMR by zero to get theta, not theta-v
        theta = Calcs_Conversions.t2theta(np.squeeze(Xnp1[0:int(nX/2)]), 0*np.squeeze(Xnp1[int(nX/2):nX]), p)

        # This creates the maximum theta
        for ii in range(len(theta))[1:]:
            if ((theta[ii] <= theta[ii-1]) & (z[ii] > vip['superadiabatic_maxht'])):
                theta[ii] = theta[ii-1]

        # Multiply WVMR by zero to work with theta, not theta-v
        Xnp1[0:int(nX/2),0] = Calcs_Conversions.theta2t(theta, 0*np.squeeze(Xnp1[int(nX/2):nX]), p)

        # Make sure we don't get any nonphysical values here that would
        # make the next iteration of the LBLRTM croak
        multiplier = 5.
        
        feh = np.arange(int(nX/2)) + int(nX/2)
        foo = np.where((Xnp1[feh,0] < minQ) | (Xnp1[feh,0] < Xa[feh] - multiplier*np.sqrt((np.diag(Sa)[feh]))))[0]

        if len(foo) > 0:
            # First check to make sure the entire profile isn't nonphysical
            if len(foo) == len(z):
                print('The entire water vapor profile is non-physical. Major error in TROPoe, must abort')
                VIP_Databases_functions.abort(lbltmpdir,date)
                sys.exit()
            
            # A nonphysical water vapor value exists so we are going interpolate across those values
            # by calling this function
            Xnp1[feh,0] = Other_functions.fix_nonphysical_wv(Xnp1[feh,0],z,foo)

        foo = np.where((Xnp1[feh,0] > maxQ) | (Xnp1[feh,0] > Xa[feh] + multiplier*np.sqrt((np.diag(Sa)[feh]))))[0]

        if len(foo) > 0:
            # First check to make sure the entire profile isn't nonphysical
            if len(foo) == len(z):
                print('The entire water vapor profile is non-physical. Major error in TROPoe, must abort')
                VIP_Databases_functions.abort(lbltmpdir,date)
                sys.exit()
            
            # A nonphysical water vapor value exists so we are going interpolate across those values
            # by calling this function
            Xnp1[feh,0] = Other_functions.fix_nonphysical_wv(Xnp1[feh,0],z,foo)

        if dolcloud == 1:
            Xnp1[nX,0] = np.nanmax([Xnp1[nX],0])
            Xnp1[nX+1,0] = np.nanmax([Xnp1[nX+1], vip['prior_lReff_mn']-multiplier*vip['prior_lReff_sd'], minLReff])
            Xnp1[nX+1,0] = np.nanmin([Xnp1[nX+1], vip['prior_lReff_mn']+multiplier*vip['prior_lReff_sd'], maxLReff-2])

        if doicloud == 1:
            Xnp1[nX+2,0] = np.nanmax([Xnp1[nX+2],0])
            Xnp1[nX+3,0] = np.nanmax([Xnp1[nX+3], vip['prior_iReff_mn']-multiplier*vip['prior_iReff_sd'], miniReff])
            Xnp1[nX+3,0] = np.nanmin([Xnp1[nX+3], vip['prior_iReff_mn']+multiplier*vip['prior_iReff_sd'], maxiReff-2])

        if doco2 > 0:
            Xnp1[nX+4,0] = np.nanmax([Xnp1[nX+4], vip['prior_co2_mn'][0]-multiplier*vip['prior_co2_sd'][0], 0])
            Xnp1[nX+4,0] = np.nanmin([Xnp1[nX+4], vip['prior_co2_mn'][0]+multiplier*vip['prior_co2_sd'][0]])

            Xnp1[nX+5,0] = np.nanmax([Xnp1[nX+5], vip['prior_co2_mn'][1]-multiplier*vip['prior_co2_sd'][1]])
            Xnp1[nX+5,0] = np.nanmin([Xnp1[nX+5], vip['prior_co2_mn'][1]+multiplier*vip['prior_co2_sd'][1]])

            if Xnp1[nX+4] + Xnp1[nX+5]  < 0:
                Xnp1[nX+5] = -Xnp1[nX+4]

            if doco2 == 1:
                Xnp1[nX+6,0] = np.nanmax([Xnp1[nX+6], vip['prior_co2_mn'][2] - multiplier*vip['prior_co2_sd'][2], -20])
                Xnp1[nX+6,0] = np.nanmax([Xnp1[nX+6], vip['prior_co2_mn'][2] + multiplier*vip['prior_co2_sd'][2], -1])
            else:
                if Xnp1[nX+6] < vip['min_PBL_height']:
                    Xnp1[nX+6,0] = vip['min_PBL_height']
                if Xnp1[nX+6] > vip['max_PBL_height']:
                    Xnp1[nX+6,0] = vip['max_PBL_height']

        if doch4 > 0:
            Xnp1[nX+7,0] = np.nanmax([Xnp1[nX+7], vip['prior_ch4_mn'][0] - multiplier*vip['prior_ch4_sd'][0], 0])
            Xnp1[nX+7,0] = np.nanmin([Xnp1[nX+7], vip['prior_ch4_mn'][0] + multiplier*vip['prior_ch4_sd'][0]])

            Xnp1[nX+8,0] = np.nanmax([Xnp1[nX+8], vip['prior_ch4_mn'][1]-multiplier*vip['prior_ch4_sd'][1]])
            Xnp1[nX+8,0] = np.nanmin([Xnp1[nX+8], vip['prior_ch4_mn'][1]+multiplier*vip['prior_ch4_sd'][1]])

            if Xnp1[nX+7] + Xnp1[nX+8]  < 0:
                Xnp1[nX+8,0] = -Xnp1[nX+7]

            if doch4 == 1:
                Xnp1[nX+9,0] = np.nanmax([Xnp1[nX+9], vip['prior_ch4_mn'][2] - multiplier*vip['prior_ch4_sd'][2], -20])
                Xnp1[nX+9,0] = np.nanmax([Xnp1[nX+9], vip['prior_ch4_mn'][2] + multiplier*vip['prior_ch4_sd'][2], -1])

            else:
                if Xnp1[nX+9] < vip['min_PBL_height']:
                    Xnp1[nX+9,0] = vip['min_PBL_height']
                if Xnp1[nX+9] > vip['max_PBL_height']:
                    Xnp1[nX+9,0] = vip['max_PBL_height']

        if don2o > 0:
            Xnp1[nX+10,0] = np.nanmax([Xnp1[nX+10], vip['prior_n2o_mn'][0] - multiplier*vip['prior_n2o_sd'][0], 0])
            Xnp1[nX+10,0] = np.nanmin([Xnp1[nX+10], vip['prior_n2o_mn'][0] + multiplier*vip['prior_n2o_sd'][0]])

            Xnp1[nX+11,0] = np.nanmax([Xnp1[nX+11], vip['prior_n2o_mn'][1]-multiplier*vip['prior_n2o_sd'][1]])
            Xnp1[nX+11,0] = np.nanmin([Xnp1[nX+11], vip['prior_n2o_mn'][1]+multiplier*vip['prior_n2o_sd'][1]])

            if Xnp1[nX+10] + Xnp1[nX+11]  < 0:
                Xnp1[nX+11,0] = -Xnp1[nX+10]

            if don2o == 1:
                Xnp1[nX+12,0] = np.nanmax([Xnp1[nX+12], vip['prior_n2o_mn'][2] - multiplier*vip['prior_n2o_sd'][2], -20])
                Xnp1[nX+12,0] = np.nanmax([Xnp1[nX+12], vip['prior_n2o_mn'][2] + multiplier*vip['prior_n2o_sd'][2], -1])

            else:
                if Xnp1[nX+12] < vip['min_PBL_height']:
                    Xnp1[nX+12] = vip['min_PBL_height']
                if Xnp1[nX+12,0] > vip['max_PBL_height']:
                    Xnp1[nX+12,0] = vip['max_PBL_height']
            
        
        # Compute the RMS difference between the observation and the
        # forward calculation. However, this will be the relative RMS
        # difference (normalizing by the observation error here), because I
        # am mixing units from all of the different types of observation
        # But I will also compute the chi-square value of the obs vs. F(Xn)

        chi2 = np.sqrt(np.sum(((Y - FXn)/ Y)**2) / float(nY))
        rmsa = np.sqrt(np.sum(((Y - FXn)/sigY)**2) / float(nY))
        feh = np.where((flagY == 1) | (flagY == 2) & (Y > -900))[0]
        if len(feh) > 0:
            rmsr = np.sqrt(np.sum(((Y[feh] - FXn[feh])/sigY[feh])**2) / float(len(feh)))
        else:
            rmsr = -999.

        # I decided to just change the metric to look at the normalized
        # distance to the climatological prior, but I will let it have either
        # positive or negative values. ONly compute this for the Tq part though

        feh = np.arange(nX)
        rmsp = np.mean( (Xa[feh] - Xn[feh])/sig_Xa[feh] )
        
        # Capture the iteration with the best RMS value
        if rmsa <= old_rmsa:
            old_rmsa = rmsa
            old_rmsr = rmsr
            old_iter = itern
        
        # Check for NaNs in the next iteration. If they exist, then
        # use the last valid sample as the solution and exit.
        foo = np.where(np.isnan(Xnp1))[0]
        if len(foo) > 0:
            print('Warning: Found NaNs in the next iteration -- using last iter')
            if itern == 0:
                print('Wow -- I never thought this could happen')
                VIP_Databases_functions.abort(lbltmpdir,date)
                sys.exit()

            converged = 9                             # Converged in "bad NaN sense
            Xn = np.copy(xsamp[itern-1]['Xn'])
            FXn = np.copy(xsamp[itern-1]['FXn'])
            Sop = np.copy(xsamp[itern-1]['Sop'])
            K = np.copy(xsamp[itern-1]['K'])
            Gain = np.copy(xsamp[itern-1]['Gain'])
            Akern = np.copy(xsamp[itern-1]['Akern'])
            vres = np.copy(xsamp[itern-1]['vres'])
            gfac = xsamp[itern-1]['gamma']
            sic = xsamp[itern-1]['sic']
            dfs = np.copy(xsamp[itern-1]['dfs'])
            cdfs = np.copy(xsamp[itern-1]['cdfs'])
            di2m = xsamp[itern-1]['di2m']
            rmsa = xsamp[itern-1]['rmsa']
            rmsr = xsamp[itern-1]['rmsr']
            rmsp = xsamp[itern-1]['rmsp']
            chi2 = xsamp[itern-1]['chi2']
            itern = -1
        
        elif itern > 1:
            # Test for "convergence by looking at the best RMS value
            if ((rmsa > np.sqrt(gfactor[old_iter])*old_rmsa) & (old_iter > 0)):
            #if ((rmsa > np.sqrt(1)*old_rmsa) & (old_iter > 0)):
                converged = 2                   # Converged in "rms increased drastically" sense

                Xn = np.copy(xsamp[old_iter]['Xn'])
                FXn = np.copy(xsamp[old_iter]['FXn'])
                Sop = np.copy(xsamp[old_iter]['Sop'])
                K = np.copy(xsamp[old_iter]['K'])
                Gain = np.copy(xsamp[old_iter]['Gain'])
                Akern = np.copy(xsamp[old_iter]['Akern'])
                vres = np.copy(xsamp[old_iter]['vres'])
                gfac = xsamp[old_iter]['gamma']
                sic = xsamp[old_iter]['sic']
                dfs = np.copy(xsamp[old_iter]['dfs'])
                cdfs = np.copy(xsamp[old_iter]['cdfs'])
                di2m = xsamp[old_iter]['di2m']
                rmsa = xsamp[old_iter]['rmsa']
                rmsr = xsamp[old_iter]['rmsr']
                rmsp = xsamp[old_iter]['rmsp']
                chi2 = xsamp[old_iter]['chi2']
                itern = old_iter

            # But also check for convergence in the normal manner
            if ((gfactor[itern-1] <= 1) & (gfactor[itern] == 1)):
                if di2m < cvgmult * nY:
                    converged = 1                    # Converged in "classical sense"

        prev_di2m = di2m
        # Place the data into a structure (before we do the update)
        xtmp = {'idx':i, 'secs':aeri['secs'][i], 'ymd':aeri['date'], 'hour':aeri['secs'][i]/3600.,
                'nX':nX, 'nY':nY, 'dimY':np.copy(dimY), 'Y':np.copy(Y), 'sigY':np.copy(sigY), 'flagY':np.copy(flagY),
                'niter':itern, 'z':np.copy(z), 'p':np.copy(p), 'hatchopen':0,
                'cbh':cbh, 'cbhflag':cbhflag,
                'X0':np.copy(X0), 'Xn':np.copy(Xn), 'FXn':np.copy(FXn), 'Sop':np.copy(Sop),
                'K':np.copy(Kij), 'Gain':np.copy(Gain), 'Akern':np.copy(Akern), 'vres':np.copy(vres),
                'gamma':gfac, 'qcflag':0, 'sic':sic, 'dfs':np.copy(dfs), 'cdfs':np.copy(cdfs), 'di2m':di2m, 'rmsa':rmsa,
                'rmsr':rmsr, 'rmsp':rmsp, 'chi2':chi2, 'converged':converged}
        
        # Update the state vector, if we need to do another iteration
        if converged == 0:
           if verbose >= 1:
               print(f"    iter is {itern:2d}, di2m is {di2m:.3e}, and RMS is {rmsa:.3e}")
           Xn = np.copy(Xnp1[:,0])
           Fxnm1 = np.copy(FXn)
           itern += 1
        
        # And store each iteration in case I would like to investigate how
        # the retrieval functioned in a sample-by-sample way

        if itern == 1:
            xsamp = [copy.deepcopy(xtmp)]
        else:
            xsamp.append(copy.deepcopy(xtmp))
        
    if continue_next_sample == 1:
        continue         # This was set if the Jacobian could not be computed

    # If the retrieval converged, then let's store the various
    # pieces of information for later. Otherwise, let's just move on...
    if converged == 1:
        print('    Converged! (di2m << nY)')
    elif converged == 2:
        print('    Converged (best RMS as RMS drastically increased)')
    elif converged == 9:
        print('    Converged (found NaN in Xnp1 so abort sample)')
    else:
        
        # If the retrieval did not converged but performed max_iter iterations
        # means that the RMS didn't really increase drastically at any one step.
        # Let's select the sample that has the best RMS but weight the value
        # so that we are picking it towards the end ofthe iterations (use gamma
        # to do so), and save it

        vval = []
        for samp in xsamp:
            vval.append(np.sqrt(samp['gamma']) * samp['rmsa'])
        vval = np.array(vval)
        foo = np.where(vval < np.min(vval)*1.00001)[0]
        converged = 3           # Converged in "best rms after max_iter" sense
        itern = int(foo[0])
        Xn = np.copy(xsamp[itern]['Xn'])
        FXn = np.copy(xsamp[itern]['FXn'])
        Sop = np.copy(xsamp[itern]['Sop'])
        K = np.copy(xsamp[itern]['K'])
        Gain = np.copy(xsamp[itern]['Gain'])
        Akern = np.copy(xsamp[itern]['Akern'])
        vres = np.copy(xsamp[itern]['vres'])
        gfac = xsamp[itern]['gamma']
        sic = xsamp[itern]['sic']
        dfs = np.copy(xsamp[itern]['dfs'])
        cdfs = np.copy(xsamp[itern]['cdfs'])
        di2m = xsamp[itern]['di2m']
        rmsa = xsamp[itern]['rmsa']
        rmsr = xsamp[itern]['rmsr']
        rmsp = xsamp[itern]['rmsp']
        chi2 = xsamp[itern]['chi2']
        xtmp = {'idx':i, 'secs':aeri['secs'][i], 'ymd':aeri['date'], 'hour':aeri['secs'][i]/3600.,
                'nX':nX, 'nY':nY, 'dimY':np.copy(dimY), 'Y':np.copy(Y), 'sigY':np.copy(sigY), 'flagY':np.copy(flagY),
                'niter':itern, 'z':np.copy(z), 'p':np.copy(p), 'hatchopen':0,
                'cbh':cbh, 'cbhflag':cbhflag,
                'X0':np.copy(X0), 'Xn':np.copy(Xn), 'FXn':np.copy(FXn), 'Sop':np.copy(Sop),
                'K':np.copy(Kij), 'Gain':np.copy(Gain), 'Akern':np.copy(Akern), 'vres':np.copy(vres),
                'gamma':gfac, 'qcflag':0, 'sic':sic, 'dfs':np.copy(dfs), 'cdfs':np.copy(cdfs), 'di2m':di2m, 'rmsa':rmsa,
                'rmsr':rmsr, 'rmsp':rmsp, 'chi2':chi2, 'converged':converged}

        xsamp.append(xtmp)
        print('Converged! (best RMS after max_iter)')
    
    # Store the data, regardless whether it converges or not
    if xret == []:
        xret = [copy.deepcopy(xsamp[len(xsamp)-1])]
    else:
        xret.append(copy.deepcopy(xsamp[len(xsamp)-1]))

    endtime = datetime.now()
    
    # Determine the QC of the sample
    # Then look for a retrieval that didn't converge
    if xret[fsample]['converged'] != 1:
        xret[fsample]['qcflag'] = 2
    # Then look for a retrieval where the RMS is too large
    if xret[fsample]['rmsa'] > vip['qc_rms_value']:
        xret[fsample]['qcflag'] = 3
    
    # Compute the various convective indices and other useful data
    # Write the data into the netCDF file

    success, noutfilename = Output_Functions.write_output_tropoe(vip, -999., -999., -999., -999.,
              globatt, xret, prior, fsample, version, (endtime-starttime).total_seconds(),
              modeflag, noutfilename, location, verbose)
    
    if success == 0:
        VIP_Databases_functions.abort(lbltmpdir,date)
        sys.exit()

    already_saved = 1
    fsample += 1




shutil.rmtree(lbltmp)

totaltime = (endtime - starttime).total_seconds()

print(('Processing took ' + str(totaltime) + ' seconds'))

shutil.rmtree(lbltmpdir)

# Successful exit
print(('>>> TROPoe retrieval on ' + str(date) + ' ended properly <<<'))
print('--------------------------------------------------------------------')
print(' ')