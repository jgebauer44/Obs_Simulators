import sys
import numpy as np
import glob
import scipy
from datetime import datetime
from subprocess import Popen, PIPE

import Other_functions
import Calcs_Conversions
import LBLRTM_Functions

###############################################################################
# This function computes the radiative transfer for infared spectrometers. It 
# is heavily based on the forward operator calculation for TROPoe
###############################################################################

def calc_ir_radiances(model,lblhome,lbldir,lblroot,lbl_std_atmos,tp5,tp3,co2,ch4,n2o,
                      lblwnum1,lblwnum2,awnum,extra_layers,stdatmos,sspl,sspi):
    
    success = 0
    quiet = 1
    
    stime = datetime.now()
    k = len(model['z'])
    zz = np.copy(model['z'])
    t = np.copy(model['t'])         #degC
    w = np.copy(model['w'])         #g/kg
    lwp = np.copy(model['lwp'])     #g/m2
    reffl = np.copy(model['effc'])    #um
    reffi = np.copy(model['effi'])   #um
    taul = np.copy(model['tauc'])     # unitless
    tausi = np.copy(model['taui'])    # unitles
    p = np.copy(model['p'])           # hPa
    zdif = np.copy(model['zdif'])     # km (height between model levels)

    t += 273.15
    
    # Make sure pressure at surface isn't lower than the first
    # model level because apparently that can happen

    if model['psfc'] < p[0]:
        print('Pressure decreased with height. Skipping Sample.')
        return success, -999. -999. -999
 
    zz = np.insert(zz,0,model['alt'])
    t = np.insert(t,0,model['tsfc']+273.15)
    w = np.insert(w,0,model['wsfc'])
    p = np.insert(p,0,model['psfc'])
    zdif = np.insert(zdif,0,zz[1]-zz[0])
    
    #These are just dummy values that won't affect the radiation but are needed
    #due to array lengths
    reffl = np.insert(reffl,0,1.01)
    reffi = np.insert(reffl,0,1.01)
    taul = np.insert(taul,0,0)
    tausi = np.insert(tausi,0,0)
    
    lblrun = lblhome + '/bin/lblrun'
    

    # Define the model layers
    if len(extra_layers) > 0:
        mlayerz = np.append(zz,extra_layers)
        mlayert = np.append(t, np.interp(extra_layers, stdatmos['z'], stdatmos['t']))
    else:
        mlayerz = zz
        mlayert = t
    
    zz = np.round(zz,decimals=3)
    mlayerz = np.round(mlayerz,decimals=3)
        
    LBLRTM_Functions.rundecker(3, lbl_std_atmos, zz, p, t, w,
             co2_profile=co2, ch4_profile=ch4, n2o_profile=n2o,
             od_only=1, mlayers=mlayerz, wnum1=lblwnum1, wnum2=lblwnum2, tape5=tp5+'.1',
             v10=True, silent=True)
    
    command1 = ('setenv LBL_HOME ' +lblhome + ' ; '+
                'rm -rf ' + lblroot + '.1 ; '+
                'mkdir ' + lblroot + '.1 ; ' +
                'setenv LBL_RUN_ROOT ' + lblroot + '.1 ; '+
                'rm -rf ' + lbldir + '.1 ; '+
                '(' + lblrun + ' ' + tp5 + '.1 ' + lbldir + '.1 ' + tp3 + ') >& /dev/null')
    
    command = '(' +command1+')& ; wait '
    
    command = '('+ command + ')>& /dev/null'
    
    print('Starting LBLRTM run')
    process = Popen(command, stdout = PIPE, stderr = PIPE, shell=True, executable = '/bin/csh')
    stdout, stderr = process.communicate()
    print('Finished LBLRTM run')
    
    # Now read in the baseline optical depths
    files1 = []
    files1 = files1 + sorted(glob.glob(lbldir+'.1/OD*'))
    if len(files1) != len(mlayerz)-1:
        print('This should not happen (0) in compute_jacobian_interpol')
        return success, -999., -999.
    
    # Use spectral spacing at 8 km because that is default in TROPoe. Might
    # have to come back to this
    spec_resolution_ht = 8              # km AGL
    foo = np.where(zz >= spec_resolution_ht)[0]
    if len(foo) == 0:
        foo = np.array([len(files1)-1])
    if foo[0] >= len(files1):
        foo[0] = len(files1)-1
    
    s0, v0 = LBLRTM_Functions.lbl_read(files1[foo[0]], do_load_data=True)
    v = np.copy(v0)
    od00 = np.zeros((len(files1),len(v)))
    
    for i in range(len(files1)):
        s0, v0 = LBLRTM_Functions.lbl_read(files1[i], do_load_data=True)
        od00[i,:] = np.interp(v,v0,s0)
    
    wnum = np.copy(v)
    gasod = np.copy(od00)
    
    # Get the desired cloud absorption optical depth spectrum
    # We sadly have to do this one height at a time because of how the function
    # is written
    
    cldodvis = np.copy(taul)
    lcldodir = np.zeros((k, len(wnum)))
    icldodir = np.zeros((k, len(wnum)))
    
    for i in range(k):
        lcldodir[i,:] = Other_functions.get_ir_cld_ods(sspl,cldodvis[i],wnum,reffl[i],zdif[i])
        icldodir[i,:] = Other_functions.get_ir_cld_ods(sspi,tausi[i],wnum,reffi[i],zdif[i])
        
    # Add the absoption cloud optical depth
    print(np.max(lcldodir))
    print(np.max(icldodir))
    
    gasod0 = np.copy(gasod)
    gasod[0:k,:] += lcldodir + icldodir
    
    # Compute the surface to layer transmission
    
    trans1 = np.copy(gasod)
    trans1[0,:] = 1
    for i in range(1, len(mlayert)-1):
        trans1[i,:] = trans1[i-1,:]*np.exp(-gasod[i-1,:])
    
    
    # Compute the reflected radiance from the cloud.
	# I am using Dave Turner's simple approximation for cloud reflectivity
	# that varies as a function of wavenumber and cloud optical
	# depth.  It assumes the surface is black and has the same
	# temperature as the lowest atmospheric layer.  We need to
	# account for the 2-way attenution by the atmosphere.  Note
	# that we are also assuming that the amount of radiation emitted
	# by the atmosphere and reflected by the cloud is negligible.
    
    # Have to loop over each height here due to the limits of the function
    reflection = np.zeros((k,len(wnum)))
    for i in range(k):
        reflection[i,:] = Other_functions.cloud_reflectivity(v,cldodvis[i]+tausi[i])
    
    sfcrad = Calcs_Conversions.planck(v,model['tsfc']+273.16)
    cldrefrad = np.sum(sfcrad * reflection * trans1[0:k,:] * trans1[0:k,:], axis = 0)
    
    # Compute the radiance
    radv = Other_functions.radxfer(v, mlayert, gasod)
    radv += cldrefrad
    bar = Other_functions.convolve_to_aeri(v,radv)
    bwnum = np.copy(bar['wnum'])
    brad = np.copy(bar['spec'])
    
    # Now cut the radiance down to the AERI bands
    wpad = 5
    foo = np.where((np.min(awnum)+wpad <= bwnum) & (bwnum <= np.max(awnum)-wpad))[0]
    wnumc  = bwnum[foo]
    
    # Now cut it down again to the specific range we want
    foo = np.where((np.min(wnumc)-0.1 <= bwnum) & (bwnum <= np.max(wnumc)+0.1))[0]
    if ((len(foo) != len(wnumc)) | (np.abs(np.min(wnumc)-np.min(bwnum[foo])) > 0.1)):
        print('PROBLEM inside compute_ir_radiance -- wavenumber do not match')
        return success, -999.,-999.,-999.
    
    rad = np.copy(brad[foo])
    
    
    # Capture the total time and return
    etime = datetime.now()
    totaltime = (etime-stime).total_seconds()
    success = 1
    
    return success, rad, wnumc, totaltime

###############################################################################
# This function calculates the brightness temperature for microwave radiometer
# zenith pointing scans.
###############################################################################

def calc_mwr_brightness_temp(model,freq,workdir,monortm_tfile,
                             monortm_exec, stdatmos):
    
    flag = 0
    t = np.copy(model['t'])
    w = np.copy(model['w'])
    p = np.copy(model['p'])
    lwp = np.copy(model['lwp'])
    cbh = np.copy(model['cbh'])
    cth = np.copy(model['cth'])
    z = np.copy(model['z'])

    if model['psfc'] < p[0]:
       print('Pressure decreasing with height. Skipping sample')
       return flag, -999, -999
    
    z = np.insert(z,0,np.round(model['alt']*1000)/1000)
    t = np.insert(t,0,model['tsfc'])
    w = np.insert(w,0,model['wsfc'])
    p = np.insert(p,0,model['psfc'])
    
    # Do a check here to see if there were clouds in the model simulation. If 
    # not set lwp to zero and use a default cbh of 2 km
    
    if np.isnan(cbh):
        print('No clouds in the simulation. Zeroing out LWP and using default cbh')
        lwp = 0
        cbh = 2.0
        cth = 2.3
    
    stime = datetime.now()
    # Perform the brightness temperature calculation using MonoRTM
    u = Calcs_Conversions.w2rh(w,p,t,0) * 100
    Other_functions.write_arm_sonde_file(z*1000,p,t,u,workdir+'/'+monortm_tfile,silent=True)
    command = monortm_exec + ' ' + monortm_tfile + ' {:3.1f} {:8.2f} {:6.3f} {:6.3f}'.format(1.0, lwp, cbh, cth)
    a = LBLRTM_Functions.run_monortm(command,freq,z,stdatmos)
    if a['status'] == 0:
        print('Problem with MonoRTM calc')
        return flag, -999., -999
    
    bt = np.copy(a['tb'])
    
    # Capture the execution time
    etime = datetime.now()
    totaltime = (etime-stime).total_seconds()
    
    flag = 1
    
    return flag, bt, totaltime

def calc_mwrscan_brightness_temp(model,freq,elev,workdir,monortm_tfile,monortm_exec,
                             stdatmos):
    
    flag = 0
    t = np.copy(model['t'])
    w = np.copy(model['w'])
    p = np.copy(model['p'])
    lwp = np.copy(model['lwp'])
    cbh = np.copy(model['cbh'])
    cth = np.copy(model['cth'])
    z = np.copy(model['z'])
    
    z = np.insert(z,0,np.round(model['alt']))
    t = np.insert(t,0,model['tsfc'])
    w = np.insert(w,0,model['wsfc'])
    p = np.insert(p,0,model['psfc'])
    
    # Do a check here to see if there were clouds in the model simulation. If 
    # not set lwp to zero and use a default cbh of 2 km
    
    if np.isnan(cbh):
        print('No clouds in the simulation. Zeroing out LWP and using default cbh')
        lwp = 0
        cbh = 2.0
        cth = 2.3
        
    # Allocate space for the brightness temperatures
    
    bt = np.ones((len(freq)*len(elev)))*-999.
    
    # Extract out the really unique angles
    nelev = np.copy(elev)
    foo = np.where(nelev > 90)[0]
    if len(foo) > 0:
        nelev[foo] = 180-nelev[foo]
    uelev = np.array([nelev[0]])
    for ii in range(len(nelev)):
        unq = 1
        for jj in range(len(uelev)):
            if np.abs(nelev[ii]-uelev[jj]) < 0.1: unq = 0
        if unq == 1: uelev = np.append(uelev, nelev[ii])
        
    for ii in range(len(uelev)):
        
        # Perform the brightness temperature calculation using MonoRTM
        u = Calcs_Conversions.w2rh(w,p,t,0) * 100
        Other_functions.write_arm_sonde_file(z*1000,p,t,u,workdir+'/'+monortm_tfile,silent=True)
        elevOff = 0.1
        cnt = 0
        didfail = 0
        command = monortm_exec + ' ' + monortm_tfile + ' {:3.1f} {:8.2f} {:6.3f} {:6.3f} {:6.3f}'.format(1.0, lwp, cbh, cth, 90-uelev[ii]+cnt*elevOff)
        a = LBLRTM_Functions.run_monortm(command, freq, z, stdatmos)
        while((a['status'] != 1) & (cnt < 2)):
            cnt += 1
            command = monortm_exec + ' ' + monortm_tfile + ' {:3.1f} {:8.2f} {:6.3f} {:6.3f} {:6.3f}'.format(1.0, lwp, cbh, cth, 90-uelev[ii]+cnt*elevOff)
            a = LBLRTM_Functions.run_monortm(command, freq, z, stdatmos)
        
        if a['status'] == 0:
            print('    Bending angle problem with MonoRTM in mwrScan0')
            didfail = 1
        
        else:
            foo = np.where(np.abs(nelev - uelev[ii]) < 0.1)[0]
            idx = np.arange(len(freq))
            for kk in range(len(foo)):
                bt[foo[kk]*len(freq)+idx] = a['tb']
    
    
        
    flag = 1
    
    return flag, bt
                
        
        
    
    
    
    
