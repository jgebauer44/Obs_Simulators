#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 14:43:28 2022

@author: joshua.gebauer
"""

import os
import sys
import numpy as np
import glob
from netCDF4 import Dataset
from datetime import datetime, timedelta
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("start_date", type=str, help="Date to start [YYYYMMDD]")
parser.add_argument("end_date",type=str, help="Date to end [YYYYMMDD]")
parser.add_argument("input_dir",type=str,help="Path to input directory")
parser.add_argument("--output_dir", help="Path to output directory")

args = parser.parse_args()

start_date = args.start_date
end_date = args.end_date
input_dir = args.input_dir
output_dir = args.output_dir

if output_dir is None:
    output_dir = os.getcwd() + '/'

start_date = datetime(int(start_date[0:4]),int(start_date[4:6]),int(start_date[6:8]))
end_date = datetime(int(end_date[0:4]),int(end_date[4:6]),int(end_date[6:8]))


# Get the wnum1 array from a ch1 file
file = glob.glob(input_dir + '/clampsaerich1C1.b1/*' +start_date.strftime('%Y%m%d') + '*')

f = Dataset(file[0],'r')

wnum = f.variables['wnum1'][:]

f.close()

count = 0
current_date = start_date
while current_date < end_date:
    
    file = []
    file = file + glob.glob(input_dir + 'clampsaerisummaryC1.b1/*summary' + current_date.strftime('%Y%m%d') + '*')
    
    if len(file) == 0:
        print('File not found for ' + current_date.strftime('%Y%m%d'))
        current_date += timedelta(days=1)
        continue

    f = Dataset(file[0],'r')
    
    wnum_sum = f.variables['wnum1'][:]
    
    temp_noise = f.variables['skyNENch1'][:]
    
    f.close()
    
    if count == 0:
        noise_sum = np.nansum(temp_noise,axis=0)
    else:
        noise_sum += np.nansum(temp_noise,axis=0)
    
    count += temp_noise.shape[0]

avg_noise = noise_sum/count

# Now interpolate the noise to the wnum1 from the ch1 file
avg_noise_interp = np.interp(wnum,wnum_sum,avg_noise)

# Write this all to a file

out = Dataset(output_dir + '/AERI_avg_noise_' + start_date.strftime('%Y%m%d') + '_' + end_date.strftime('%Y%m%d'))

wdim = out.createDimension('wnum',len(wnum))

wnum1 = out.createVariable('wnum','f4', ('wnum',))
wnum1.long_name = 'Wavenumber'
wnum1[:] = wnum[:]

noise = out.createVariable('noise','f4', ('wnum',))
noise.long_name = 'Average noise'
noise[:] = avg_noise_interp[:]

out.close()



    
    
    
    
    





