#!/opt/local/anaconda3/bin/python
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 11:16:18 2021

@author: jgebauer
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
from scipy.optimize import leastsq

##############################################################################
# This is a class created for VAD data
##############################################################################
class VAD:
    def __init__(self,u,v,w,speed,wdir,du,dv,dw,z,residual,correlation,time,el,nbeams):
       self.u = np.array(u)
       self.v = np.array(v)
       self.w = np.array(w)
       self.speed = np.array(speed)
       self.wdir = np.array(wdir)
       self.du = np.array(du)
       self.dv = np.array(dv)
       self.dw = np.array(dw)
       self.z = z
       self.residual = np.array(residual)
       self.correlation = np.array(correlation)
       self.time = np.array(time)
       self.el = el
       self.nbeams = nbeams

##############################################################################
# This is a class created for DBS data
##############################################################################
class DBS:
    def __init__(self,u,v,w,speed,wdir,z,z_w,time):
       self.u = np.array(u)
       self.v = np.array(v)
       self.w = np.array(w)
       self.speed = np.array(speed)
       self.wdir = np.array(wdir)
       self.z = z
       self.z_w = np.array(z_w)
       self.time = np.array(time)
       
##############################################################################
# This is a class created for gridded RHI data
##############################################################################

class gridded_RHI:
    def __init__(self,field,x,z,dx,offset,grid_el,grid_range,time):
        self.field = np.array(field)
        self.x = np.array(x)
        self.z = np.array(z)
        self.dx = dx
        self.x_offset = offset[0]
        self.z_offset = offset[1]
        self.time = np.array(time)
        self.grid_el = grid_el
        self.grid_range = grid_range

##############################################################################
# This is a class created for a vertical profile from a lidar RHI or multiple
# VADs
##############################################################################

class vertical_vr:
    def __init__(self,field,x_loc,y_loc,z,dz,offset,z_el,z_range,az,time):
        self.field = np.array(field)
        self.x = x_loc
        self.y = y_loc
        self.z = np.array(z)
        self.dz = dz
        self.x_offset = offset[0]
        self.y_offset = offset[1]
        self.z_offset = offset[2]
        self.time = np.array(time)
        self.azimuth = az
        self.elevation = np.array(z_el)
        self.range = np.array(z_range)

def ARM_VAD(radial_vel,ranges,el,az,time=None,missing=None):
    '''
    Calculates VAD wind profiles using technique shown in Newsom et al. (2019)

    Parameters
    ----------
    radial_vel : 1 or 2D array
        Radial velocity data in array formatted either as (ray,range) or
        (scan,ray,range)
    ranges : 1D array
        Array that contains the range of the lidar gates
    el : float
        Elevation of the the VAD
    az : 1D Array
        Array that contains the azimuths the rays
    time : array, optional
        An array that contains the time of each scan. If None, this 
        becomes scan number.The default is None.
    missing : int or float, optional
        The value that represents missing data in the VAD. The default is None.

    Returns
    -------
    VAD class
        A VAD class that contains all information needed from VAD.

    '''
    
    if (time is None) & (len(radial_vel.shape) == 2):
        times = 1
        time = [0]
        vr = np.array([radial_vel])
    elif (time is None) & (len(radial_vel.shape) == 3):
        time = np.arange(radial_vel.shape[0])
        vr = np.copy(radial_vel)
        times = len(time)
    else:
        times = len(time)
        vr = np.copy(radial_vel)
        
    if missing is not None:
        vr[vr==missing] = np.nan
    
    x = ranges[None,:]*np.cos(np.radians(el))*np.sin(np.radians(az[:,None]))
    y = ranges[None,:]*np.cos(np.radians(el))*np.cos(np.radians(az[:,None]))
    z = ranges*np.sin(np.radians(el))
    
    u = []
    v = []
    w = []
    du = []
    dv = []
    dw = []
    residual = []
    vr_error = []
    speed = []
    wdir = []
    correlation = []
    
    for j in range(times):
        temp_u = np.ones(len(ranges))*np.nan
        temp_v = np.ones(len(ranges))*np.nan
        temp_w = np.ones(len(ranges))*np.nan
        temp_du = np.ones(len(ranges))*np.nan
        temp_dv = np.ones(len(ranges))*np.nan
        temp_dw = np.ones(len(ranges))*np.nan
        
        for i in range(len(ranges)):
            foo = np.where(~np.isnan(vr[j,:,i]))[0]
            
            # Need at a least rays to do a VAD
            if len(foo) < 3:
                temp_u[i] = np.nan
                temp_v[i] = np.nan
                temp_w[i] = np.nan
                temp_du[i] = np.nan
                temp_dv[i] = np.nan
                temp_dw[i] = np.nan
                continue
            
            
            A11 = (np.cos(np.deg2rad(el))**2) * np.sum(np.sin(np.deg2rad(az[foo]))**2)
            A12 = (np.cos(np.deg2rad(el))**2) * np.sum(np.sin(np.deg2rad(az[foo])) * np.cos(np.deg2rad(az[foo])))
            A13 = (np.cos(np.deg2rad(el))*np.sin(np.deg2rad(el))) * np.sum(np.sin(np.deg2rad(az[foo])))
            A22 = (np.cos(np.deg2rad(el))**2) * np.sum(np.cos(np.deg2rad(az[foo]))**2)
            A23 = (np.cos(np.deg2rad(el))*np.sin(np.deg2rad(el))) * np.sum(np.cos(np.deg2rad(az[foo])))
            A33 = len(az[foo]) * (np.sin(np.deg2rad(el))**2)

            A = np.array([[A11,A12,A13],[A12,A22,A23],[A13,A23,A33]])
            invA = np.linalg.inv(A)
    
            temp_du[i] = invA[0,0]
            temp_dv[i] = invA[1,1]
            temp_dw[i] = invA[2,2]
            
            b1 = np.cos(np.deg2rad(el)) * np.sum(vr[j,foo,i] * np.sin(np.deg2rad(az[foo])))
            b2 = np.cos(np.deg2rad(el)) * np.sum(vr[j,foo,i] * np.cos(np.deg2rad(az[foo])))
            b3 = np.sin(np.deg2rad(el)) * np.sum(vr[j,foo,i])
    
            b = np.array([b1,b2,b3])
    
            temp = invA.dot(b)
            temp_u[i] = temp[0]
            temp_v[i] = temp[1]
            temp_w[i] = temp[2]
        
            summ = np.sqrt(np.nansum(((((temp_u[i]*x[:,i])+(temp_v[i]*y)[:,i]+(temp_w[i]*z[None,i]))/np.sqrt(x[:,i]**2+y[:,i]**2+z[None,i]**2))-vr[j,:,i])**2,axis = 0))
            temp_du[i] = summ*np.sqrt(temp_du[i]/(len(foo)))
            temp_dv[i] = summ*np.sqrt(temp_dv[i]/(len(foo)))
            temp_dw[i] = summ*np.sqrt(temp_dw[i]/(len(foo)))
            
        u.append(np.copy(temp_u))
        v.append(np.copy(temp_v))
        w.append(np.copy(temp_w))
        du.append(np.copy(temp_du))
        dv.append(np.copy(temp_dv))
        dw.append(np.copy(temp_dw))
    
        residual.append(np.sqrt(np.nanmean(((((temp_u*x)+(temp_v*y)+((temp_w*z)[None,:]))/np.sqrt(x**2+y**2+z[None,:]**2))-vr[j])**2,axis = 0)))
        speed.append(np.sqrt(temp_u**2 + temp_v**2))
        temp_wdir = 270 - np.rad2deg(np.arctan2(temp_v,temp_u))
        
        

        foo = np.where(temp_wdir >= 360)[0]
        temp_wdir[foo] -= 360

        wdir.append(temp_wdir)
        
        u_dot_r = ((temp_u*x)+(temp_v*y)+((temp_w*z)[None,:]))/np.sqrt(x**2+y**2+z**2)
        mean_u_dot_r = np.nanmean(((temp_u*x)+(temp_v*y)+((temp_w*z)[None,:]))/\
                       np.sqrt(x**2+y**2+z[None,:]**2),axis=0)
        mean_vr = np.nanmean(vr[j],axis=0)
        correlation.append(np.nanmean((u_dot_r-mean_u_dot_r)*(vr[j]-mean_vr),axis=0)/\
                           (np.sqrt(np.nanmean((u_dot_r-mean_u_dot_r)**2,axis=0))*\
                            np.sqrt(np.nanmean((vr[j]-mean_vr)**2,axis=0))))

    return VAD(u,v,w,speed,wdir,du,dv,dw,z,residual,correlation,time,el,len(az))

def Calc_DBS(radial_vel,ranges,el,fifth_beam = False,time=None,missing=None):
    '''

    Parameters
    ----------
    radial_vel : 1 or 2D array
        Radial velocity data in array formatted either as (ray,range) or
        (scan,ray,range). Rays azimuth must be ordered (0,90,180,270,vertical).
    ranges : 1D array
        Array that contains the range of the lidar gates
    el : float
        Elevation of the the VAD
    fifth_beam : Boolean, optional
        If true the DBS uses the vertical beam for w. The default is False.
    time : array, optional
        An array that contains the time of each scan. If None, this 
        becomes scan number.The default is None.
    missing : int or float, optional
        The value that represents missing data in the VAD. The default is None.

    Returns
    -------
    DBS class
        A DBS class that contains all information needed from DBS.

    '''
    if (time is None) & (len(radial_vel.shape) == 2):
        times = 1
        time = [0]
        vr = np.array([radial_vel])
        
        if fifth_beam:
            if vr.shape[0] < 5:
                raise IOError('Must have 5 rays when fifth beam is true')
        
    elif (time is None) & (len(radial_vel.shape) == 3):
        time = np.arange(radial_vel.shape[0])
        vr = np.copy(radial_vel)
        times = len(time)
        if fifth_beam:
            if vr.shape[1] < 5:
                raise IOError('Must have 5 rays when fifth beam is true')
    else:
        times = len(time)
        vr = np.copy(radial_vel)
        
    if missing is not None:
        vr[vr==missing] = np.nan
    
    z = ranges*np.sin(np.radians(el))

    u = []
    v = []
    w = []
    speed = []
    wdir = []
    
    for j in range(times):
        temp_u = np.ones(len(ranges))*np.nan
        temp_v = np.ones(len(ranges))*np.nan
        temp_w = np.ones(len(ranges))*np.nan
        
        for i in range(len(ranges)):
            foo = np.where(np.isnan(vr[j,:,i]))[0]
        
            if len(foo) > 0:
                temp_u[i] = np.nan
                temp_v[i] = np.nan
                temp_w[i] = np.nan
                continue
        
            temp_u[i] = vr[j,1,i] - vr[j,3,i]/(2*np.sin(np.deg2rad(90-el)))
            temp_v[i] = vr[j,0,i] - vr[j,2,i]/(2*np.sin(np.deg2rad(90-el)))
            if fifth_beam:
                temp_w[i] = vr[j,4,i]
            else:
                temp_w[i] = (vr[j,0,i]+vr[j,1,i]+vr[j,2,i]+vr[j,3,i])/4*np.cos(np.deg2rad(90-el))
        
        u.append(np.copy(temp_u))
        v.append(np.copy(temp_v))
        w.append(np.copy(temp_w))
        
        speed.append(np.sqrt(temp_u**2 + temp_v**2))
        temp_wdir = 270 - np.rad2deg(np.arctan2(temp_v,temp_u))

        foo = np.where(temp_wdir >= 360)[0]
        temp_wdir[foo] -= 360
        
        wdir.append(temp_wdir)
    
    if fifth_beam:    
        return DBS(u,v,w,speed,wdir,z,ranges,time)
    else:
        return DBS(u,v,w,speed,wdir,z,z,time)
        
def plot_VAD(vad,dname,plot_time_index = None, title=None):
    '''
    This function will plot profiles from VAD class

    Parameters
    ----------
    vad : VAD
        VAD class
    dname : str
        Directory to save plots in
    plot_time_index : int, optional
        Index to specific time from VAD class to plot.
        If None, all times will be plotted. The default is None.
    title : str, optional
        Title to put on the plot. The default is None.

    '''
    if plot_time_index is None:
        plot_time = len(vad.time) - 1
        start = 0
    else:
        start = plot_time
    
    for i in range(start,plot_time+1):
        plt.figure(figsize=(12,8))
        ax = plt.subplot(141)
        plt.plot(vad.speed[i],vad.z)
        ax.set_xlim(0,30)
        ax.set_ylim([0,np.max(vad.z)])
        ax.set_xlabel('Wind Speed [m/s]')
        ax.set_ylabel('Height [m]')
 
        ax = plt.subplot(142)
        plt.plot(vad.wdir[i],vad.z)
        ax.set_xlim([0,360])
        ax.set_ylim([0,np.max(vad.z)])
        ax.set_xlabel('Wind Direction')

        ax = plt.subplot(143)
        plt.plot(vad.w[i],vad.z)
        ax.set_xlim(-5,5)
        ax.set_ylim([0,np.max(vad.z)])
        ax.set_xlabel('Vertical Velocity [m/s]')
    
        ax = plt.subplot(144)
        plt.plot(vad.residual[i],vad.z)
        ax.set_xlim(0,10)
        ax.set_ylim([0,np.max(vad.z)])
        ax.set_xlabel('Residual')

        if title is not None:
            plt.title(title)
    
        plt.tight_layout()
        
        if os.path.isdir(dname):
            plt.savefig(dname + '/VAD_' + str(vad.time[i]) + '.png')
        else:
            os.mkdir(dname)
            plt.savefig(dname + '/VAD_' + str(vad.time[i]) + '.png')
        
        plt.close()
        
    return


def grid_rhi(field,elevation,ranges,azimuth,dims,dx,offset=None,
             time=None,missing=None):
    '''
    Puts data from lidar RHIs into 2D cartesian grid using linear interpolation

    Parameters
    ----------
    field : 1 or 2D array
        Data in array formatted either as (ray,range) or (scan,ray,range)
    elevation : 1D array
        Elevation angle of each ray in the scan
    ranges : 1D array
        Range of the lidar range gates
    azimuth : float
        The azimuth angle at which the RHI is done
    dims : tuple like ((xmin,xmax),(zmin,zmax))
        The dimensions in meters of the 2D grid.
    dx : int or float
        Grid spacing in meters of the 2D grid.
    offset : tuple like (x_offset, z_offset), optional
        Distance lidar is from grid origin. If none the lidar is 
        assumed to be at the grid origin. The default is None.
    time : array, optional
        An array that contains the time of each scan. If None, this 
        becomes scan number.The default is None.
    missing : int or float, optional
        The value that represents missing data in the VAD. The default is None.

    Returns
    -------
    gridded_RHI class
        A class that contains the all relevant data for the gridded RHI

    '''
    if len(dims) != 2:
        raise IOError('Dims must be a 2 length tuple')
    
    if offset is not None:
        if len(offset) != 2:
            raise IOError('If offset is specified it must be 2 length tuple')
    else:
        offset = (0,0)
        
    if (time is None) & (len(field.shape) == 2):
        times = 1
        time = [0]
        raw = np.array([field])
        el = np.array([elevation])
    elif (time is None) & (len(field.shape) == 3):
        time = np.arange(field.shape[0])
        el = np.copy(elevation)
        raw = np.copy(field)
        times = len(time)
    else:
        times = len(time)
        raw = np.copy(field)
        el = np.copy(elevation)
        
    if missing is not None:
        raw[raw==missing] = np.nan
        
    x = ranges[None,:] * np.cos(np.deg2rad(180-el))[:,:,None] + offset[0]
    z = ranges[None,:] * np.sin(np.deg2rad(el))[:,:,None] + offset[1]
    
    grid_x, grid_z = np.meshgrid(np.arange(dims[0][0],dims[0][1]+1,dx),np.arange(dims[1][0],dims[1][1]+1,dx))
    
    grid_range = np.sqrt((grid_x-offset[0])**2 + (grid_z-offset[1])**2)
    grid_el = 180 - np.rad2deg(np.arctan2(grid_z-offset[1],grid_x-offset[0]))

    grid_field = []
    for i in range(times):
        foo = np.where(~np.isnan(x[i]))
        
        grid_field.append(scipy.interpolate.griddata((x[i,foo[0],foo[1]],z[i,foo[0],foo[1]]),
                                                     raw[i,foo[0],foo[1]],(grid_x,grid_z)))
    
    return gridded_RHI(grid_field,grid_x,grid_z,dx,offset,grid_el,grid_range,time)

##############################################################################
# This function calculates a coplanar wind field from two gridded RHIs
##############################################################################

def coplanar_analysis(vr1,vr2,el1,el2,az):
    '''
    Calculates a coplaner wind field from two RHIs that share the same grid

    Parameters
    ----------
    vr1 : 2-D array
        Gridded radial velocity field from first lidar.
    vr2 : 2-D array
        Gridded radial velocity field from second radar.
    el1 : 2-D array
        Gridded elevation angles from first lidar.
    el2 : 2-D array
        Gridded elevation angles from second lidar.
    az : float
        Azimuth of the RHIs (same for both lidars)

    Returns
    -------
    u : 2D array
        Horizontal velocity field
    w : 2D array
       Vertical velocity field

    '''
    
    u = np.ones(vr1.shape)*np.nan
    w = np.ones(vr1.shape)*np.nan
    
    for i in range(vr1.shape[0]):
        for j in range(vr1.shape[1]):
            
            if ((~np.isnan(vr1[i,j])) & (~np.isnan(vr2[i,j]))):
                M = np.array([[np.sin(np.deg2rad(az))*np.cos(np.deg2rad(el1[i,j])),np.sin(np.deg2rad(el1[i,j]))],
                              [np.sin(np.deg2rad(az))*np.cos(np.deg2rad(el2[i,j])),np.sin(np.deg2rad(el2[i,j]))]])
    
                U = np.linalg.solve(M,np.array([vr1[i,j],vr2[i,j]]))
                u[i,j] = U[0]
                w[i,j] = U[1]
    
    return u, w


def vr_variance(field,time,t_avg,axis=0):
    '''
    Calculates Vr-variance from a timeseries of scans

    Parameters
    ----------
    field : 3d array
        Radial velocity data from scans
    time : 1D array
        Time of each scan in seconds
    t_avg : float
        Window of variance calculation
    axis : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    var : 2d array
        Vr-variance from scans
    
    time_avg : 1d array
        Time in the middle of the window for variance calculations

    '''
    
    t_avg = t_avg*60
    start = 0
    yo = np.where(time < (time[0]+t_avg))[0]
    end = yo[-1]+1
    var = []
    time_avg = []
    while end < len(time):
        var.append(np.nanvar(field[start:end,:,:],axis=0))
        time_avg.append(np.nanmean(time[start:end]))
        start = end
        yo = np.where(time < (time[start]+t_avg))[0]
        end = yo[-1]+1
    
    return np.array(var),np.array(time_avg)

##############################################################################
# This function puts vr observations from a RHI onto a vertical grid at one
# location to be used for virtual tower calculations
##############################################################################

def rhi_vertical_profile(field,elevation,azimuth,ranges,heights,dz,loc,offset=None,
                         time=None,missing=None):
    '''
    Generates gridded vertical profiles from RHI

    Parameters
    ----------
    field : 2 or 3d array
        Data in array formatted either as (ray,range) or (scan,ray,range)
    elevation : 1d array
        Elevation angle for each ray
    azimuth : float
        Azimuth of the RHI
    ranges : 1d array
        Range of the lidar range gates
    heights : tuple like (zmin, zmax)
        The range of heights for the extracted profile
    dz : float
        Vertical grid spacing of the profile
    loc : tuple like (x,y)
        Location of the extracted profile 
    offset : tuple like (x_offset,y_offset,z_offset), optional
        Distance lidar is offset from grid origin used to define loc.
        If None it is assumed the lidar is grid origin. The default is None.
    time : array, optional
        An array that contains the time of each scan. If None, this 
        becomes scan number.The default is None.
    missing : int or float, optional
        The value that represents missing data in the VAD. The default is None.

    Returns
    -------
    vertical_vr class
        A class that contains all relevant data for the extracted profile

    '''
    
    if len(loc) != 2:
        raise IOError('Dims must be a 2 length tuple')
    
    if offset is not None:
        if len(offset) != 3:
            raise IOError('If offset is specified it must be 3 length tuple')
    else:
        offset = (0,0,0)
        
    if (time is None) & (len(field.shape) == 2):
        times = 1
        time = [0]
        raw = np.array([field])
        el = np.array([elevation])
    elif (time is None) & (len(field.shape) == 3):
        time = np.arange(field.shape[0])
        el = np.copy(elevation)
        raw = np.copy(field)
        times = len(time)
    else:
        times = len(time)
        raw = np.copy(field)
        el = np.copy(elevation)
    
    if missing is not None:
        raw[raw==missing] = np.nan
        
    x = ranges[None,:] * np.cos(np.deg2rad(el))[:,:,None]*np.sin(np.deg2rad(azimuth)) + offset[0]
    y = ranges[None,:] * np.cos(np.deg2rad(el))[:,:,None]*np.cos(np.deg2rad(azimuth)) + offset[1]
    z = ranges[None,:] * np.sin(np.deg2rad(el))[:,:,None] + offset[2]
    
    r = np.sqrt(x**2 + y**2)

    z_interp = np.arange(heights[0],heights[1],dz)
    
    z_ranges = np.sqrt((loc[0]-offset[0])**2 + (loc[1]-offset[1])**2+(z_interp-offset[2])**2)
    z_el = np.rad2deg(np.arctan2(z_interp-offset[2],np.sqrt((loc[0]-offset[0])**2+(loc[1]-offset[1])**2)))
    
    grid_field = []
    grid_x,grid_z = np.meshgrid(np.array(np.sqrt(loc[0]**2 + loc[1]**2)),z_interp)
    for i in range(times):
        grid_field.append(scipy.interpolate.griddata((r[i].ravel(),z[i].ravel()),
                                                     raw[i].ravel(),(grid_x,grid_z))[:,0])
    
    return vertical_vr(grid_field,loc[0],loc[1],z_interp,dz,offset,z_el,z_ranges,azimuth,time)
    
##############################################################################
# This function calculates the wind components for a virtual towers
##############################################################################

def virtual_tower(vr,elevation,azimuth,height,uncertainty = 0.45):
    '''
    Calculates virtual towers from vertical profiles of radial velocity

    Parameters
    ----------
    vr : tuple like (vr1(z), vr2(z)) or (vr1(z), vr2(z), vr3(z))
        Radial velocities from the 2 or 3 lidars for virtual towers
    elevation : tuple like (el1(z), el2(z)) or (el1(z), el2(z), el3(z))
        Elevation angles for the extracted profiles from 2 or 3 lidars.
    azimuth : tuple like (az1,az2) or (az1,az2,az3)
        Azimuth of the extracted profile from lidar
    height : 1d array
        Heights for the virtual tower
    uncertainty : float, optional
        The assumed uncertainty of the radial velocities. The default is 0.45.

    Returns
    -------
    wind - 2 or 3d array
        Array that contains the wind components for the virtual tower as [u,v]
        or [u,v,w].
    
    uncertainty - 2 or 3d array
        Array that contains the uncertainties for each wind component in same
        array style as the winds
    '''
    
    if len(vr) == 2:
        u = []
        v = []
        u_uncertainty = []
        v_uncertainty = []
        el1 = np.deg2rad(elevation[0])
        el2 = np.deg2rad(elevation[1])
        az1 = np.deg2rad(azimuth[0])
        az2 = np.deg2rad(azimuth[1])
        for i in range(len(height)):
            if ((np.isnan(vr[0][i]) | np.isnan(vr[1][i]))):
                u.append(np.nan)
                v.append(np.nan)
                
                M = np.array([[np.sin(az1)*np.cos(el1[i]),np.cos(az1)*np.cos(el1[i])],
                              [np.sin(az2)*np.cos(el2[i]),np.cos(az2)*np.cos(el2[i])]])
                invM = np.linalg.inv(M)
                temp_u = np.sqrt((invM[0,0]**2)*(uncertainty**2) + (invM[0,1]**2)*(uncertainty**2))
                temp_v = np.sqrt((invM[1,0]**2)*(uncertainty**2) + (invM[1,1]**2)*(uncertainty**2))
                u_uncertainty.append(temp_u)
                v_uncertainty.append(temp_v)

            else:
                M = np.array([[np.sin(az1)*np.cos(el1[i]),np.cos(az1)*np.cos(el1[i])],
                              [np.sin(az2)*np.cos(el2[i]),np.cos(az2)*np.cos(el2[i])]])
                temp = np.linalg.solve(M,np.array([vr[0][i],vr[1][i]]))
                u.append(np.copy(temp[0]))
                v.append(np.copy(temp[1]))
                
                invM = np.linalg.inv(M)
                temp_u = np.sqrt((invM[0,0]**2)*(uncertainty**2) + (invM[0,1]**2)*(uncertainty**2))
                temp_v = np.sqrt((invM[1,0]**2)*(uncertainty**2) + (invM[1,1]**2)*(uncertainty**2))
                u_uncertainty.append(temp_u)
                v_uncertainty.append(temp_v)
                
        return np.array([u,v]),np.array([u_uncertainty,v_uncertainty])
                              
    elif len(vr) == 3:
        u = []
        v = []
        w = []
        u_uncertainty = []
        v_uncertainty = []
        w_uncertainty = []
        el1 = np.deg2rad(elevation[0])
        el2 = np.deg2rad(elevation[1])
        el3 = np.deg2rad(elevation[2])
        az1 = np.deg2rad(azimuth[0])
        az2 = np.deg2rad(azimuth[1])
        az3 = np.deg2rad(azimuth[2])
        for i in range(len(height)):
            if ((np.isnan(vr[0][i]) | np.isnan(vr[1][i]))):
                u.append(np.nan)
                v.append(np.nan)
                
                M = np.array([[np.sin(az1)*np.cos(el1[i]),np.cos(az1)*np.cos(el1[i]),np.sin(el1)],
                              [np.sin(az2)*np.cos(el2[i]),np.cos(az2)*np.cos(el2[i]),np.sin(el2)],
                              [np.sin(az3)*np.cos(el3[i]),np.cos(az3)*np.cos(el3[i]),np.sin(el3)]])
                invM = np.linalg.inv(M)
                temp_u = np.sqrt((invM[0,0]**2)*(uncertainty**2) + (invM[0,1]**2)*(uncertainty**2) + (invM[0,2]**2)*(uncertainty**2))
                temp_v = np.sqrt((invM[1,0]**2)*(uncertainty**2) + (invM[1,1]**2)*(uncertainty**2) + (invM[1,2]**2)*(uncertainty**2))
                temp_w = np.sqrt((invM[2,0]**2)*(uncertainty**2) + (invM[2,1]**2)*(uncertainty**2) + (invM[2,2]**2)*(uncertainty**2))
                
                u_uncertainty.append(temp_u)
                v_uncertainty.append(temp_v)
                w_uncertainty.append(temp_w)
            else:
                M = np.array([[np.sin(az1)*np.cos(el1[i]),np.cos(az1)*np.cos(el1[i]),np.sin(el1)],
                              [np.sin(az2)*np.cos(el2[i]),np.cos(az2)*np.cos(el2[i]),np.sin(el2)],
                              [np.sin(az3)*np.cos(el3[i]),np.cos(az3)*np.cos(el3[i]),np.sin(el3)]])
                temp = np.linalg.solve(M,np.array([vr[0][i],vr[1][i],vr[2][i]]))
                u.append(np.copy(temp[0]))
                v.append(np.copy(temp[1]))
                w.append(np.copy(temp[2]))
                
                invM = np.linalg.inv(M)
                temp_u = np.sqrt((invM[0,0]**2)*(uncertainty**2) + (invM[0,1]**2)*(uncertainty**2) + (invM[0,2]**2)*(uncertainty**2))
                temp_v = np.sqrt((invM[1,0]**2)*(uncertainty**2) + (invM[1,1]**2)*(uncertainty**2) + (invM[1,2]**2)*(uncertainty**2))
                temp_w = np.sqrt((invM[2,0]**2)*(uncertainty**2) + (invM[2,1]**2)*(uncertainty**2) + (invM[2,2]**2)*(uncertainty**2))
                
                u_uncertainty.append(temp_u)
                v_uncertainty.append(temp_v)
                w_uncertainty.append(temp_w)
                
        return np.array([u,v,w]), np.array([u_uncertainty,v_uncertainty,w_uncertainty])
    else:
        print('Input needs to be a length 2 or 3 tuple')
        return np.nan


def lenshow(x, freq=1, tau_min=3, tau_max=12, plot=False):
    '''
    Lenshow correction to calculate vertical velocity variance

    Parameters
    ----------
    x : 1d array
        Timeseries of radial velocity data
    freq : float, optional
        Frequency of the data in Hz. The default is 1.
    tau_min : int, optional
        Sets the minimum lag to use for interpolation of the
        autocorrelation function. The default is 3.
    tau_max : int, optional
        Sets the maximum lag to use for interpolation of the
        autocorrelation function. The default is 12.
    plot : boolean, optional
        If True will plot the autocorrelation and fit. The default is False.

    Returns
    -------
    vvar : float
        Vertical velocity variance
    dif : float
        Difference from fit autocorrelation function at lag 0 from actual
        autocovariance at lag 0   
    '''
    
    # Find the perturbation of x
    mean = np.mean(x)
    prime = x - mean
    # Get the autocovariance 
    acorr, lags = xcorr(prime, prime)
    var = np.var(prime)
    acov = acorr# * var
    # Extract lags > 0
    lags = lags[int(len(lags)/2):] * freq
    acov = acov[int(len(acov)/2):]
    # Define the start and end lags
    lag_start = int(tau_min / freq)
    lag_end = int(tau_max / freq)
    # Fit the structure function
    fit_funct = lambda p, t: p[0] - p[1]*t**(2./3.) 
    err_funct = lambda p, t, y: fit_funct(p, t) - y
    p1, success = leastsq(err_funct, [1, .001], args=(lags[lag_start:lag_end], acov[lag_start:lag_end]))
    if plot:
        new_lags = np.arange(tau_min, tau_max)
        plt.plot(lags, acov)
        plt.plot(new_lags, fit_funct(p1, new_lags), 'gX')
        plt.plot(0, fit_funct(p1, 0), 'gX')
        plt.xlim(0, tau_max+20)
        plt.xlabel("Lag [s]")
        plt.ylabel("$M_{11} [m^2s^{-2}$]")
    return p1[0], acov[0] - p1[0]

    
def lenshow_bonin(x, tau_min=1, tint_first_guess=3, freq=1, max_iter=100, plot=False):
    '''
    Modified Lenshow correction that adaptively selects taus.

    Parameters
    ----------
    x : 1d array
        Timeseries of radial velocity data
    tau_min : int, optional
        Sets the minimum lag to use for interpolation of the
        autocorrelation function. The default is 1.
    tint_first_guess : int, optional
        First guess for adaptively selecting tau. The default is 3.
    freq : float, optional
        Frequency of the data in Hz. The default is 1.
    max_iter : int, optional
       Maximum iterations for calculating best taus. The default is 100.
    plot : boolean, optional
        If True will plot the autocorrelation and fit. The default is False.

    Returns
    -------
    vvar : float
        Vertical velocity variance
    dif : float
        Difference from fit autocorrelation function at lag 0 from actual
        autocovariance at lag 0 
    tau_max : int
        The tau_max that was adaptively selected
    '''
    
    # Find the perturbation of x
    mean = np.mean(x)
    prime = x - mean
    # Get the autocovariance 
    acorr, lags = xcorr(prime, prime)
    var = np.var(prime)
    acov = acorr #* var
    # Extract lags > 0
    lags = lags[int(len(lags)/2):] * freq
    acov = acov[int(len(acov)/2):]
    # Define the start and end lags
    lag_start = int(tau_min / freq)
    lag_end = int((tau_min+3) / freq)
    # Fit the structure function
    fit_funct = lambda p, t: p[0] - p[1]*t**(2./3.) 
    err_funct = lambda p, t, y: fit_funct(p, t) - y
    # Iterate to find t_int
    last_tint = (tau_min+3)
    i = 0
    p1, success = leastsq(err_funct, [.10, .001], args=(lags[lag_start:lag_end], acov[lag_start:lag_end]))
    tint = calc_tint(p1[0], freq, acov, lags)
    while np.abs(last_tint - tint) > 1.:
        if i >= max_iter:
            return None
        else:
            i += 1
            last_tint = tint
        p1, success = leastsq(err_funct, [.10, .001], args=(lags[lag_start:lag_end], acov[lag_start:lag_end]))
        tint = calc_tint(p1[0], freq, acov, lags)
    # Find the time where M11(t) = M11(0)/2
    ind = np.min(np.where(acov <= acov[0]/2))
    # Determine what tau to use
    tau_max = np.min([tint/2., lags[ind]])
    # Do the process
    lag_end = int(tau_max / freq)
    if lag_start+1 >= lag_end:
        lag_end = lag_start + 2
#     print lag_start, lag_end
    p1, success = leastsq(err_funct, [.10, .001], args=(lags[lag_start:lag_end], acov[lag_start:lag_end]))
    if plot:
        new_lags = np.arange(tau_min, tau_max)
        plt.plot(lags, acov, 'k')
        plt.plot(new_lags, fit_funct(p1, new_lags), 'rX', label='Adaptive')
        plt.plot(0, fit_funct(p1, 0), 'rX')
        plt.xlim(0, tau_max+20)
        plt.xlabel("Lag [s]")
        plt.ylabel("$M_{11} [m^2s^{-2}$]")
    return p1[0], np.abs(acov[0] - p1[0]), tau_max


def calc_tint(var, freq, acov):
    '''
    Function that used by lenshow_bonin to calculate tau_max
    
    Parameters
    ----------
    var : float
        Variance
    freq : int
        Frequency of the data tha calculated the variance
    acov : 1d array
        Autocorrelation function

    Returns
    -------
    tau_int : int
        The next tau_max estimate
    '''
    
    ind = np.min(np.where(acov < 0))
    return freq**-1. + 1./var * sum(acov[1:ind] / freq)


def xcorr(y1,y2):
    '''
    Function that calculates the lag correlation of a time series

    Parameters
    ----------
    y1 : 1d array
        First timeseries
    y2 : 1d array
        Second timeseries

    Returns
    -------
    corr : 1d array
        Lag correlations
    lags : int
       Number of lags correlations in corr array

    '''
    if len(y1) != len(y2):
        raise ValueError('The lenghts of the inputs should be the same')
    
    corr = np.correlate(y1,y2,mode='full')
    unbiased_size = np.correlate(np.ones(len(y1)),np.ones(len(y1)),mode='full')
    corr = corr/unbiased_size
    
    maxlags = len(y1)-1
    lags = np.arange(-maxlags,maxlags + 1)
    
    return corr,lags


def process_LidarSim_scan(scan,scantype,elevation,azimuth,ranges,time):
    '''
    A quick function to process data from LidarSim

    Parameters
    ----------
    scan : 3d array
        Radial velocity data from a LidarSim scan
    scantype : str
        Type of scan to process: "VAD", "DBS", or "DBS5"
    elevation : 1d array
        Elevation array for LidarSim scan
    azimuth : 1d array
        Azimuth array for LidarSim scan
    ranges : 1d array
        Range array for LidarSim scan
    time : 1d array
        Time array for LidarSim scan

    Returns
    -------
    VAD or DBS class
        Returns the appropriate class depending on scantype
    '''
    
    if scantype == 'VAD':
        el = np.nanmean(elevation)
        vad = ARM_VAD(scan,ranges,el,azimuth,time)
        
        return vad
    
    elif scantype == 'DBS':
        el = np.nanmean(elevation)
        dbs = Calc_DBS(scan,ranges,el,time=time)
        return dbs
    elif scantype == 'DBS5':
        el = np.nanmean(elevation)
        dbs = Calc_DBS(scan,ranges,el,fifth_beam=True,time=time)
        return dbs
    else:
        print('Not a valid scan type')
        return np.nan
        
        
        
        
    
