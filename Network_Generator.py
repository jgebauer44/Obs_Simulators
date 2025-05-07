import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import glob as glob
import pyproj
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("reference_file", help="Name of the namelist file (string)")
parser.add_argument("output_dir",help="Output directory for the network coordinates file")
parser.add_argument("site_spacing",type=int, help="Average spacing of the observation locations in m")
parser.add_argument("scatter_distance",type=int,help="Maximum distance a site can be displaced from the average spacing in m")
parser.add_argument("network_name",type=str,help="Name of the network for filename")

args = parser.parse_args()

file = args.reference_file
output_dir = args.output_dir
site_dx = args.site_spacing
scatter = args.scatter_distance
netname = args.network_name

fid = Dataset(file)

if fid.MAP_PROJ == 1:
    wrf_proj = pyproj.Proj(proj='lcc',lat_1 = fid.TRUELAT1, lat_2 = fid.TRUELAT2, lat_0 = fid.MOAD_CEN_LAT, lon_0 = fid.STAND_LON, a = 6370000, b = 6370000)
    wgs_proj = pyproj.Proj(proj='latlong',datum='WGS84')
    transformer = pyproj.Transformer.from_proj(wgs_proj,wrf_proj)

e, n = transformer.transform(fid.CEN_LON, fid.CEN_LAT)
dx,dy = fid.DX, fid.DY
nx, ny = fid.dimensions['west_east'].size, fid.dimensions['south_north'].size

Lx = nx*dx
Ly = ny*dy

new_nx = int(Lx/site_dx)
new_ny = int(Ly/site_dx)

x0 = -(new_nx-1) / 2. * site_dx + e
y0 = -(new_ny-1) / 2. * site_dx + n
x_grid = np.arange(new_nx) * site_dx + x0
y_grid = np.arange(new_ny) * site_dx + y0
xx, yy = np.meshgrid(x_grid, y_grid)

rng = np.random.default_rng()

r_e = rng.random(xx.shape)
r_e_sign = rng.random(xx.shape)
r_n = rng.random(xx.shape)
r_n_sign = rng.random(xx.shape)

r_e_sign = np.where(r_e_sign < 0.5,-1,1)
r_n_sign = np.where(r_n_sign < 0.5,-1,1)

final_x = xx + r_e_sign*r_e*scatter
final_y = yy + r_n_sign*r_n*scatter

transformer2 = pyproj.Transformer.from_proj(wrf_proj,wgs_proj)

final_lon,final_lat = transformer2.transform(final_x,final_y)

final_lon = final_lon.ravel()
final_lat = final_lat.ravel()

filename = output_dir +'/' + netname + '.txt'

f = open(filename,"w")

for i in range(len(final_lon)):
    f.write("%20.14f %20.14f \n"%(final_lat[i],final_lon[i]))

f.close()