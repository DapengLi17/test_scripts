# === readme ===
# descrip: plot to compare volume transports along the Florida Strait between CESM-POP diagnostics outputs and my own calculation by postprocessing CESM-POP output UVEL

# update history: 
# v1.0 DL 2021Oct20 
# v1.1 DL 2021Oct22 correct for partial bottom cell following Steve's POP_MOC (https://github.com/sgyeager/POP_MOC/blob/main/pop_moc_0p1deg.py)
# v1.2 DL 2021Oct24 compare with CESM-POP diagnostics outputs 
# v1.3 DL 2021Oct26 use Sanjiv's submeso FOSI runs 

# extra notes:
# CESM-POP subroutine diag_transport line# 2010-2255 at https://github.com/ESCOMP/POP2-CESM/blob/master/source/diagnostics.F90 <br>
# Channel index (https://ncar.github.io/POP/doc/build/html/users_guide/model-diagnostics-and-output.html)
# ==============


# === import modules ===
# general python packages
import numpy as np
# from scipy import stats
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
%matplotlib inline
import proplot as plot
# import cartopy
# import cartopy.crs as ccrs
# cartopy.config['pre_existing_data_dir']='/ihesp/shared/cartopy_features'
# from cartopy.mpl.geoaxes import GeoAxes
# from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
# from mpl_toolkits.axes_grid1 import AxesGrid
import glob
import cftime
import util

# python GCM(POP) packages
# import xgcm
# import pop_tools
# import xoak
# import xesmf as xe

# dask jupyter lab packages
from dask.distributed import Client
# from dask.distributed import performance_report

# file name with time packages
# from itertools import product
# from cftime import DatetimeNoLeap
# ======================


# === incorporate dask ===
#client = Client("tcp://10.73.1.1:36170")
#client
# ========================


# === define parameters ===
# --- t12 ---
# FSm: Florida Strait Meridional section
# 292  292 1418 1443    1   42 merid  Florida Strait 
# reference: https://ncar.github.io/POP/doc/build/html/users_guide/model-diagnostics-and-output.html
ilon1_FSm_t12 = 292-1 
ilat1_FSm_t12, ilat2_FSm_t12 = 1418-1-1, 1443 
# -1 is because python index starts from 0 while Fortran index starts from 1
# CESM-POP subroutine diag_transport line # 2010-2255 at https://github.com/ESCOMP/POP2-CESM/blob/master/source/diagnostics.F90
# compute averages along nlat dim, so I read one point ahead of 1418, the two sides are on the lands (KMU=0, see codes below)
# -----------
# =========================


# === read files ===
# CESM-POP UVEL HR
infile_uvel= ('/scratch/user/sanjiv/shared/dapeng/cmpr_submeso_5km.pop.h.0277-01.nc')

ds_uvel = xr.open_dataset(infile_uvel)
ds_uvel

# CESM-POP diagnostic transport (dt)
infile_dt = ('/scratch/user/sanjiv/shared/dapeng/submeso_5km.pop.dt.0277-01-01-00000')
colnames=['time', 'Qv', 'Qh', 'Qs','channel_name'] 
df = pd.read_csv(infile_dt, sep='  ', engine='python',  usecols=[0,1,2,3,4], 
                header=None, skiprows=[0], names=colnames, index_col=[0]) 
df
# ==================



# === data analysis ===
# --- Compute 3D DZU with partial bottom cell corrected, copied from Steve's github ---
# (https://github.com/sgyeager/POP_MOC/blob/main/pop_moc_0p1deg.py) 
# and commented a few lines. Need a POP history file for 3D DZU
in_file = infile_uvel

ds = xr.open_dataset(in_file)
pd     = ds['PD']
pd=pd.drop(['ULAT','ULONG'])            # this is a python bug that we are correcting
temp   = ds['TEMP']
temp=temp.drop(['ULAT','ULONG'])
# salt   = ds['SALT']
# salt=salt.drop(['ULAT','ULONG'])
# u_e   = ds['UVEL']/100
# u_e=u_e.drop(['TLAT','TLONG'])
# u_e.attrs['units']='m/s'
# v_e   = ds['VVEL']/100
# v_e=v_e.drop(['TLAT','TLONG'])
# v_e.attrs['units']='m/s'
# w_e   = ds['WVEL']/100
# w_e=w_e.drop(['ULAT','ULONG'])
# w_e.attrs['units']='m/s'
ulat   = ds['ULAT']
ulon   = ds['ULONG']
tlat   = ds['TLAT']
tlon   = ds['TLONG']
kmt  = ds['KMT']
kmt.values[np.isnan(kmt.values)]=0	# get rid of _FillValues
kmu  = ds['KMU']
kmu.values[np.isnan(kmu.values)]=0	# get rid of _FillValues
dxu    = ds['DXU']/100
dxu.attrs['units']='m'
dyu    = ds['DYU']/100
dyu.attrs['units']='m'
rmask  = ds['REGION_MASK']
tarea  = ds['TAREA']/100/100
tarea.attrs['units']='m^2'
uarea  = ds['UAREA']/100/100
uarea.attrs['units']='m^2'
time   = ds['time']
time.encoding['_FillValue']=None    
z_t   = ds['z_t']/100
z_t.attrs['units']='m'
z_w   = ds['z_w']/100
z_w.attrs['units']='m'
z_w_bot   = ds['z_w_bot']/100
z_w_bot.attrs['units']='m'
dz   = ds['dz']/100
dz.attrs['units']='m'
dzw   = ds['dzw']/100
dzw.attrs['units']='m'
hu   = ds['HU']/100
hu.attrs['units']='m'
ht   = ds['HT']/100
ht.attrs['units']='m'
dims = np.shape(temp)
nt = dims[0]
nz = dims[1]
ny = dims[2]
nx = dims[3]
km = int(np.max(kmt).values)
mval=pd.encoding['_FillValue']

# Create a k-index array for masking purposes
kji = np.indices((nz,ny,nx))
kindices = kji[0,:,:,:] + 1

# Define top/bottom depths of POP T-grid
z_bot=z_w.values
z_bot=z_w.values+dz.values
z_top=z_w.values

# Compute PBC from grid info:
dzt = util.pbc_dzt(dz,kmt,ht,z_w_bot,mval)

# Regrid PBC thicknesses to U-grid
tmp=dzt
tmpe=tmp.roll(nlon=-1,roll_coords=False)        # wraparound shift to west, without changing coords
tmpn=tmp.shift(nlat=-1)                         # shift to south, without changing coords
tmpne=tmpn.roll(nlon=-1,roll_coords=False)      # wraparound shift to west, without changing coords
tmpall=xr.concat([tmp,tmpe,tmpn,tmpne],dim='dummy')
dzu=tmpall.min('dummy')
dzu.attrs['units'] = 'm'
del tmp,tmpe,tmpn,tmpne,tmpall

dzu # 3D DZU with partial bottom cell corrected

# --- compute kmu ---
kmu_FSm = ds_uvel.KMU.isel(nlon=ilon1_FSm_t12, 
                           nlat=slice(ilat1_FSm_t12, ilat2_FSm_t12)).values
kmu_FSm # two sides are land points, kMU=0

# --- compute dyu ---
dyu_FSm = ds_uvel.DYU.isel(nlon=ilon1_FSm_t12, 
                           nlat=slice(ilat1_FSm_t12, ilat2_FSm_t12)).values/100 # unit: m 
dyu_FSm

# --- compute dzu ---
dzu_FSm = dzu.isel(nlon=ilon1_FSm_t12, 
                   nlat=slice(ilat1_FSm_t12, ilat2_FSm_t12))
dzu_FSm

# --- extract uvel ---
u_FSm = ds_uvel.UVEL.isel(nlon=ilon1_FSm_t12, nlat=slice(ilat1_FSm_t12, ilat2_FSm_t12)
                    ).compute().where(kmu_FSm>0, np.nan) # mask land points 
u_FSm.plot(cmap='jet') # land points are nan
u_FSm

# --- compute flux Q on individual grid point ---
Q_FSm = (u_FSm/100*dyu_FSm*dzu_FSm).fillna(0) # Q is flux, fill land with 0 before computing sum
Q_FSm

# --- compute flux Q summed over the section ---
Q_FSm_av = (Q_FSm[:,:,0:-1] + Q_FSm[:,:,1:])/2
# compute mean along nlat dim following line 2103-2109 at https://github.com/ESCOMP/POP2-CESM/blob/master/source/diagnostics.F90
Q_FSm_sum = Q_FSm_av.sum('z_t').sum('nlat')/1e6 # convert unit to Sv
Q_FSm_sum # Flux from my postprocessing scripts

# --- convert POP diagnostic transport from panda data frame to xarray --- 
nchannel = 141
ds1_FSm = df.iloc[1::nchannel,:].to_xarray() # Florida Strait
ds1_FSm

ds1_FSm.Qv.isel(time=slice(0,31)) # the first 31 days for Jan mean

ds1_FSm.Qv.isel(time=slice(0,31)).mean('time') # Jan mean diagnostics transport
# =====================