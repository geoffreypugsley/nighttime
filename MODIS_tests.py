# %%


from csat2 import misc, ECMWF, GOES
from csat2.ECMWF.ECMWF import _calc_eis
from csat2.misc import fileops
import numpy as np
from advection_functions.air_parcel import AirParcel25
from datetime import datetime, timedelta
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from advection_functions import advection_funcs,GOES_AMSR_colocation,LWP_correction_funcs,plotting
from scipy.stats import binned_statistic_2d
import xarray as xr
from netCDF4 import Dataset
import pvlib
import RSS
import cftime
import pandas as pd
import os
import matplotlib.patches as mpatches
import xarray as xr
from matplotlib import gridspec
from matplotlib.colors import LogNorm
import csat2.MODIS
import logging



# %%

gran = csat2.MODIS.Granule.fromtext('2015011.2110A')




# %%
gran.download('06_L2')

# %%
#cer = gran.get_variable('06_L2', ['Cloud_Effective_Radius'])
#print(cer)


# %%
cdnc = gran.get_cdnc(bowtie_corr=False)

lon,lat = gran.get_lonlat() 

print(cdnc.shape)
print(lon.shape)

# %%
data = csat2.MODIS.readin('cdnc_best', year=2020, doy=100,sds =['Nd_G18'])

Nd = data['Nd_G18']
lon = data['lon']
lat = data['lat']
time = data['time']

plt.imshow(Nd.T,origin='lower',extent=[lon.min(),lon.max(),lat.min(),lat.max()])
plt.colorbar()
plt.show()

# %%
plt.plot(lat)
plt.show()

# %%
