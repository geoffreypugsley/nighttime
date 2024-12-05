#%%

## load in the relavant packages

from csat2 import misc, ECMWF, GOES
from csat2.ECMWF.ECMWF import _calc_eis
from csat2.misc import fileops
import numpy as np
from advection_functions.air_parcel import AirParcel25
from advection_functions import plotting
from datetime import datetime, timedelta
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from advection_functions import advection_funcs,GOES_AMSR_colocation,LWP_correction_funcs,plotting
from scipy.stats import binned_statistic_2d,linregress
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
import cartopy.crs as ccrs
import seaborn as sns
from matplotlib.animation import FuncAnimation

#%%

year,month,day_of_month,hour = 2020,1,1,15  ## initial date of interest

start_time = datetime(year,month,day_of_month,hour)    # Start time of simulation

duration =72           # Duration of time of advection in hours
t_step = 0.5                         # Time step of that we sample data at
lon_start, lon_end = 220, 230 #180,260#
lat_start, lat_end = 25, 35 #20, 45 #

# Create arrays of longitudes and latitudes with a separation of 0.25 degrees
lons = np.arange(lon_start, lon_end + 0.25, 0.25)
lats = np.arange(lat_start, lat_end + 0.25, 0.25)

# Create a 2D grid of lon and lat using meshgrid
lon_init, lat_init = np.meshgrid(lons, lats)    # Size of initial domain, in CONUS pixels (i.e. 1 = 2km)
n_trajectories = 1050
time_between_trajectories = timedelta(hours=24)
channel = 7
 ## number of pixels in a given spatial direction, this must be an integer
rho_w = 1000 # density of water in kg/m^3
#size = 5 ## number of pixels overwhich to carry out the CF calculation, note one pixel corresponds to 2km
## number of pixels in a given spatial direction, this must be an integer
domain_dim = lon_init.shape[0]
n_days = 3


winddata = ECMWF.ERA5WindData(level="1000hPa",res="1grid",linear_interp="both") # wind data on 1 degree grid, linearly interpolated in space and time, using finer resolution has led to problems


lat_grid = np.arange(-90, 90, 0.25)  # bins for the 0.25 degree latitude and longitude grid
lon_grid = np.arange(0, 360, 0.25)

#%%

def dqf_filter(dqf_data):
    # Good quality data points have a flag value of 0
    return dqf_data == 0


# make a netCDF file that is all the GOES CONUS data for 2020 at 0.25 degree resolution of 0.25 degree grid


init_time = datetime(2020,1,1,0,0,0)

#end_time = datetime(2020,12,31,23,59,59)
channel = 7

#%%

# create GOES grid

GOES_dir = '/disk1/Data/GOES/Geoff/2020'

file = os.path.join(GOES_dir,'GOES_LWC_2020_25_grid_20200101.nc')

GOES = xr.open_dataset(file)

#print(GOES)

glon = GOES['lon'].values # GOES longitude is in the range 0 to 360
glat = GOES['lat'].values
CSM_25 = GOES['CSM_25'].values
CTP_25 = GOES['CTP_25'].values
CER_25 = GOES['CER_25'].values
COD_25 = GOES['COD_25'].values

print(glon.shape,glat.shape,CSM_25.shape,CTP_25.shape,CER_25.shape,COD_25.shape)


lon_grid,lat_grid = np.meshgrid(glon,glat) # create GOES lon and lat meshgrid

#%%

all_positions = {} #empty nested dictionary for saving the positions of the parcels. The keys are the start time for each trajectory
index = []

## this cell loops over number of trajectories and calculates their subsequent positions a=by advecting them using wind data





for i in range(n_trajectories):

    initial_times = start_time + i * time_between_trajectories ## initial time for each individual parcel, note start time is the start time of all the data and is different to the initial_times, yikes
    parcel = AirParcel25(start_time + i * time_between_trajectories, duration, t_step, lon_init,lat_init, winddata)
    positions = parcel.advect()
    all_positions[parcel.start_time] = positions # save the positions of the parcel in the dictionary

sorted_keys,repeated_keys = advection_funcs.sorted_unique_nested_keys(all_positions) # returns the sorted unique keys of the dictionary, this is used to loop over each unique timestep so that the data is only loaded in once




start_times = all_positions.keys()
positions_arr = advection_funcs.dict_to_numpy(all_positions)

#%%

lon_init = positions_arr[0,0,0,:,:]
lat_init = positions_arr[0,0,1,:,:]
lon_init_vector = lon_init[0,:] # 1d vector of the initial longitudes
lat_init_vector = lat_init[:,0] # 1d vector of the initial latitudes
lon_init_vector_180 = np.where(lon_init_vector>180,lon_init_vector-360,lon_init_vector) # convert longitudes to -180 to 180


# use these indice for the Nd and LWP GOES data
GOES_indices = {
    'dawn': 4, # this corresponds to 8am LST
    'dusk': 24, # this corresponds to 4pm LST NOTE: we cannot choose 0 and 24 for dawn and dusk respectively due to the retrieval bias at these times. Integer steps of 48 will give the following dawn and dusk values (since the data is taken at 30min intervals)
    'terra': 9, #this corresponds to 10:30am LST and is the time of the terra overpass
    'aqua': 15 # this corresponds to 1:30pm LST and is the time of the aqua overpass
}

#%%

# Initialize outputs with dictionaries
GOES_LWP_weighted = {init_time: {} for init_time in start_times}
CF_dict = {init_time: {} for init_time in start_times}
Nd_dict = {init_time: {} for init_time in start_times}
CER_dict = {init_time: {} for init_time in start_times}
EIS_dict = {init_time: {} for init_time in start_times}



for init_time in start_times:
    print(init_time)
    GOES_dir = f'/disk1/Data/GOES/Geoff/{init_time.year}'
    try:
        # Construct file path
        GOES_nc_file = os.path.join(GOES_dir, f'GOES_LWC_{init_time.year}_25_grid_{init_time.strftime("%Y%m%d")}.nc')
        
        # Open GOES dataset
        GOES_data = xr.open_dataset(GOES_nc_file)
        
    except Exception as e:
        # If GOES data doesn't exist or fails to open, populate with NaNs
        for key in all_positions[init_time].keys():
            GOES_LWP_weighted[init_time][key] = np.full((domain_dim,domain_dim),np.nan)
            CF_dict[init_time][key] = np.full((domain_dim,domain_dim),np.nan)
        
        print(f"Failed to load GOES data for {init_time}. Skipping.")
        print(e)
        continue

    # If GOES data is successfully loaded, process it
    for key in all_positions[init_time].keys():
        time = key.strftime("%H%M")
        
        # Convert time to the index used in the netCDF file
        index = int(2 * int(key.strftime("%H")) + int(key.strftime("%M")) / 30)

        # Retrieve parcel positions
        lon = all_positions[init_time][key][0]
        lat = all_positions[init_time][key][1]

        # Calculate GOES data indices
        g_x_index = np.round((lon - glon.min()) * 4).astype(int)  # x index
        g_y_index = np.round((lat - glat.min()) * 4).astype(int)  # y index

        try:
            CF = GOES_data['CF_25'].values[index, g_x_index, g_y_index]
            CER = GOES_data['CER_25'].values[index, g_x_index, g_y_index]
            COD = GOES_data['COD_25'].values[index, g_x_index, g_y_index]

        except Exception as e:
            CF = np.full((domain_dim,domain_dim),np.nan)
            CER = np.full((domain_dim,domain_dim),np.nan)
            COD = np.full((domain_dim,domain_dim),np.nan)
            print('fallen off swath')

        #t1000 = temp_data_1000.get_data(lon,lat,key) # temperature at 1000hPa
        #t700 = temp_data_700.get_data(lon,lat,key) # temperature at 700hPa

        #_,EIS_dict[init_time][key],_ = _calc_eis(t700,t1000) # not totally sure what third output corresponds to

      

        # Calculate the weighted LWP and assign to dictionaries
        GOES_LWP_weighted[init_time][key] = 5 / 9 * CER * COD #* CF 
        CF_dict[init_time][key] = CF
        Nd_dict[init_time][key] = 1.37 * 10 **(-5)*COD**(1/2)*CER**(-5/2)#(0.55*CER+0.44)**(-5/2)
        CER_dict[init_time][key] = CER#(0.55*CER+0.44)

     

#%%

GOES_LWP_weighted_arr = advection_funcs.dict_to_numpy(GOES_LWP_weighted)
CF_arr = advection_funcs.dict_to_numpy(CF_dict)
CER_arr = advection_funcs.dict_to_numpy(CER_dict)
Nd_arr = advection_funcs.dict_to_numpy(Nd_dict)


CF_arr = np.ma.masked_invalid(CF_arr)
Nd_arr = np.ma.masked_invalid(Nd_arr)*10**9
CER_arr = np.ma.masked_invalid(CER_arr)
LWP_arr = np.ma.masked_invalid(GOES_LWP_weighted_arr)

#%%

delta_CF_daytime = {}
delta_CF_nighttime = {}
Nd_dusk = {}
LWP_dusk = {}
EIS_dusk = {}
EIS_dawn = {}
CF_dusk = {}
CF_dawn = {}    
lon_dusk = {}
lat_dusk = {}
MOD_Nd_dusk = {}
lon_aqua = {}
lat_aqua = {}
lon_terra = {}
lat_terra = {}

for i in range(0,n_days-1):  ## this needs to be fixed to account for the fact that the data is not available for the last day
    delta_CF_daytime[f'night_{i}'] =  CF_arr[:,GOES_indices['dusk']+i*48,:,:] - CF_arr[:,GOES_indices['dawn']+(i)*48,:,:]
    delta_CF_nighttime[f'night_{i}'] = CF_arr[:,GOES_indices['dawn']+(i+1)*48,:,:] - CF_arr[:,GOES_indices['dusk']+(i)*48,:,:]
    Nd_dusk[f'night_{i}'] = Nd_arr[:,GOES_indices['dusk']+i*48,:,:]
    LWP_dusk[f'night_{i}'] = LWP_arr[:,GOES_indices['dusk']+i*48,:,:]
    #EIS_dusk[f'night_{i}'] = EIS_arr[:,GOES_indices['dusk']+i*48,:,:]
    #EIS_dawn[f'night_{i}'] = EIS_arr[:,GOES_indices['dawn']+i*48,:,:]
    CF_dusk[f'night_{i}'] = CF_arr[:,GOES_indices['dusk']+i*48,:,:]
    CF_dawn[f'night_{i}'] = CF_arr[:,GOES_indices['dawn']+(i)*48,:,:]
    lon_dusk[f'night_{i}']= positions_arr[:,GOES_indices['dusk']+i*48,0,:,:]
    lat_dusk[f'night_{i}'] = positions_arr[:,GOES_indices['dusk']+i*48,1,:,:]
    lon_aqua[f'night_{i}'] = positions_arr[:,GOES_indices['aqua']+i*48,0,:,:]
    lat_aqua[f'night_{i}'] = positions_arr[:,GOES_indices['aqua']+i*48,1,:,:]
    lon_terra[f'night_{i}'] = positions_arr[:,GOES_indices['terra']+i*48,0,:,:]
    lat_terra[f'night_{i}'] = positions_arr[:,GOES_indices['terra']+i*48,1,:,:]



#%%

MODIS_sats = ['aqua', 'terra']  # Aqua overpasses in the afternoon and Terra in the morning
MOD_Nd = {}
MOD_polluted_mask = {}
nights = ['night_0','night_1','night_2']

GOES_lons_MOD_op_time = {'aqua': lon_aqua, 'terra': lon_terra}
GOES_lats_MOD_op_time = {'aqua': lat_aqua, 'terra': lat_terra}

for satellite in MODIS_sats:
    MODIS_Nd_total = []
    
    for init_time in start_times:
        print(init_time)
        # Read in the MODIS data for the current day and satellite
        try:
            MODIS_data = csat2.MODIS.readin('cdnc_best', year=init_time.year, doy=misc.time.date_to_doy(init_time.year,init_time.month,init_time.day)[1], sds=['Nd_G18'], sat=satellite) ## load in the MODIS data for the current day and satellite
            Nd = MODIS_data.Nd_G18 #droplet number concentration
            MODIS_lon = MODIS_data.lon #lon for the day
            MODIS_lat = MODIS_data.lat # latitude for the day
        
            # Initialize a list to hold Nd values for each (lon, lat) pair
            MODIS_Nd_pairs = []
        
            # Loop through each (lon, lat) pair
            for lon, lat in zip(advection_funcs.normalize_longitude_180(GOES_lons_MOD_op_time[satellite]['night_0'][i-1]).flatten(),GOES_lats_MOD_op_time[satellite]['night_0'][i-1].flatten()): # it is i-1 because the index starts at 0
            # Select the nearest Nd value for each (lon, lat) pair
                MODIS_Nd_value = Nd.sel(lon=lon, lat=lat, method='nearest').values[0]  # Select 0th element to avoid extra dimensions ## finds the nearest Nd value for a given lat and lon
                MODIS_Nd_pairs.append(MODIS_Nd_value) # Append the Nd value to the list, loops over every lon lat value
        
            # Append the list of Nd values for this day to the total
            MODIS_Nd_total.append(MODIS_Nd_pairs) ## for each time step appends the Nd values for all the lon lat values
        except:
            print(f'missing MODIS Nd data for {satellite} on {init_time}')
            MODIS_Nd_total.append([np.nan]*len(GOES_lons_MOD_op_time[satellite]['night_0'][i-1].flatten())) ## if the data is missing append a list of nans
    
    # Store the results for each satellite
    MOD_Nd[satellite] = np.array(MODIS_Nd_total)    

#%%

## reshape the array
MOD_Nd_aqua =np.ma.masked_invalid(MOD_Nd['aqua'].reshape((n_trajectories,domain_dim,domain_dim)))

print(MOD_Nd_aqua.shape)
GOES_aqua = Nd_arr[:,GOES_indices['aqua'],:,:]

MOD_Nd_terra = np.ma.masked_invalid(MOD_Nd['terra'].reshape((n_trajectories,domain_dim,domain_dim)))
GOES_Nd_terra = Nd_arr[:,GOES_indices['terra'],:,:]

delta_CF_day_0 = delta_CF_daytime['night_0']
delta_CF_night_0 = delta_CF_nighttime['night_0']
CF_dusk_night_0 = CF_dusk['night_0']
CF_dawn_night_0 = CF_dawn['night_0']
#EIS_dusk_night_0 = EIS_dusk['night_0']
#EIS_dawn_night_0 = EIS_dawn['night_0']

#%%


## function to convert the delta CF, CF and Nd arrays into xarrays and calcualte some useful statistics

def delta_CF_x_array(delta_CF,CFi,Ndi,n_CFi_bins=50,n_Ndi_bins=50):


    delta_CF_flat,CFi_flat,Ndi_flat = advection_funcs.combine_masked_arrays(delta_CF,CFi,Ndi) ## flatten arrays and remove any nans common to both
    CF_bin_edges = np.linspace(0,1, n_CFi_bins) # needs to be 11 since we have 10 bins
    CF_bin_centres = (CF_bin_edges[:-1] + CF_bin_edges[1:]) / 2
    breakpoints=[10, 30, 100, 300,600]
    num_Nd_bins = 50

    Nd_bin_edges = np.concatenate([
        np.logspace(np.log10(breakpoints[i]), np.log10(breakpoints[i + 1]), num_Nd_bins // (len(breakpoints) - 1) + 1)[:-1]
        for i in range(len(breakpoints) - 1)
    ])
    Nd_bin_edges = np.append(Nd_bin_edges, breakpoints[-1]) ## these are the bin edges for the Nd bins
    Nd_bin_centres = (Nd_bin_edges[:-1] + Nd_bin_edges[1:]) / 2


    mean_delta_CF = np.full((len(CF_bin_centres), len(Nd_bin_centres) ), np.nan) # #initalize arrays
    CF_sensitivity_to_Nd = np.full((len(CF_bin_centres), len(Nd_bin_centres) ), np.nan)
    counts = np.full((len(CF_bin_centres) , len(Nd_bin_centres)), np.nan)
    delta_CF_CFi_signal = np.full((len(CF_bin_centres) , len(Nd_bin_centres)), np.nan)

    CF_indices = np.digitize(CFi_flat, CF_bin_edges) #-1  # Bin indices for CF
    MOD_indices = np.digitize(Ndi_flat, Nd_bin_edges)  #-1# Bin indices for MOD

    # Calculate the mean delta_CF_night_0 for each bin
    for i in range(len(CF_bin_centres) ):
        for j in range(len(Nd_bin_centres) ):
            # Mask for current bin
            mask = (CF_indices == i+1) & (MOD_indices == j+1) ## the +1 is to account for the fact that for np.digitize an index of 0 cirresponds to less than the 1st bin edge
            counts[i, j] = np.sum(mask)
            mean_delta_CF[i, j] = np.nanmean(delta_CF_flat[mask])
            #print(counts[i,j])
            try:  # Check if there are data points in the bin
                   
                CF_sensitivity_to_Nd[i,j] = linregress(Ndi_flat[mask].flatten(),delta_CF_flat[mask].flatten())[0]
                #print( slope,intercept)
            except:
                #print('no Nd slope')
                CF_sensitivity_to_Nd[i,j] = np.nan
            try:
                CF_sensitivity_to_CFi = linregress(CFi_flat[mask].flatten(),delta_CF_flat[mask].flatten()) ## calculate linear regression for CF vs delta CF to pick out signal from initial CF
                slope,intercept,_,_,_, = CF_sensitivity_to_CFi
                delta_CF_CFi_signal[i,j] = np.nanmean(CF_sensitivity_to_CFi[0]*CFi_flat[mask].flatten()+CF_sensitivity_to_CFi[1]) #contribution of the singal from CF_initial 
            except:
                delta_CF_CFi_signal[i,j] = np.nan
                #print('no CF slope')
                

    mean_delta_CF_given_Nd = np.nanmean(mean_delta_CF,axis=1)

    mean_delta_CF_given_Nd_repeated = np.repeat(np.expand_dims(mean_delta_CF_given_Nd, axis=-1), mean_delta_CF.shape[1], axis=-1)
    delta_CF_Nd_signal_M1 = mean_delta_CF - mean_delta_CF_given_Nd_repeated
    delta_CF_Nd_signal_M2 = mean_delta_CF - delta_CF_CFi_signal

    delta_CF_xarray = xr.Dataset(
        data_vars={
            'mean_delta_CF': (('CFi_bin', 'Ndi_bin'), mean_delta_CF),
            'CF_sensitivity_to_Nd': (('CFi_bin', 'Ndi_bin'), CF_sensitivity_to_Nd),
            'counts': (('CFi_bin', 'Ndi_bin'), counts),
            'delta_CF_Nd_signal_M1': (('CFi_bin', 'Ndi_bin'), delta_CF_Nd_signal_M1),
            'delta_CF_Nd_signal_M2': (('CFi_bin', 'Ndi_bin'), delta_CF_Nd_signal_M2),
        },
        coords={
            'CFi_bin': CF_bin_centres[:],
            'Ndi_bin': Nd_bin_centres[:],
        },
        attrs={
            'description': 'Mean delta_CF with sensitivity and counts per bin'
     }
    )

    return delta_CF_xarray

day_arr= delta_CF_x_array(delta_CF_day_0,CF_dawn_night_0,MOD_Nd_terra,n_CFi_bins=50,n_Ndi_bins=50)
night_arr = delta_CF_x_array(delta_CF_night_0,CF_dusk_night_0,MOD_Nd_aqua,n_CFi_bins=50,n_Ndi_bins=50)

#%%

# save the outputs as netCDF files

day_arr.to_netcdf('delta_CF_outputs/day_arr.nc')
night_arr.to_netcdf('delta_CF_outputs/night_arr.nc')