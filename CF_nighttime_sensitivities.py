# %% load in the relavant packages


## load in the relavant packages

from csat2 import misc, ECMWF, GOES
from csat2.ECMWF.ECMWF import _calc_eis
from csat2.misc import fileops
import numpy as np
from advection_functions.air_parcel import AirParcel25
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
import itertools



# %%
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
n_trajectories = 360
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




# %%
def dqf_filter(dqf_data):
    # Good quality data points have a flag value of 0
    return dqf_data == 0


# make a netCDF file that is all the GOES CONUS data for 2020 at 0.25 degree resolution of 0.25 degree grid


init_time = datetime(2020,1,1,0,0,0)

end_time = datetime(2020,12,31,23,59,59)
channel = 7

# %%
GOES_dir = '/disk1/Data/GOES/Geoff/LWC'

file = os.path.join(GOES_dir,'GOES_LWC_2020_25_grid_20200101.nc')

GOES = xr.open_dataset(file)

#print(GOES)

glon = GOES['lon'].values
glat = GOES['lat'].values
CSM_25 = GOES['CSM_25'].values
CTP_25 = GOES['CTP_25'].values
CER_25 = GOES['CER_25'].values
COD_25 = GOES['COD_25'].values

print(glon.shape,glat.shape,CSM_25.shape,CTP_25.shape,CER_25.shape,COD_25.shape)


lon_grid,lat_grid = np.meshgrid(glon,glat) # create GOES lon and lat meshgrid


# %%
all_positions = {} #empty nested dictionary for saving the positions of the parcels. The keys are the start time for each trajectory
index = []

## this cell loops over number of trajectories and calculates their subsequent positions a=by advecting them using wind data





for i in range(n_trajectories):

    initial_times = start_time + i * time_between_trajectories ## initial time for each individual parcel, note start time is the start time of all the data and is different to the initial_times, yikes
    


    

    #resa

    
    #except IndexError:
    #    print("No matching file found. Skipping lon-lat extraction.")

    #try:
    #    GOES_index_init_point = np.array(gran.locate(np.array([init_point]), channel = channel))[0] 
    #except IndexError:
    #    print("No matching file found. Skipping GOES index extraction.")

    parcel = AirParcel25(start_time + i * time_between_trajectories, duration, t_step, lon_init,lat_init, winddata)
    
    #index.append(GOES_index_init_point)
    positions = parcel.advect()
    all_positions[parcel.start_time] = positions # save the positions of the parcel in the dictionary

   

sorted_keys,repeated_keys = advection_funcs.sorted_unique_nested_keys(all_positions) # returns the sorted unique keys of the dictionary, this is used to loop over each unique timestep so that the data is only loaded in once




start_times = all_positions.keys()
positions_arr = advection_funcs.dict_to_numpy(all_positions)

# %%




lon_init = positions_arr[0,0,0,:,:]
lat_init = positions_arr[0,0,1,:,:]
lon_init_vector = lon_init[0,:] # 1d vector of the initial longitudes
lat_init_vector = lat_init[:,0] # 1d vector of the initial latitudes
lon_init_vector_180 = np.where(lon_init_vector>180,lon_init_vector-360,lon_init_vector) # convert longitudes to -180 to 180

# %%
# use these indice for the Nd and LWP GOES data
GOES_indices = {
    'dawn': 4, # this corresponds to 8am LST
    'dusk': 20, # this corresponds to 4pm LST NOTE: we cannot choose 0 and 24 for dawn and dusk respectively due to the retrieval bias at these times. Integer steps of 48 will give the following dawn and dusk values (since the data is taken at 30min intervals)
    'terra': 9, #this corresponds to 10:30am LST and is the time of the terra overpass
    'aqua': 15 # this corresponds to 1:30pm LST and is the time of the aqua overpass
}

# %%

MODIS_sats = ['aqua','terra'] # aqua overpasses in the afternoon and terra in the morning
MOD_Nd = {}
MOD_polluted_mask = {}
for satellite in MODIS_sats:
    MODIS_Nd_total = []
    for i in range(1,n_trajectories+1):
        #print(i)
        MODIS_data = csat2.MODIS.readin('cdnc_best', year=2020, doy=i,sds =['Nd_G18'],sat = satellite)
        Nd = MODIS_data.Nd_G18
        MODIS_lon = MODIS_data.lon
        MODIS_lat = MODIS_data.lat
        time = MODIS_data.time
        MODIS_Nd = Nd.sel(lon = lon_init_vector_180.flatten(),lat = lat_init_vector.flatten(),method = 'nearest').values[0] # since time is also a dimension we need to select the 0th element to avoid carrying extra stuff around
        MODIS_Nd_total.append(MODIS_Nd)
    
    MOD_Nd[satellite] = np.ma.masked_invalid(MODIS_Nd_total)

    MODIS_Nd_array = np.ma.masked_invalid(MODIS_Nd_total)

    median_MODIS_Nd = np.ma.median(MODIS_Nd_array)

    MODIS_polluted = MODIS_Nd_array > median_MODIS_Nd ## true if the MODIS data is polluted i.e. higher droplet number concentration than the median

    MODIS_polluted_expand_dims = np.expand_dims(MODIS_polluted,axis = 1)

    MOD_polluted_mask[satellite] = np.repeat(MODIS_polluted_expand_dims,positions_arr.shape[1],axis = 1)
    

    # %%

    temp_data_1000 = ECMWF.ERA5Data('Temperature', level='1000hPa', res='1grid')
temp_data_700 = ECMWF.ERA5Data('Temperature', level='700hPa', res='1grid')

LTS_initial = np.zeros((n_trajectories,domain_dim,domain_dim)) #shape is (n_trajectories, x_coord, y_coord)
EIS_initial = np.zeros((n_trajectories,domain_dim,domain_dim))
other = np.zeros((n_trajectories,domain_dim,domain_dim))

for i in range(n_trajectories):
    #initial_times.append(init_time + i * time_between_trajectroies)

    time_current = start_time + i * time_between_trajectories

    t1000 = temp_data_1000.get_data(positions_arr[0,0,0,:,:],positions_arr[0,0,1,:,:],time_current) # temperature at 1000hPa
    t700 = temp_data_700.get_data(positions_arr[0,0,0,:,:],positions_arr[0,0,1,:,:],time_current) # temperature at 700hPa

    LTS_initial[i],EIS_initial[i],other[i] = _calc_eis(t700,t1000) # not totally sure what third output corresponds to
    

#indices_transpose = np.transpose(indices,(1,0,2,3))

median_EIS = np.median(EIS_initial.flatten())

LTS_stable = EIS_initial > median_EIS # true when LTS is larger than the median array

# Expand the mask to match the shape of array_to_mask
expanded_mask = np.expand_dims(LTS_stable, axis=0)  # Shape becomes (1, 2, 50, 50)
expanded_mask = np.repeat(expanded_mask, positions_arr.shape[1], axis=0)  # Shape becomes (49, 2, 50, 50)


LTS_stable = expanded_mask

# %%

# Initialize outputs with dictionaries
GOES_LWP_weighted = {init_time: {} for init_time in start_times}
CF_dict = {init_time: {} for init_time in start_times}
Nd_dict = {init_time: {} for init_time in start_times}
CER_dict = {init_time: {} for init_time in start_times}
EIS_dict = {init_time: {} for init_time in start_times}
EIS_dict = {init_time: {} for init_time in start_times}  # dictionary for LTS


for init_time in start_times:
    try:
        # Construct file path
        GOES_nc_file = os.path.join(GOES_dir, f'GOES_LWC_2020_25_grid_{init_time.strftime("%Y%m%d")}.nc')
        
        # Open GOES dataset
        GOES_data = xr.open_dataset(GOES_nc_file)
        
    except Exception as e:
        # If GOES data doesn't exist or fails to open, populate with NaNs
        for key in all_positions[init_time].keys():
            GOES_LWP_weighted[init_time][key] = np.full((domain_dim,domain_dim),np.nan)
            CF_dict[init_time][key] = np.full((domain_dim,domain_dim),np.nan)
        
        # Skip to the next init_time since data isn't available
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

        t1000 = temp_data_1000.get_data(lon,lat,key) # temperature at 1000hPa
        t700 = temp_data_700.get_data(lon,lat,key) # temperature at 700hPa

        EIS_dict[init_time][key],_,_ = _calc_eis(t700,t1000) # not totally sure what third output corresponds to

      

        # Calculate the weighted LWP and assign to dictionaries
        GOES_LWP_weighted[init_time][key] = 5 / 9 * CER * COD #* CF 
        CF_dict[init_time][key] = CF
        Nd_dict[init_time][key] = 1.37 * 10 **(-5)*COD**(1/2)*CER**(-5/2)
        CER_dict[init_time][key] = CER

# %%

GOES_LWP_weighted_arr = advection_funcs.dict_to_numpy(GOES_LWP_weighted)
CF_arr = advection_funcs.dict_to_numpy(CF_dict)
CER_arr = advection_funcs.dict_to_numpy(CER_dict)
Nd_arr = advection_funcs.dict_to_numpy(Nd_dict)
EIS_arr = advection_funcs.dict_to_numpy(EIS_dict)
LTS_stable_arr = np.transpose(LTS_stable,(1,0,2,3))

CF_arr = np.ma.masked_invalid(CF_arr)
Nd_arr = np.ma.masked_invalid(Nd_arr)*10**9
CER_arr = np.ma.masked_invalid(CER_arr)
LWP_arr = np.ma.masked_invalid(GOES_LWP_weighted_arr)

# %%

print(CF_arr.shape)
delta_CF = {}
Nd_dusk = {}
LWP_dusk = {}
EIS_dusk = {}
CF_dusk = {}
lon_dusk = {}
lat_dusk = {}
MOD_Nd_dusk = {}
lon_aqua = {}
lat_aqua = {}

for i in range(0,n_days):
    delta_CF[f'night_{i}'] = CF_arr[:,GOES_indices['dusk']+i*48,:,:] - CF_arr[:,GOES_indices['dawn']+i*48,:,:]
    Nd_dusk[f'night_{i}'] = Nd_arr[:,GOES_indices['dusk']+i*48,:,:]
    LWP_dusk[f'night_{i}'] = LWP_arr[:,GOES_indices['dusk']+i*48,:,:]
    EIS_dusk[f'night_{i}'] = EIS_arr[:,GOES_indices['dusk']+i*48,:,:]
    CF_dusk[f'night_{i}'] = CF_arr[:,GOES_indices['dusk']+i*48,:,:]
    lon_dusk[f'night_{i}']= positions_arr[:,GOES_indices['dusk']+i*48,0,:,:]
    lat_dusk[f'night_{i}'] = positions_arr[:,GOES_indices['dusk']+i*48,1,:,:]
    lon_aqua[f'night_{i}'] = positions_arr[:,GOES_indices['aqua']+i*48,0,:,:]
    lat_aqua[f'night_{i}'] = positions_arr[:,GOES_indices['aqua']+i*48,1,:,:]


# %%

MODIS_sats = ['aqua', 'terra']  # Aqua overpasses in the afternoon and Terra in the morning
MOD_Nd = {}
MOD_polluted_mask = {}

for satellite in MODIS_sats:
    MODIS_Nd_total = []
    
    for i in range(1, n_trajectories + 1):
        # Read in the MODIS data for the current day and satellite
        MODIS_data = csat2.MODIS.readin('cdnc_best', year=2020, doy=i, sds=['Nd_G18'], sat=satellite)
        Nd = MODIS_data.Nd_G18
        MODIS_lon = MODIS_data.lon
        MODIS_lat = MODIS_data.lat
        
        # Initialize a list to hold Nd values for each (lon, lat) pair
        MODIS_Nd_pairs = []
        
        # Loop through each (lon, lat) pair
        for lon, lat in zip(lon_aqua['night_0'][i-1].flatten(),lat_aqua['night_0'][i-1].flatten()):
            # Select the nearest Nd value for each (lon, lat) pair
            MODIS_Nd_value = Nd.sel(lon=lon, lat=lat, method='nearest').values[0]  # Select 0th element to avoid extra dimensions
            MODIS_Nd_pairs.append(MODIS_Nd_value)
        
        # Append the list of Nd values for this day to the total
        MODIS_Nd_total.append(MODIS_Nd_pairs)
    
    # Store the results for each satellite
    MOD_Nd[satellite] = np.array(MODIS_Nd_total)



# %%

MOD_Nd_aqua =np.ma.masked_invalid(MOD_Nd['aqua'].reshape((n_trajectories,domain_dim,domain_dim)))

## calculate the correlation between the change in cloud fraction and the droplet number concentration for different EIS, LWP bins

Nd_dusk_arr = Nd_dusk['night_0']
delta_CF_arr = delta_CF['night_0']
EIS_dusk_arr = EIS_dusk['night_0']
LWP_dusk_arr = LWP_dusk['night_0']
CF_dusk_arr = CF_dusk['night_0']

# %%

MOD_Nd_aqua,Nd_dusk_flat,delta_CF_flat,EIS_dusk_flat,LWP_dusk_flat,CF_dusk_flat = advection_funcs.combine_masked_arrays(MOD_Nd_aqua,Nd_dusk_arr,delta_CF_arr,EIS_dusk_arr,LWP_dusk_arr,CF_dusk_arr)


# %%

def calculate_slope_vs_bins(x_arr, y_arr, bin1_arr, bin2_arr, num_bin1_bins=10, num_bin2_bins=10, 
                             percentile_bins=True):
    """
    Calculate the slope of y_arr vs. x_arr in bin1_arr and bin2_arr bins.

    Parameters:
    x_arr : 1D array
        Array of values for x-axis (independent variable).
    y_arr : 1D array
        Array of values for y-axis (dependent variable).
    bin1_arr : 1D array
        Array for the first binning variable.
    bin2_arr : 1D array
        Array for the second binning variable.
    num_bin1_bins : int
        Number of bins for the first binning variable.
    num_bin2_bins : int
        Number of bins for the second binning variable.
    percentile_bins : bool
        If True, bins will be split into equal percentiles. If False, bins will be evenly spaced.
    
    Returns:
    slopes : 2D array
        Array of slopes for each bin.
    bin1_centers : 1D array
        Centers of the bins for the first variable.
    bin2_centers : 1D array
        Centers of the bins for the second variable.
    counts : 2D array
        Counts of data points in each bin.
    """

    # Calculate bin edges for both bin variables based on the percentile_bins flag
    if percentile_bins:
        bin1_bins = np.percentile(bin1_arr, np.linspace(0, 100, num_bin1_bins + 1))
        bin2_bins = np.percentile(bin2_arr, np.linspace(0, 100, num_bin2_bins + 1))
    else:
        bin1_bins = np.linspace(np.min(bin1_arr), np.max(bin1_arr), num_bin1_bins + 1)
        bin2_bins = np.linspace(np.min(bin2_arr), np.max(bin2_arr), num_bin2_bins + 1)

    # Create arrays for slopes and counts
    slopes = np.zeros((num_bin1_bins, num_bin2_bins))
    counts = np.zeros((num_bin1_bins, num_bin2_bins))

    # Iterate over bin1 and bin2 bins
    for i in range(num_bin1_bins):
        for j in range(num_bin2_bins):
            # Adjust binning logic to handle bin edges
            if i == 0:
                bin1_mask = (bin1_arr >= bin1_bins[i]) & (bin1_arr < bin1_bins[i + 1])
            else:
                bin1_mask = (bin1_arr > bin1_bins[i]) & (bin1_arr <= bin1_bins[i + 1])
            
            if j == 0:
                bin2_mask = (bin2_arr >= bin2_bins[j]) & (bin2_arr < bin2_bins[j + 1])
            else:
                bin2_mask = (bin2_arr > bin2_bins[j]) & (bin2_arr <= bin2_bins[j + 1])

            # Combine masks
            combined_mask = bin1_mask & bin2_mask
            
            # Get x and y values for the current bin
            x_bin_values = x_arr[combined_mask]
            y_bin_values = y_arr[combined_mask]

            # Count the number of points in the current bin
            counts[i, j] = len(x_bin_values)

            # Check if there are enough data points in the bin
            if len(x_bin_values) > 1:  # Need at least 2 points to calculate slope
                slope, intercept, r_value, p_value, std_err = linregress(x_bin_values, y_bin_values)
                slopes[i, j] = slope

    # Calculate bin centers for ticks
    bin1_centers = 0.5 * (bin1_bins[:-1] + bin1_bins[1:])
    bin2_centers = 0.5 * (bin2_bins[:-1] + bin2_bins[1:])

    # Return slopes, bin centers, and counts
    return slopes, bin1_centers, bin2_centers, counts


# Define your variables as a dictionary
vars = {'Nd':MOD_Nd_aqua, 'EIS': EIS_dusk_flat, 'LWP': LWP_dusk_flat}

# Create unique pairs of variable names
unique_pairs = list(itertools.combinations(vars.keys(), 2))

# Define the ranges for CF_dusk_flat
cf_ranges = [(0, 0.33), (0.33, 0.67), (0.67, 1)]
cf_labels = ['0 - 0.33', '0.33 - 0.67', '0.67 - 1']

# Set up the mosaic plot
num_rows = len(unique_pairs)
num_cols = len(cf_ranges)
fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5))

# Loop through unique pairs and CF ranges to compute slopes
for i, (key1, key2) in enumerate(unique_pairs):
    bin1_arr = vars[key1]
    bin2_arr = vars[key2]

    for j, (cf_range, label) in enumerate(zip(cf_ranges, cf_labels)):
        # Filter the data based on the current CF range
        mask = (CF_dusk_flat >= cf_range[0]) & (CF_dusk_flat < cf_range[1])
        filtered_delta_CF = delta_CF_flat[mask]
        filtered_Nd = MOD_Nd_aqua[mask]
        filtered_bin1_arr = bin1_arr[mask]
        filtered_bin2_arr = bin2_arr[mask]

        # Calculate slopes and bin centers
        slopes, bin1_centers, bin2_centers, counts = calculate_slope_vs_bins(
            np.log(filtered_Nd), 
            filtered_delta_CF, 
            filtered_bin1_arr, 
            filtered_bin2_arr,
            num_bin1_bins=10, 
            num_bin2_bins=10,
            percentile_bins=True
        )

        # Create the mosaic plot for slopes
        im = axs[i, j].imshow(slopes, origin='lower', cmap='RdBu_r', aspect='auto', 
                              extent=[bin2_centers[0], bin2_centers[-1], bin1_centers[0], bin1_centers[-1]])#,vmin = -0.25,vmax = 0.25)
        axs[i, j].set_title(f'Slopes for {key1} vs {key2} (CF: {label})')
        axs[i, j].set_xlabel(f'{key2}')
        axs[i, j].set_ylabel(f'{key1}')
        #axs[i, j].set_xticks(bin2_centers)
        #axs[i, j].set_yticks(bin1_centers)
        axs[i, j].set_xticklabels([f'{center:.2f}' for center in bin2_centers], rotation=45)
        axs[i, j].set_yticklabels([f'{center:.2f}' for center in bin1_centers])

        # Add a colorbar to each subplot
        plt.colorbar(im, ax=axs[i, j], label='Slope')

# Adjust layout
plt.tight_layout()
plt.show()
