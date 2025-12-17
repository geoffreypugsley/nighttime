#%%

#%%
import csat2.CERES
from csat2.CERES import granule
from csat2.misc.time import utc_to_lst
from csat2.misc.astro import solar_zenith_angle_time
import matplotlib.pyplot as plt
import datetime as dt
import xarray as xr
import numpy as np
from tqdm import tqdm
import matplotlib.animation as animation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import pandas as pd
from multiprocessing import Pool


#%% load positions

positions_data_location = '/disk1/Users/gjp23/outputs/traj_positions/global_analysis/yearly'
positions_data = os.path.join(positions_data_location, 'trajectories_2023.nc')
positions_ds = xr.open_dataset(positions_data)


#%%
# Define region of interest ######### Comment this bit out if analysing globally #######################

lat_min,lat_max = -25,-5
lon_min,lon_max = 240,270



# use step=0 as the trajectory start
start_lon = positions_ds.lon.isel(step=0)
start_lat = positions_ds.lat.isel(step=0)

region_mask = (
    (start_lon >= lon_min) & (start_lon <= lon_max) &
    (start_lat >= lat_min) & (start_lat <= lat_max)
)

region_mask = region_mask.reset_coords("step", drop=True)

positions_ds = positions_ds.where(region_mask, drop=True)




#%% function to process each start time

def process_start_time(start_time):
    """Process one 24-hour trajectory."""
    
    ds_selected = positions_ds.where(positions_ds.start_time == start_time, drop=True)



    if ds_selected.dims.get("trajectory", 0) == 0:
        return None
    lon = ds_selected.lon
    lat = ds_selected.lat

    times = pd.date_range(start=start_time, periods=24, freq="h")

    LWP_traj = []
    Nd_traj = []
    CF_traj = []
    SZA_traj = []
    csat2_sza_traj = []

    for i, time_dt in enumerate(times):
        gran = csat2.CERES.Granule(time_dt.year, time_dt.timetuple().tm_yday, time_dt.hour)

        lon_tstep = lon.isel(step=i)
        lat_tstep = lat.isel(step=i)
        lst = utc_to_lst(gran.time, lon_tstep.values)

        vars = gran.geolocate(
            ['obs_cld_lwp', 'obs_cld_liq_radius', 'obs_cld_od',
             'obs_cld_amount', 'sza'],
            lon_tstep, lat_tstep
        )

        lwp = vars['obs_cld_lwp'].sel(cloud_layer=4)
        r_eff = vars['obs_cld_liq_radius'].sel(cloud_layer=4)
        cot = vars['obs_cld_od'].sel(cloud_layer=4)
        cf = vars['obs_cld_amount'].sel(cloud_layer=4)
        sza = vars['sza']

        csat2_sza = solar_zenith_angle_time(lat_tstep.values, gran.doy, lst)

        Nd = 1.37e-11*(cot**0.5)*((r_eff*1e-6)**-2.5)

        LWP_traj.append(lwp.values)
        Nd_traj.append(Nd.values)
        CF_traj.append(cf.values)
        SZA_traj.append(sza.values)
        csat2_sza_traj.append(csat2_sza)

    return (LWP_traj, Nd_traj, CF_traj, SZA_traj, csat2_sza_traj)


#%% run all start times in parallel

#%%
start_times = pd.date_range("2023-01-01", "2023-01-03  23:00", freq="h")

with Pool(processes=12) as P:
    results = list(tqdm(P.imap(process_start_time, start_times),
                        total=len(start_times)))
    
#%% unpack results

LWP_tot       = [r[0] for r in results]
Nd_tot        = [r[1] for r in results]
CF_tot        = [r[2] for r in results]
SZA_tot       = [r[3] for r in results]
csat2_sza_tot = [r[4] for r in results]

#%% reshape data for plotting

LWP_arr = np.array(LWP_tot)
Nd_arr = np.array(Nd_tot)
CF_arr = np.array(CF_tot)
SZA_arr = np.array(SZA_tot)
csat2_sza_arr = np.array(csat2_sza_tot)

LWP_arr_transposed = np.transpose(LWP_arr,(0,2,1))
Nd_arr_transposed = np.transpose(Nd_arr,(0,2,1))
CF_arr_transposed = np.transpose(CF_arr,(0,2,1))
SZA_arr_transposed = np.transpose(SZA_arr,(0,2,1))
csat2_sza_arr_transposed = np.transpose(csat2_sza_arr,(0,2,1))

LWP_arr = LWP_arr_transposed.reshape(-1,LWP_arr_transposed.shape[-1])
Nd_arr = Nd_arr_transposed.reshape(-1,Nd_arr_transposed.shape[-1])
CF_arr = CF_arr_transposed.reshape(-1,CF_arr_transposed.shape[-1])
SZA_arr = SZA_arr_transposed.reshape(-1,SZA_arr_transposed.shape[-1])
csat2_sza_arr = csat2_sza_arr_transposed.reshape(-1,csat2_sza_arr_transposed.shape[-1])


#%%

# Indices for colouring

Nd_plot = np.nanmean(Nd_arr,axis=0)
LWP_plot = np.nanmean(LWP_arr,axis=0)
SZA_plot = np.nanmean(SZA_arr,axis=0)
csat2_sza_plot = np.nanmean(csat2_sza_arr,axis=0)
CF_plot = np.nanmean(CF_arr,axis=0)
indices = np.arange(len(Nd_plot))

plt.figure(figsize=(8,6))

# Scatter plot with color by index
sc = plt.scatter(Nd_plot, LWP_plot, c=indices, cmap='viridis', s=1000)

# Draw arrows between successive points
for i in range(len(Nd_plot)-1):
    plt.arrow(
        Nd_plot[i], LWP_plot[i],
        Nd_plot[i+1]-Nd_plot[i],
        LWP_plot[i+1]-LWP_plot[i],
        shape='full', lw=0.8, length_includes_head=True,
        head_width=0.01*np.max(Nd_plot), head_length=0.01*np.max(LWP_plot),
        color='black'
    )

plt.xlabel('$N_d$ (cm$^{-3}$)')
plt.ylabel('LWP (g m$^{-2}$)')
plt.title(f'Trajectory flow field , lon {lon_min}-{lon_max}, lat {lat_min}-{lat_max}')
plt.grid(True)


cbar = plt.colorbar(sc)
cbar.set_label('Hours since 6am')

plt.show()
plt.savefig(f'/disk1/Users/gjp23/outputs/CERES/LWP_Nd_phase_space_{lon_min}_{lon_max}_{lat_min}_{lat_max}_2023_12months.png', dpi=300)



plt.figure(figsize=(8,6))

# Scatter plot with color by SZA
sc = plt.scatter(Nd_plot, LWP_plot, c=csat2_sza_plot, cmap='twilight', s=1000, alpha=0.8)

# Optional: draw arrows between successive points
for i in range(len(Nd_plot)-1):
    plt.arrow(
        Nd_plot[i], LWP_plot[i],
        Nd_plot[i+1]-Nd_plot[i],
        LWP_plot[i+1]-LWP_plot[i],
        shape='full', lw=0.8, length_includes_head=True,
        head_width=0.01*np.max(Nd_plot), head_length=0.01*np.max(LWP_plot),
        color='black'
    )

plt.xlabel('$N_d$ (cm$^{-3}$)')
plt.ylabel('LWP (g m$^{-2}$)')
plt.title(f'Trajectory flow field colored by SZA, lon {lon_min}-{lon_max}, lat {lat_min}-{lat_max}')
plt.grid(True)

cbar = plt.colorbar(sc)
cbar.set_label('Solar Zenith Angle (deg)')

plt.show()
plt.savefig(f'/disk1/Users/gjp23/outputs/CERES/LWP_Nd_phase_space_SZA_{lon_min}_{lon_max}_{lat_min}_{lat_max}_2023_12months.png', dpi=300)

## repeat with CF

plt.figure(figsize=(8,6))

# Scatter plot with color by SZA
sc = plt.scatter(Nd_plot, CF_plot, c=csat2_sza_plot, cmap='twilight', s=1000, alpha=0.8)

# Optional: draw arrows between successive points
for i in range(len(Nd_plot)-1):
    plt.arrow(
        Nd_plot[i], CF_plot[i],
        Nd_plot[i+1]-Nd_plot[i],
        CF_plot[i+1]-CF_plot[i],
        shape='full', lw=0.8, length_includes_head=True,
        head_width=0.005*np.max(Nd_plot), head_length=0.01*np.max(CF_plot),
        color='black'
    )

plt.xlabel('$N_d$ (cm$^{-3}$)')
plt.ylabel('Cloud Fraction')
plt.title(f'Trajectory flow field colored by SZA, lon {lon_min}-{lon_max}, lat {lat_min}-{lat_max}')
plt.grid(True)

cbar = plt.colorbar(sc)
cbar.set_label('Solar Zenith Angle (deg)')

plt.show()

plt.savefig(f'/disk1/Users/gjp23/outputs/CERES/CF_Nd_phase_space_SZA_{lon_min}_{lon_max}_{lat_min}_{lat_max}_2023_12months.png', dpi=300)


plt.figure(figsize=(8,6))

# Scatter plot with color by index
sc = plt.scatter(Nd_plot, CF_plot, c=indices, cmap='viridis', s=1000)

# Draw arrows between successive points
for i in range(len(Nd_plot)-1):
    plt.arrow(
        Nd_plot[i], CF_plot[i],
        Nd_plot[i+1]-Nd_plot[i],
        CF_plot[i+1]-CF_plot[i],
        shape='full', lw=0.8, length_includes_head=True,
        head_width=0.005*np.max(Nd_plot), head_length=0.01*np.max(CF_plot),
        color='black'
    )

plt.xlabel('$N_d$ (cm$^{-3}$)')
plt.ylabel('Cloud Fraction')
plt.title(f'Trajectory flow field, lon {lon_min}-{lon_max}, lat {lat_min}-{lat_max}')
plt.grid(True)


cbar = plt.colorbar(sc)
cbar.set_label('Hours since 6am')

plt.show()
plt.savefig(f'/disk1/Users/gjp23/outputs/CERES/CF_Nd_phase_space_{lon_min}_{lon_max}_{lat_min}_{lat_max}_2023_12months.png', dpi=300)

# %%
