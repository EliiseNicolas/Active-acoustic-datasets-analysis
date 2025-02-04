"""
Wrote by Elise Nicolas
January 2025

This file is meant to provide basic functions to extract data from netCDF files
"""
#%% ------------------------ Imports
import os
import netCDF4 as nc
import numpy as np
import datetime
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import pandas as pd

#%% ------------------------ Get files from folder / Open or close dataset

def get_list_files(folder_path="../data/IMOS_18_and_38_Hz") -> List[str] : 
    list_cdf_files = []

    # Create list of nc files
    for filename in os.listdir(folder_path) :
        if filename.endswith('.nc') : 
            filepath = os.path.join(folder_path, filename)
            list_cdf_files.append(filepath)

    return list_cdf_files

def open_dataset_xr(i:int, list_cdf_files:List[str]) -> xr.Dataset : 
    # Get path to file i
    cdf_file = list_cdf_files[i]

    # Open file i
    dataset = xr.open_dataset(cdf_file)
    
    return dataset

def open_dataset(i:int, list_cdf_files:List[str]) -> nc.Dataset :
    
    # Get path to file i
    cdf_file = list_cdf_files[i]

    # Open file i
    dataset=nc.Dataset(cdf_file, mode='r')
    
    return dataset

def close_dataset(dataset)-> None :
    dataset.close()

#%% ------------------------ Show dataset
def show_dataset(dataset:nc.Dataset)->None :

    # Show dataset
    print(dataset)

    # Extract some variables from dataset
    latitude = dataset.variables['LATITUDE'][:]
    longitude = dataset.variables['LONGITUDE'][:]
    depth = dataset.variables.get('DEPTH', dataset.variables.get('RANGE', None))[:]
    sv = dataset.variables['Sv'][:]
    time = get_datetime(dataset)
    channel = get_channels(dataset)
    period = dataset.variables['day'][:] # day/sunset/sunrise/night (à garder : day/night potentiellement)

    # Display variables
    print("Longitude : ", longitude[:2])
    print("Latitude : ", latitude[:2])
    print("Depth : ", depth[:2]) # 240 levels of depth, of shape (240,)
    print("Sv : ", sv[:2]) # 2 or 4 channels, of shape(time, depth, channels)
    print("Formatted date : ", time[:2])
    print("channels : ", channel)
    print("period : ", period[:2])
    print("longitude : ", longitude.shape, ", latitude : ", latitude.shape, ", depth : ", depth.shape, ", sv : ",  sv.shape, ", date : ", time.shape, ", period : ", period.shape)

#%% ------------------------ Plot echogram

def plot_echogram(dataset:nc.Dataset, frequency:int, path:str)-> None : 
    
    # Create figure
    plt.figure()

    # Get sv data and extract channel of interest
    sv = dataset.variables['Sv'][:]
    try : 
        channel = frequency # Channel = frequency of sonar
        sv_data = sv[:, :, channel]
        # put freq in datastring
        channel_str = get_channels(dataset)[frequency]
    except : 
        print("Frequency not found")
        return 
    
    # Get depth var and invert it
    depth = dataset.variables["DEPTH"][:]
    depth = depth[::-1]

    # Get time var
    time_list = get_datetime(dataset)
    min_time, max_time = min(time_list).strftime("%Y-%m-%d %H:%M:%S"), max(time_list).strftime("%Y-%m-%d %H:%M:%S")

    # Create x axis with less values to display
    indices = np.linspace(0, len(time_list) - 1, num=10, dtype=int)  # Afficher 10 labels
    time_labels = [time_list[i].strftime("%Y-%m-%d %H:%M:%S") for i in indices]

    # Convert datetime into dtr
    time_list = [dt.strftime("%Y-%m-%d %H:%M:%S") for dt in time_list]

    # Plot
    plt.pcolormesh(time_list, -depth, 10 * np.log10(sv_data.T), shading='auto', cmap='jet')

    # Labels and title
    plt.colorbar(label="Sv (dB re 1m⁻¹)")
    plt.xlabel("Time")
    plt.xticks(indices, time_labels, rotation=90)
    plt.ylabel("Depth (m)")
    plt.title(f"Echogram of acoustic data recorded between {min_time} and {max_time} at {channel_str}")
    plt.suptitle(f"For file {path}")
    plt.savefig(f"./figures/echogram_{min_time}_to_{max_time}_at_{channel_str}.png", dpi=300, bbox_inches="tight")

    plt.show()

plt.show()

#%% ------------------------ Extract time(float) and convert into datetime
def get_datetime(dataset:nc.Dataset, convert_datetime=True)->np.ndarray : 
    """
    Extracts and converts the TIME variable from a netCDF4 Dataset object into 
    an array of human-readable datetime in datetime format.

    Parameters:
    ----------
    dataset : nc.Dataset
        The netCDF4 Dataset object containing the TIME variable.
    
    Returns:
    -------
    np.ndarray of shape (n,)
        An array of datetime objects representing the date and time of each sample, 
        preserving the original order (dataset['TIME'][0] == datetime[0]).
    """

    try :
        time_var = dataset.variables['TIME'] # time since 1950-01-01 00:OO:OO UTC

    except ValueError:
        print("ValueError: Could not find the TIME variable in the dataset.")
        return None
    
    # Convert in gregorian date format
    date_var = nc.num2date(time_var[:], units=time_var.units, calendar=getattr(time_var, 'calendar', 'standard')) 
    
    # Convert into datetime objects
    date_var = np.array([datetime.datetime(date.year, date.month, date.day, date.hour, date.minute, date.second) for date in date_var]) 
    
    return date_var
    
def get_seasons_from_datetime(dataset:nc.Dataset) -> List[str]: 
    """
    Extracts the season(s) from a dataset based on the datetime of each sample.

    This function assumes that the dataset contains a TIME variable which is 
    converted into a list of datetime objects. It then sorts the datetime 
    values and determines in which season each sample was recorded, returning
    a list of corresponding seasons.

    Parameters:
    ----------
    dataset : nc.Dataset
        The netCDF4 Dataset object containing the TIME variable.
        

    Returns:
    -------
    np.ndarray of shape(n,) with n the number of samples in dataset
        A numpy arrayy containing strings representing the season(s) for each sample as values. 
        Possible values are 1, 2, 3, 4 corresponding respectively to "winter", "spring", "summer", and "fall".
        Seasons are calculated based on north hemisphere seasons.
    """
    # Get datetime
    datetime_array = get_datetime(dataset)
    
    # Initialize seasons
    seasons=np.zeros(len(datetime_array))

    # Get seasons
    for i, datetime in enumerate(datetime_array) : 
        # Extract month and year from datetime format
        month = datetime.month
    
        # Add season in corresponding year of recording
        if month in [6, 7, 8]: # Winter 
             seasons[i]= 1
        elif month in [9, 10, 11]: # Spring
             seasons[i]= 2
        elif month in [12, 1, 2]: # Summer
             seasons[i]= 3
        elif month in [3, 4, 5]: # fall
             seasons[i]= 4

    return seasons

#%% ------------------------------------------- Histogram
def count_season(dataset:nc.Dataset)->Dict[int, np.ndarray] : 
    """
    Calculate number of occurences of season s, period p in dataset

    parameters : 
        - dataset : nc.Dataset
    returns : 
        season_counts : Dict[int, np.ndarray] 
            - keys (int) represent seasons : 1-Winter, 2-Spring, 3-Summer, 4-Fall
            - values (np.ndarray[int] of shape(p,)) represent number of occurence of season s(key) for period p
    """
    #Initialisation
    season_labels = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Fall"}
    season_counts = {season: np.zeros(4) for season in season_labels}

    # Get seasons
    seasons = get_seasons_from_datetime(dataset)

    # Get periods
    periods = dataset.variables['day'][:]

    # Compute freqs
    for season in range(1, 5):
        for period in range(1, 5):
            mask = (seasons == season) & (periods == period) # create array of bool (mask)
            season_counts[season][period - 1] = np.sum(mask)

    return season_counts

def count_season_all_files(list_cdf_files):
    
    season_labels = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Fall"}
    # Initialize dict containing count of periods per seasons
    all_season_counts = {season: np.zeros(4) for season in season_labels} 
    n_all = 0 # total number of samples (for all datasets of folder)

    for i in range(1,len(list_cdf_files)):
        # Open dataset i
        dataset=open_dataset(i, list_cdf_files)

        # Count number of samples in dataset i
        n_all+= get_datetime(dataset).shape[0]

        # Count periods per season for dataset i
        season_counts_dataset = count_season(dataset)

        # Add counts of periods/season of dataset i into total one
        for season in season_labels.keys():
            all_season_counts[season]+=season_counts_dataset[season]

    # normalization
    for season in season_labels.keys():
        all_season_counts[season]=all_season_counts[season]/n_all

    return all_season_counts, n_all

def plot_histogram(season_counts:Dict[int,np.ndarray], save:bool=False, dataset_name:str="")->None :
    
    season_labels = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Fall"}
    period_labels = {1: "Day", 2: "Sunset", 3: "Sunrise", 4: "Night"}
    
    fig, ax = plt.subplots(figsize=(8, 6))

    # Convert dictionary to stacked bar chart format
    data = np.array(list(season_counts.values()))
    bottom = np.zeros(len(season_labels))

    for i, period_label in enumerate(period_labels.values()):
        ax.bar(season_labels.values(), data[:, i], bottom=bottom, label=period_label)
        bottom += data[:, i]

    ax.set_xlabel("Season")
    ax.set_ylabel("Frequency (%)")
    ax.set_title(f"Histogram of Periods per Season for {dataset_name}")
    ax.legend(title="Period")

    if save : 
        plt.savefig(f"./figures/hist_of_periods_per_season_dataset_{dataset_name}.png", dpi=300, bbox_inches="tight")
    plt.show()

#%% -------------------------------------------- Extract channels 
def get_channels(dataset:nc.Dataset)->np.ndarray : 
    try : 
        channel= dataset.variables["CHANNEL"][:]
        decoded_channel = np.char.decode(channel, 'utf-8')
        decoded_channel = [''.join(row).strip() for row in decoded_channel]  
        return decoded_channel
    except ValueError:
        print("ValueError: Could not find the CHANNEL variable in the dataset.")
        return None 

# %% -------------------------------------------- Display Trajectories
def get_enveloppe_convexe_into_xr() : 
    enveloppe_convexe = "../data/geographic_data/convex_hull.xlsx"
    df = pd.read_excel(enveloppe_convexe)
    df = df.rename(columns={'lon': 'LONGITUDE', 'lat': 'LATITUDE'})
    ds = xr.Dataset.from_dataframe(df)
    return ds

def display_trajectories(dataset:xr.Dataset, enveloppe:bool=False, save:bool=False, dataset_name:str="") -> None : 
    ## Display trajectories
    # Extraire les variables de longitude et latitude
    longitude = dataset['LONGITUDE'].values
    latitude = dataset['LATITUDE'].values 
    dates = [min(dataset['TIME'].values), max(dataset['TIME'].values)]
    dates = [np.datetime_as_string(date_np, unit='D') for date_np in dates]
    channels = "_".join(get_channels(dataset))
    title=f"Trajectories recorded from {dates[0]} to {dates[1]} at {channels}"

    # Create figure
    plt.figure(figsize=(6, 3))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Add coastlines to map
    ax.coastlines()

    # Add trajectories to map
    ax.scatter(longitude, latitude, color='red', s=2, transform=ccrs.PlateCarree(), label="Trajectories")

    ## Display enveloppe convexe
    if enveloppe : 
        enveloppe = get_enveloppe_convexe_into_xr()
        ax.scatter(enveloppe['LONGITUDE'].values, enveloppe['LATITUDE'].values, color='green', s=2, transform=ccrs.PlateCarree(), label="Enveloppe convexe")

    # Crop map
    ax.set_extent([min(longitude)-30, max(longitude)+30, min(latitude)-30, max(latitude)+30], crs=ccrs.PlateCarree())

    # Add legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=1, fontsize=12)

    # Add title 
    plt.title(title)

    # Save fig 
    if save : 
        plt.savefig(f"./figures/trajectories_from_{dates[0]}_to{dates[1]}_{channels}_{dataset_name}.png", dpi=300, bbox_inches="tight")

    # display map
    plt.show()


