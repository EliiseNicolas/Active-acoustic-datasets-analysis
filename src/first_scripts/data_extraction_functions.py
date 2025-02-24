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
import shutil
from shapely.geometry import Point, Polygon

#%% ------------------------ Get files from folder / Open or close dataset

def get_list_files(folder_path="../data/acoustic_data/18_38Hz/IMOS_18_and_38_Hz") -> List[str] : 
    """
    Retrieves a list of netCDF (.nc) files from the specified folder.

    Parameters:
    ----------
    folder_path : str, optional
        The path to the folder containing netCDF files (default is "../data/acoustic_data/18_38Hz/IMOS_18_and_38_Hz").

    Returns:
    -------
    List[str]
        A list of file paths to all netCDF files found in the specified folder.
    """

    list_cdf_files = []

    # Create list of nc files
    for filename in os.listdir(folder_path) :
        if filename.endswith('.nc') : 
            filepath = os.path.join(folder_path, filename)
            list_cdf_files.append(filepath)

    return list_cdf_files

def open_dataset_xr(i:int, list_cdf_files:List[str]) -> xr.Dataset : 
    """
    Opens a netCDF file as an xarray Dataset.

    Parameters:
    ----------
    i : int
        Index of the file to open in the list of netCDF file paths.
    list_cdf_files : List[str]
        List containing paths to netCDF files.

    Returns:
    -------
    xr.Dataset
        The opened dataset in xarray format.
    """

    # Get path to file i
    cdf_file = list_cdf_files[i]

    # Open file i
    dataset = xr.open_dataset(cdf_file)
    
    return dataset

def open_dataset(i:int, list_cdf_files:List[str]) -> nc.Dataset :
    """
    Opens a netCDF file as a netCDF4 Dataset.

    Parameters:
    ----------
    i : int
        Index of the file to open in the list of netCDF file paths.
    list_cdf_files : List[str]
        List containing paths to netCDF files.

    Returns:
    -------
    nc.Dataset
        The opened dataset in netCDF4 format.
    """

    # Get path to file i
    cdf_file = list_cdf_files[i]

    # Open file i
    dataset=nc.Dataset(cdf_file, mode='r')
    
    return dataset

def close_dataset(dataset)-> None :
    """
    Closes an open netCDF/xarray dataset.

    Parameters:
    ----------
    dataset : nc.Dataset or xr.Dataset
        The dataset to close.

    Returns:
    -------
    None
    """

    dataset.close()

#%% ------------------------ Show dataset
def show_dataset(dataset:nc.Dataset)->None :
    """
    Displays basic information about a netCDF dataset, including key variables.

    Parameters:
    ----------
    dataset : nc.Dataset
        The netCDF dataset to display.

    Returns:
    -------
    None
    """
    
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

def plot_echogram(dataset:nc.Dataset, frequency:int, path:str, save:bool=False, save_path:str="")-> None :
    """
    Plots an echogram from the provided dataset.

    Parameters:
    ----------
    dataset : nc.Dataset
        The netCDF dataset containing echogram data.
    frequency : int
        The channel index corresponding to the frequency to be plotted.
    path : str
        Path to the dataset file.
    save : bool, optional
        Whether to save the figure (default is False).

    Returns:
    -------
    None
    """
    
    # Create figure
    plt.figure()

    # Get sv data and extract channel of interest
    sv = dataset.variables['Sv'][:]
    try : 
        channel = frequency # Channel = frequency of sonar
        sv_data = sv[:, :, channel]
        print(sv_data.shape)
        # put freq in datastring
        channel_str = get_channels(dataset)[frequency]
    except : 
        print("Frequency not found")
        return 
    
    # Get depth var and invert it
    depth = dataset.variables["DEPTH"][:]
    # depth = depth[::-1]

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

    if save : 
        parent_folder = os.path.dirname(save_path)
        plt.savefig(parent_folder + f"/figures/echogram_{min_time}_to_{max_time}_at_{channel_str}.png", dpi=300, bbox_inches="tight")

    plt.show()

plt.show()

#%% ------------------------ Extract time(float) and convert into datetime
def get_datetime(dataset:nc.Dataset)->np.ndarray : 
    """
    Extracts and converts the TIME variable from a netCDF4 Dataset object into 
    an array of human-readable datetime objects.

    Parameters:
    ----------
    dataset : nc.Dataset
        The netCDF4 Dataset object containing the TIME variable.

    Returns:
    -------
    np.ndarray
        An array of datetime objects.
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
    converted into a list of datetime objects. 

    Parameters:
    ----------
    dataset : nc.Dataset
        The netCDF4 Dataset object containing the TIME variable.
        

    Returns:
    -------
    np.ndarray of shape(n,) with n the number of samples in dataset
        A numpy arrayy containing strings representing the season(s) for each sample as values. 
        Possible values are 1, 2, 3, 4 corresponding respectively to "winter", "spring", "summer", and "fall".
        Seasons are calculated based on SOUTH hemisphere seasons.
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

    Parameters:
    ----------
    dataset : nc.Dataset
        The netCDF dataset.

    Returns : 
    ----------
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

def count_season_all_files(list_cdf_files:List[str]):
    """
    Counts and normalizes seasonal occurrences across multiple netCDF files.

    Parameters:
    ----------
    list_cdf_files : List[str]
        List of netCDF file paths.

    Returns:
    -------
    Tuple[Dict[int, np.ndarray], int]
        A dictionary of season counts and the total number of samples.
    """

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

def plot_histogram(season_counts:Dict[int,np.ndarray], save:bool=False, save_path:str="", dataset_name:str="")->None :
    """
    Plots a histogram of seasonal occurrences.

    Parameters:
    ----------
    season_counts : Dict[int, np.ndarray]
        Dictionary of season counts.
    save : bool, optional
        Whether to save the figure (default is False).
    dataset_name : str, optional
        Name of the dataset (default is an empty string).

    Returns:
    -------
    None
    """

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
        parent_folder = os.path.dirname(save_path)
        plt.savefig(parent_folder + f"/figures/hist_of_periods_per_season_dataset_{dataset_name}.png", dpi=300, bbox_inches="tight")
    
    plt.show()

#%% -------------------------------------------- Extract channels 
def get_channels(dataset:nc.Dataset)->np.ndarray : 
    """
    Extracts and decodes the available channels from a dataset.

    Parameters:
    ----------
    dataset : nc.Dataset
        The netCDF dataset.

    Returns:
    -------
    np.ndarray
        Array of channel names.
    """
    
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
    """
    Loads and converts a convex hull (enveloppe convexe) geographic dataset 
    into an xarray dataset.

    Returns:
    -------
    xr.Dataset
        The convex hull dataset.
    """
    
    enveloppe_convexe = "../data/geographic_data/convex_hull.xlsx"
    df = pd.read_excel(enveloppe_convexe)
    df = df.rename(columns={'lon': 'LONGITUDE', 'lat': 'LATITUDE'})
    ds = xr.Dataset.from_dataframe(df)
    return ds

def get_enveloppe_convexe_into_list_tuple()-> List[Tuple[float]] : 
    """
    Loads and converts a convex hull (enveloppe convexe) geographic dataset 
    into a list of Tuple (points).

    Returns:
    -------
    List[Tuple[float]]
        The convex hull dataset.
    """
    
    enveloppe_convexe = "../data/geographic_data/convex_hull.xlsx"
    df = pd.read_excel(enveloppe_convexe)
    l = []
    for lon, lat in zip(df["lon"], df["lat"]):
        point = (lon, lat)
        l.append(point)
    return l

def display_trajectories(dataset:xr.Dataset, enveloppe:bool=False, save:bool=False, dataset_name:str="") -> None : 
    """
    Displays the trajectories recorded in a dataset on a map.

    Parameters:
    ----------
    dataset : xr.Dataset
        The dataset containing the trajectory data.
    enveloppe : bool, optional
        Whether to overlay a convex hull (default is False).
    
    dataset_name : str, optional
        Name of the dataset (default is an empty string).

    Returns:
    -------
    None
    """

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
    
def display_all_trajectories_folder(folder_path:str="../data/acoustic_data/18_38Hz/IMOS_18_and_38_Hz", enveloppe:bool=False, save:bool=False) :
    list_files = get_list_files(folder_path)
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    min_longitude, max_longitude, min_latitude, max_latitude = float('inf'),float('-inf'),float('inf'),float('-inf')
    # Ajouter la carte de base
    ax.coastlines()

    # Ajouter chaque trajectoire
    for file_path in list_files:
        ds = xr.open_dataset(file_path)
        longitude = ds['LONGITUDE'].values
        latitude = ds['LATITUDE'].values
        ax.scatter(longitude, latitude, color='red', s=2, transform=ccrs.PlateCarree(), label="Trajectoire")
        min_long_file = min(longitude)
        max_long_file = max(longitude)
        min_lat_file = min(latitude)
        max_lat_file = max(latitude)
        if min_long_file < min_longitude : min_longitude = min_long_file
        if max_long_file > max_longitude : max_longitude = max_long_file
        if min_lat_file < min_latitude : min_latitude = min_lat_file
        if max_lat_file > max_latitude : max_latitude = max_lat_file
        
    # Option : Ajouter l'enveloppe convexe
    if enveloppe:
        enveloppe = get_enveloppe_convexe_into_xr()  # Fonction à définir
        ax.scatter(enveloppe['LONGITUDE'].values, enveloppe['LATITUDE'].values, color='green', s=2, transform=ccrs.PlateCarree(), label="Enveloppe Convexe")
        
    if save : 
        parent_folder = os.path.dirname(folder_path)
        plt.savefig(parent_folder + f"/figures/trajectories_all_18_38Hz.png", dpi=300, bbox_inches="tight")

    print(min_latitude, max_latitude, min_longitude, max_longitude)
    ax.set_extent([min_longitude-10, max_longitude+10, min_latitude-10, max_latitude+10], crs=ccrs.PlateCarree())

    # Ajuster l'affichage
    plt.tight_layout()
    plt.show()        

#%% --------------------------------- Count missing data
def count_missing_data(dataset:nc.Dataset, channels:List[str])->np.ndarray[float]:
    """
    Computes the proportion of missing (NaN) depth values for given sonar frequencies in the dataset.

    Parameters:
    ----------
    dataset : nc.Dataset
        The netCDF dataset containing acoustic data.
    channelss : List[str]
        List of channels (sonar frequency values, as strings) to analyze.

    Returns:
    -------
    np.ndarray[float]
        A 2D array where each column corresponds to a requested frequency, 
        and each row represents the proportion of missing data at a given depth.
        Returns None if requested frequencies are not found in the dataset.
    """
 
    # Get all acoustic data 
    sv = dataset.variables['Sv'][:]

    # Get total number of samples in dataset
    n_samples = sv.shape[0]

    # Get all frequency of interest
    all_channels_dataset=get_channels(dataset) # All channels present in dataset
    try : 
        index_channels=[all_channels_dataset.index(channel) for channel in channels] # dataset of interest
    except : 
        print("Dataset doesn't contain data at frequency requested")
        return
    
    # Count number of missing data
    n_Nan = np.ma.count_masked(sv, axis=0) # number of NaN for every channel of interest for every depths (240,)
    n_Nan = n_Nan/n_samples # Normalize counts into frequency
    n_Nan_channels = n_Nan[:, index_channels] # Select only channels of interest

    return n_Nan_channels

def group_data_mean(data: np.ndarray, depth: np.ndarray[float], step: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Groups data by computing the mean over depth intervals.

    This function averages the data over depth intervals of size `step`, reducing 
    the resolution of the data while preserving the overall trend.

    Parameters:
    ----------
    data : np.ndarray
        A 2D array of shape (d, c), where:
        - d: Number of depth levels,
        - c: Number of frequency channels or other measured variables.

    depth : np.ndarray[float]
        A 1D array of shape (d,) containing depth values corresponding to `data`.

    step : int
        The number of depth levels to average together in each group.

    Returns:
    -------
    Tuple[np.ndarray, np.ndarray]
        - new_data: A 2D array of shape (new_d, c), where new_d = d // step.
            Contains the averaged data for each depth interval.
        - new_depth: A 1D array of shape (new_d,) containing the averaged depth values.
    """

    d, c = data.shape
    new_d = d//step
    new_data = np.zeros((new_d, c))
    new_depth = np.zeros(new_d)

    for i in range(new_d):
        start = i * step
        end = start + step
        new_data[i] = np.mean(data[start:end], axis=0)
        new_depth[i] = np.mean(depth[start:end])

    return new_data, new_depth

def get_missing_data_all_files(folder_path:str="../data/acoustic_data/18_38Hz/IMOS_18_and_38_Hz", channels:List[str]=['18kHz', '38kHz'], step:int=10)->np.ndarray :
    # Get every cdf file of the folder
    list_cdf_files=get_list_files(folder_path=folder_path)

    # Initialize np.ndarray containing missing datas frequency
    n_files = len(list_cdf_files)
    n_channels = len(channels)
    d=240
    all_missing_datas = np.zeros((d, n_channels))

    # Open every file
    for i in range(n_files) : 
        # Load file into nc.Dataset
        ds = open_dataset(i, list_cdf_files) 

        # Count missing datas for each file
        all_missing_datas += count_missing_data(ds, channels)

        # Close nc.Dataset
        close_dataset(ds)

    # Normalize counts to get frequencies
    all_missing_datas=all_missing_datas/n_files

    return all_missing_datas

def plot_missing_data(missing_data:np.ndarray[float], depths:np.ndarray[float], step:int, channel_indexes:Dict[int,str], title:str="", save:bool=False, save_path:str="") -> None:
    """
    Plots the percentage of missing acoustic data as a function of depth for specified sonar frequencies.

    Parameters:
    ----------
    dataset : nc.Dataset
        The netCDF dataset containing acoustic data.
        Shape: (n, d, c), where:
        - n: Number of samples,
        - d: Number of depth levels,
        - c: Number of frequency channels.

    step : 
    
    channels : List[str]
        List of sonar frequencies (as strings) to analyze, e.g., ['18Hz', '38Hz', '70Hz'].

    missing_data : np.ndarray[float]
        2D array (shape: d × len(channels)) containing the percentage of missing data at each depth.

    save : bool, optional
        If True, saves the figure as a PNG file (default: False).

    Returns:
    -------
    None
        The function displays the plot but does not return any value.
    """

    # Create figure
    plt.figure(figsize=(10, 6))

    # # Get depths, channels
    # depth = dataset['DEPTH'][:]
    
    # Group missing data by depth with step
    new_missing_data, new_depth =group_data_mean(missing_data, depths, step)
    


    for i, (channel_index, channel_str) in enumerate(channel_indexes.items()) : 
        bar_width = 20
        x_offset = i * bar_width
        plt.bar(new_depth-bar_width/2+x_offset, new_missing_data[:, channel_index], width=bar_width, label=f'Frequency of sonar : {channel_str}')

    # Put label, title, legend
    plt.xlabel('Depth (m)')
    plt.ylabel('Missing data (%)')
    plt.title('Plot of missing data (%) depending on depth (m) and recording frequency (Hz)')
    plt.legend()

    # Save fig
    if save : 
        parent_folder = os.path.dirname(save_path)
        plt.savefig(parent_folder + f"/figures/plot_missing_data_depending_on_depth_and_recording_frequence_{title}.png", dpi=300, bbox_inches="tight")

    # Show
    plt.show()

#%% ----------------------------------------- Classify data
# By season 
def classify_seasons(folder_path:str="../data/acoustic_data/18_38Hz/IMOS_18_and_38_Hz")-> None:
    list_cdf_files=get_list_files(folder_path=folder_path)
    for i, filepath in enumerate(list_cdf_files):
        ds = nc.Dataset(filepath, mode='r')
        seasons = get_seasons_from_datetime(ds)
        all_seasons=np.unique(seasons)
        parent_folder = os.path.dirname(folder_path)
        print(parent_folder)
        for season in all_seasons : 
            if season == 1 :
                shutil.copy(filepath, parent_folder+"/winter") 
            elif season == 2 : 
                shutil.copy(filepath, parent_folder+"/spring") 
            elif season == 3 :
                shutil.copy(filepath, parent_folder+"/summer") 
            elif season == 4 : 
                shutil.copy(filepath, parent_folder+"/fall") 
            else : 
                print("error, season not managed")
                return
        ds.close()

# By localisation
def are_points_in_polygon(longitude:np.ndarray[float], latitude:np.ndarray[float], polygon:Polygon)-> bool:
    """
    Check if each point (longitude, latitude) is inside the given polygon.

    Parameters:
    ----------
    longitude : np.ndarray[float]
        Array containing the longitudes of the points.
    latitude : np.ndarray[float]
        Array containing the latitudes of the points.
    polygon : Polygon
        A Shapely polygon representing the area to check.

    Returns:
    -------
    np.ndarray[bool]
        Boolean array indicating whether each point is inside the polygon (`True`) or not (`False`).
    """
    points = [Point(lon, lat) for lon, lat in zip(longitude, latitude)]
    return np.array([polygon.contains(point) for point in points])
    

    