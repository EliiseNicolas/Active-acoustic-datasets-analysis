#%% ------------------------ Imports
import os
import numpy as np
import datetime
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import pandas as pd
import shutil
from shapely.geometry import Point, Polygon
import pickle

#%% ------------------------ Extract time(float) and convert into datetime
def get_datetime_from_xr(dataset: xr.Dataset) -> np.ndarray:
    """
    Extracts and converts the TIME variable from an xarray Dataset object into 
    an array of human-readable datetime objects.

    Parameters:
    ----------
    dataset : xr.Dataset
        The xarray Dataset object containing the TIME variable.

    Returns:
    -------
    np.ndarray
        An array of datetime objects of shape (n, 1)
    """

    try:
        time_var = dataset['TIME'].values

    except KeyError:
        print("KeyError: Could not find the TIME variable in the dataset.")
        return None
    
    datetime_array = np.array(time_var.astype('datetime64[s]').tolist())
    
    return datetime_array



#%% -------------------------------------------------------------- Get seasons from datetime
def get_seasons_from_datetime(datetime_array:np.ndarray[datetime.datetime]) -> np.ndarray[int]: 
    """
    Get corresponding season of every datetime.datetime object in a np.ndarray

    Parameters:
    ----------
    datetime_array : np.ndarray[datetime.datetime]
        An array of shape (n, 1) containing datetime.datetime objects
        

    Returns:
    -------
    np.ndarray[int] of shape(n,) with n the number of samples in dataset
        A numpy array containing int representing the season(s) for each sample 
        Possible values are 0, 1, 2, 3 corresponding respectively to "winter", "spring", "summer", and "fall".
        Seasons are calculated based on SOUTH hemisphere seasons.
    """
    # Initialize seasons
    seasons=np.zeros(len(datetime_array), dtype=int)

    # Get seasons
    for i, datetime in enumerate(datetime_array) : 
        # Extract month and year from datetime format
        month = datetime.month

        # Add season in corresponding year of recording
        if month in [6, 7, 8]: # Winter 
            seasons[i]= 0
        elif month in [9, 10, 11]: # Spring
            seasons[i]= 1
        elif month in [12, 1, 2]: # Summer
            seasons[i]= 2
        elif month in [3, 4, 5]: # fall
            seasons[i]= 3

    return seasons

# %% ------------------------------------------ sample rate analysis

def get_sample_rate_cropped_files(file_path:str) :
    sample_rates = []
    with open(file_path, 'rb') as p_file:
        # read data in pkl as stream
        while True : 
            try:
                date, date_dict = pickle.load(p_file)
                print(date)
                mean_sample_rate = 0

                # Get datetime
                time = date_dict["TIME"]
                if (time.shape[0]==1) : 
                    continue
                
                for i in range(time.shape[0]-1) : 
                    mean_sample_rate+= (time[i+1] - time[i]).total_seconds()
                mean_sample_rate=mean_sample_rate/(time.shape[0]-1)

                if mean_sample_rate> 40 : 
                    print("unusual data : ", date, mean_sample_rate)
                
                sample_rates.append(mean_sample_rate)
            
            except EOFError:
                print("End of file.")
                break
        sample_rates = np.array(sample_rates)
        return sample_rates, np.mean(sample_rates), np.std(sample_rates)

#%% -------------------------- Plot echogram from .pkl containing dict [date : dict[depth, lat, long, Sv, channels,..]]

def plot_echogramps_from_pkl_18kHz(pkl_path:str, save:bool=False, dest_path:str="") : 
    with open(pkl_path, 'rb') as pkl_file:
        # read data in pkl as stream
        while True:
            try:
                date, data_dict = pickle.load(pkl_file) 
                print(date)
                # Get times values
                Sv = data_dict["Sv"]
                if Sv.ndim >2 :
                    Sv = Sv[:, :, 0]
                print(Sv.shape)
                time = data_dict["TIME"]
                time_str = np.array([dt.strftime("%H:%M:%S") for dt in time])
                depth = data_dict["DEPTH"]
                
                tick_indices = np.linspace(0, len(time) - 1, num=10, dtype=int)
                tick_labels = time_str[tick_indices]

                time_indices = np.array([i.strftime("%H:%M:%S") for i in time[tick_indices]])
                
                # Plot
                plt.pcolormesh(time_str, -depth, 10 * np.log10(Sv.T), shading='auto', cmap='jet')

                # Labels and title
                plt.colorbar(label="Sv (dB re 1m⁻¹)")
                plt.xlabel("Time")
                plt.xticks(time_indices, tick_labels, rotation=90)
                plt.ylabel("Depth (m)")
                plt.title(f"Echogram of acoustic data recorded the {date} for 18 kHz channel")

                if save : 
                    src_basename = os.path.basename(pkl_path)
                    src_filename = os.path.splitext(src_basename)[0]
                    fig_path = dest_path + src_filename + "_"+str(date)
                    print(fig_path)
                    plt.savefig(f"{fig_path}.png", dpi=300, bbox_inches="tight")

                plt.show()

            except EOFError :
                print("End of file")
                break

