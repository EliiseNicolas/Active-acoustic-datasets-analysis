# ---------------------------------------------- Imports 
import pickle as pkl
import numpy as np
import os
import xarray as xr
from scipy.spatial import KDTree
from basic_functions import *

#%% -------------------------------------------- Create pkl
def extract_and_save_netcdf_data(src_path:str, dest_path:str, channel:str):
    """
    Extracts data from NetCDF (.nc) files in the source directory and saves it as a pickle (.pkl) file.

    Parameters:
    - src_path (str): Path to the folder containing NetCDF (.nc) files to process.
    - dest_path (str): Path to the folder where the pickle file will be saved.
    - pickle_file_name (str): Name of the output pickle file (without the .pkl extension).

    Outputs:
    - A pickle file containing dictionaries with extracted data for each processed NetCDF file.
        pickle file of this shape : Tuple(name_IMOS_file, Dict)      and Dict.keys() = ["DEPTH", "TIME", "SEASON", "LONGITUDE", "LATITUDE", "CHANNEL", "DAY", "in_ROI"]
    """
    if not dest_path.endswith("/"):
        dest_path += "/"
    src_basename = os.path.basename(src_path)
    pickle_file_name =  os.path.splitext(src_basename)[0]

    pickle_file = os.path.join(dest_path, pickle_file_name + ".pkl")

    if os.path.exists(pickle_file):
        os.remove(pickle_file)

    list_files = [f for f in os.listdir(src_path) if f.endswith('.nc')]
    
    for file in list_files[:5]:
        # Load IMOS file as xr dataset
        file_path = os.path.join(src_path, file)
        ds = xr.open_dataset(file_path)
        
        # Create dict containing relevant data
        data_file = {}

        # Extract features of interest
        data_file["DEPTH"] = ds.coords["DEPTH"].values
        data_file["LONGITUDE"] = ds["LONGITUDE"].values
        data_file["LATITUDE"] = ds["LATITUDE"].values
        data_file["DAY"] = ds["day"].values
        time = get_datetime_from_xr(ds) # Convert time into np array, dtype = datetime.datetime
        data_file["TIME"] = time
        data_file["SEASON"] = get_seasons_from_datetime(time) # Get seasons(south hemisphere) from datetime
        
        # Get channels
        try:
            channels = [ds.attrs["channel"]]
        except KeyError:
            channels = ds.coords["CHANNEL"].values
            channels = [x.decode('utf-8').strip() for x in channels]
        
        # Get index of channel of interest in Sv array
        try : 
            i_chan = channels.index(channel)
        except IndexError :
            print(f"{channel} not in {channels}")
        
        # Only keep channel of interest in Sv array
        Sv = ds["Sv"].values
        if Sv.ndim > 2:
            Sv = Sv[:, :, i_chan]
        data_file["Sv"] = Sv

        # Get title of current file
        title = ds.attrs["title"]
        print(title)
    
        # Save current file in pkl
        with open(pickle_file, 'ab') as p_file:
            pkl.dump((title, data_file), p_file)
        
        # Close dataset
        ds.close()
        
    return pickle_file


#%%------------------------------------------------------ filter by bathymetry
def filter_by_bathymetry_gebco(pkl_path, gebco_path, bath_thr) :
    print('\n filter by GEBCO \n')
    # Does not create a new pkl
    temp_path = pkl_path + ".tmp" 

    # open gebco
    gebco = xr.open_dataset(gebco_path)
    gebco_bath = gebco["elevation"].values
    gebco_lat = gebco["lat"].values
    gebco_lon = gebco["lon"].values

    with open(pkl_path, 'rb') as pkl_file, open(temp_path, 'wb') as temp_file : 
        while True :
            try : 
                name_traj, data_dict = pkl.load(pkl_file)
                print(name_traj)

                # Find bathymetry of data in data_dict
                latitude = data_dict["LATITUDE"]
                longitude = data_dict["LONGITUDE"]

                # find indexes of (lat, lon) in gebco_bath
                lat_tree = KDTree(gebco_lat[:, None])  # Create KDTree for latitude
                lon_tree = KDTree(gebco_lon[:, None])  # Create KDTree for longitude
                
                lat_closest_indexes= lat_tree.query(latitude[:, None])[1] 
                lon_closest_indexes = lon_tree.query(longitude[:, None])[1]

                # find bathymetry for each point of ds
                bath = gebco_bath[lat_closest_indexes, lon_closest_indexes]
                
                # Filter data_dict by bath
                filtered_dict = {"DEPTH": data_dict["DEPTH"], "BATH":bath}

                # remove data where depth is inferior to thr
                mask = np.abs(bath) >= np.abs(bath_thr) # true if data is conserved
                
                if not np.any(mask):  
                    print(f"Trajectory removed (no valid points)")
                    continue  # Skip storing this trajectory

                bath_filtered = bath[mask]
                filtered_dict["BATH"] = bath_filtered
                
                # Prepare data for the specific season
                for key, array in data_dict.items():
                    if key not in ["DEPTH"]:
                        filtered_dict[key] = array[mask]

                # Save filtered data to the temporary file
                pkl.dump((name_traj, filtered_dict), temp_file)
                

            except EOFError : 
                print("end of file")
                break
    # Replace original file with the filtered one
    os.replace(temp_path, pkl_path)
    print("Filtering complete. Original file updated.")

#%% ------------------------------------------------------------- crop depth
def remove_extreme_depth(pkl_path) :
    print("\n REMOVE XTREME DEPTHS \n")
    min_depth = 20
    max_depth = 1000

    # Does not create a new pkl
    temp_path = pkl_path + ".tmp"

    with open(pkl_path, 'rb') as pkl_file, open(temp_path, 'wb') as temp_file : 
        while True :
            try : 
                name_traj, data_dict = pkl.load(pkl_file)
                sv = data_dict["Sv"]
                depth = data_dict["DEPTH"]

                # find index 
                index_min_depth = np.searchsorted(depth, min_depth, side='right')
                index_max_depth = np.searchsorted(depth, max_depth, side='right')

                # Crop Sv and depth 
                depth = depth[index_min_depth:index_max_depth]
                sv = sv[:, index_min_depth:index_max_depth]

                # Update data_dict
                data_dict["DEPTH"] = depth
                data_dict["Sv"] = sv

                # Save filtered data to the temporary file
                pkl.dump((name_traj, data_dict), temp_file)  
                     
            except EOFError : 
                break

    # Replace original file with the filtered one
    os.replace(temp_path, pkl_path)


#%% ------------------------------------------------------ separate data by season and period

def separate_by_season(src_path:str)->None:
    """
    Splits data from the source pickle file into separate season-wise pickle files 
    and saves them. It handles the removal of existing season files, reading the 
    source file, splitting the data based on seasons, and saving it into appropriate 
    files.

    Args:
        src_path (str): Path to the source pickle file containing the original data. 
                            The script will split the data by seasons and create new files based on this source.

    Outputs:
        Four pickle files: one for each of the following seasons:
            - <source_filename>_winter.pkl
            - <source_filename>_spring.pkl
            - <source_filename>_summer.pkl
            - <source_filename>_fall.pkl
    """
    print("\n SPLIT BY SEASON \n")

    # Create 4 pkl files, one per season
    src_basename = os.path.basename(src_path)
    src_filename = os.path.splitext(src_basename)[0]
    dir = os.path.dirname(src_path)
    pkl_basename =  dir + "/"+ src_filename

    pkl_seasons = {
        1: pkl_basename + "_winter.pkl",
        2: pkl_basename + "_spring.pkl",
        3: pkl_basename + "_summer.pkl",
        4: pkl_basename + "_fall.pkl"
    }

    # Remove existing pkl files if they exist
    for pickle_file in pkl_seasons.values():
        if os.path.exists(pickle_file):
            os.remove(pickle_file)

    # Open source pickle file
    with open(src_path, 'rb') as pkl_file:
        while True:
            try:
                # Load pkl file
                data_title, data_dict = pkl.load(pkl_file)

                # Organize data per season
                try:
                    seasons = data_dict["SEASON"]
                    unique_seasons = np.unique(seasons)
                    print(unique_seasons)
                except KeyError:
                    print("SEASON not found in keys of dictionary containing data")
                    break

                # Handle data for files recorded during only 1 season
                if len(unique_seasons) == 1:
                    season = seasons[0]
                    with open(pkl_seasons[season], 'ab') as p_file_season:
                        pkl.dump((data_title, data_dict), p_file_season)

                # Handle data for files recorded across multiple seasons
                else:
                    print("Handling multiple seasons in one file.")
                    for season in unique_seasons:
                        indices_season = np.where(seasons == season)[0]
                        dict_season = {"DEPTH": data_dict["DEPTH"]}

                        # Prepare data for the specific season
                        for key, array in data_dict.items():
                            if key not in ["DEPTH"]:
                                dict_season[key] = array[indices_season]

                        with open(pkl_seasons[season+1], 'ab') as p_file_season:
                            pkl.dump((data_title, dict_season), p_file_season)

            except EOFError:
                print("End of file")
                break
    return pkl_seasons

def separate_by_period(src_path):
    """
    Splits data from the source pickle file into separate period-wise pickle files 
    and saves them. The data is organized by 'DAY' and saved into four distinct pickle files: 
    day, sunset, sunrise, and night. If the pickle files already exist, they are deleted before 
    being recreated with the split data.

    Args:
        src_path (str): Path to the source pickle file containing the original data. 
                         The script will split the data by periods (day, sunset, sunrise, night) 
                         and create new files based on this source.
        dest_path (str): Path to the directory where the output pickle files will be saved.

    Outputs:
        Four pickle files: one for each of the following periods:
            - <source_filename>_day.pkl
            - <source_filename>_sunset.pkl
            - <source_filename>_sunrise.pkl
            - <source_filename>_night.pkl
    """
    print("\n SPLIT BY PERIOD \n")

    # Get source file base name without extension
    src_basename = os.path.basename(src_path)
    src_filename = os.path.splitext(src_basename)[0]
    dir = os.path.dirname(src_path)
    pkl_basename =  dir + "/"+ src_filename

    # Define pickle file paths for each period
    pkl_periods = {
        1: pkl_basename + "_day.pkl",
        2: pkl_basename + "_sunset.pkl",
        3: pkl_basename + "_sunrise.pkl",
        4: pkl_basename + "_night.pkl"
    }

    # Remove existing pickle files if they exist
    for file in pkl_periods.values():
        if os.path.exists(file):
            os.remove(file)

    # Open the source pickle file and read the data in stream
    with open(src_path, 'rb') as pkl_file:
        while True:
            try:
                # Load data
                traj_title, data_dict = pkl.load(pkl_file)

                try:
                    days = data_dict["DAY"]
                    unique_days = np.unique(days)
                except KeyError:
                    print("DAY not found in keys of dictionary containing data")
                    break

                # Filter data by day
                for day in unique_days:
                    day = int(day)  # Ensure day is an integer
                    indices_days = np.where(days == day)[0]
                    dict_day = {
                        "DEPTH": data_dict["DEPTH"]
                    }

                    # Prepare the filtered data for each key
                    for key, array in data_dict.items():
                        if key in ["DEPTH"]:
                            continue
                        filtered_array = array[indices_days]
                        dict_day[key] = filtered_array
                    
                    # Save the data for the specific day to the corresponding pkl file
                    with open(pkl_periods[day], 'ab') as p_file_day:
                        pkl.dump((traj_title, dict_day), p_file_day)

            except EOFError:
                print("End of file.")
                break

    return pkl_periods