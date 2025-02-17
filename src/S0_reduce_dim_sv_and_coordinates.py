import os
import xarray as xr
import pickle
import sys
from basic_functions import get_datetime_from_xr, get_seasons_from_datetime

"""
Script for extracting and saving data from NetCDF (.nc) files into a pickle (.pkl) file.

This script reads NetCDF files from a specified directory, extracts relevant data including time, depth, 
longitude, latitude, season, and the chosen variable (default: "Sv"). It then organizes and saves this data into a pickle file 
for further processing. The pickle file is created in the specified destination path, and any existing file with the same name is 
overwritten.

Usage:
------
Run from the terminal with the following command:

    python script.py <source_folder> <destination_folder> <pickle_file_name> [variable_name]

Arguments:
----------
- source_folder (str) : Path to the folder containing NetCDF (.nc) files to be processed.
- destination_folder (str) : Path to the folder where the resulting pickle file will be saved.
- pickle_file_name (str) : Name of the output pickle file (without the ".pkl" extension).
- variable_name (str, optional) : The variable to extract from the NetCDF files. Default is "Sv".

Outputs:
--------
- A pickle file containing a dictionary for each NetCDF file processed. Each dictionary contains:
    - "TIME" : List of datetime objects corresponding to the time in the NetCDF file.
    - "DEPTH" : Depth values.
    - "SEASON" : Season corresponding to each time.
    - "LONGITUDE" : Longitude values.
    - "LATITUDE" : Latitude values.
    - "CHANNEL" : Channel information.
    - The selected variable (default "Sv") data as a NumPy array.
    - "DAY" : Day values from the NetCDF file.
    - "TITLE" : Title of the dataset.

Example:
--------
To process NetCDF files and save the results in a pickle file:
    python script.py /path/to/nc_files /path/to/save /output_pickle_name

To specify a variable (e.g., "Temperature"):
    python script.py /path/to/nc_files /path/to/save /output_pickle_name Temperature
"""


# Get args
src_path = sys.argv[1]
dest_path = sys.argv[2]
pickle_file_name = sys.argv[3]
if len(sys.argv) > 4 :
    var_name = sys.argv[4]
else : 
     var_name = "Sv"

# Create pickle dict
if not dest_path.endswith("/") :
     dest_path += "/"
pickle_file = dest_path + pickle_file_name + ".pkl"

if os.path.exists(pickle_file):
    os.remove(pickle_file)

# Get every file of folder
list_files = [f for f in os.listdir(src_path) if f.endswith('.nc')]

for i, file in enumerate(list_files) :

    # Load file in xarray.Dataset
    file_path = os.path.join(src_path, file)
    ds=xr.open_dataset(file_path)

    # Create dict containing data of current file
    data_file = {}

    # Keep coordinates (time, depth, longitude, latitude, channels)
    data_file["DEPTH"] = ds.coords["DEPTH"].values
    time = get_datetime_from_xr(ds) # convert into datetime
    data_file["TIME"] = time
    data_file["SEASON"] = get_seasons_from_datetime(time) # convert datetime into season
    data_file["LONGITUDE"] = ds["LONGITUDE"].values
    data_file["LATITUDE"] = ds["LATITUDE"].values
    
    try : 
        # Not every dataset has the channel data in its attributes, sometimes it is in its coords
        channel = [ds.attrs["channel"]]
    except : 
        channel = ds.coords["CHANNEL"].values
        channel = [x.decode('utf-8').strip() for x in channel]
    finally :
            data_file["CHANNEL"]=channel


    # Keep only variable of interest and day
    data_file["Sv"] = ds[var_name].values #np.ndarray of shape (n_pings, d_depths, n_channels)
    data_file["DAY"]= ds["day"].values

    # Get title of file
    title = ds.attrs["title"]

    # Save selected data of current dataset in pickle file
    with open(pickle_file, 'ab') as p_file:
        pickle.dump((title, data_file), p_file)

    # Close xarray.Dataset
    ds.close()  