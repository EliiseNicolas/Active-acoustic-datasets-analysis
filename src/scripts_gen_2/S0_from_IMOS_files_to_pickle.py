import os
import xarray as xr
import pickle
import sys
from basic_functions import get_datetime_from_xr, get_seasons_from_datetime, are_points_in_polygon

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
    
    for file in list_files:
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
            print(i_chan)
        except IndexError :
            print(f"{channel} not in {channels}")
        
        # Only keep channel of interest in Sv array
        Sv = ds["Sv"].values
        if Sv.ndim > 2:
            Sv = Sv[:, :, i_chan]
        data_file["Sv"] = Sv


        # Put key "in_ROI" in dict
        mask = are_points_in_polygon(data_file["LONGITUDE"], data_file["LATITUDE"])
        if True in mask:
            data_file["in_ROI"] = True  
        else:
            data_file["in_ROI"] = False

        # Get title of current file
        title = ds.attrs["title"]
    
        # Save current file in pkl
        with open(pickle_file, 'ab') as p_file:
            pickle.dump((title, data_file), p_file)
        
        # Close dataset
        ds.close()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <source_folder> <destination_folder> <pickle_file_name> [variable_name]")
        sys.exit(1)

    src_path = sys.argv[1]
    dest_path = sys.argv[2]
    channel = sys.argv[3]

    extract_and_save_netcdf_data(src_path, dest_path, channel)
