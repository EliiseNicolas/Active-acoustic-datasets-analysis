import os
import xarray as xr
import pickle
import sys
from basic_functions import get_datetime_from_xr, get_seasons_from_datetime

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