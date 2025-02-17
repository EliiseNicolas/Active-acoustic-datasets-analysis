import pickle
import numpy as np
import os
import sys
from basic_functions import *

"""
Script for splitting and saving acoustic data by day from a pickle (.pkl) file.

This script reads a .pkl file containing acoustic data, extracts timestamps, 
and separates the data into 24-hour segments. Each segment is saved as a 
new .pkl file.

Usage:
------
Run from the terminal with the following command:

    python script.py <source_pkl_file> <destination_folder>

Arguments:
----------
- source_pkl_file (str) : Path to the .pkl file containing acoustic data.
- destination_folder (str) : Path to the folder where the new .pkl files will be saved.

Outputs:
--------
- Creates a new .pkl file for each unique date found in the dataset.
- Each output file is named: `<source_filename>_cropped_24h.pkl`.

Example:
--------
To split and save acoustic data by day:
    python script.py /path/to/data.pkl /path/to/save/

Notes:
------
- If a file with the same name already exists in the destination, it will be overwritten.
- The script preserves depth and channel information as they remain constant.
"""

# Get args
src_path = sys.argv[1]
dest_path = sys.argv[2]

# Create Pickle file
src_basename = os.path.basename(src_path)
src_filename = os.path.splitext(src_basename)[0]
pkl_twenty_four_h = dest_path + src_filename + "_cropped_24h.pkl"

# Remove files that are named the same
if os.path.exists(pkl_twenty_four_h):
        os.remove(pkl_twenty_four_h)

with open(src_path, 'rb') as p_file:
    # read data in pkl as stream
    while True:
        try:
            # Load data
            data_title, data_file = pickle.load(p_file) 
            
            # Get times values
            times = data_file["TIME"]

            # Extract dates from datetime
            dates = np.array([time.date() for time in times])
            dates_uniques = np.unique(dates)
            
            for d in dates_uniques : 
                # Create dict that will be of this shape {date : data_dict}
                dict_by_day={}

                # Add Channel and depth in dict, there size won't change regarding the date
                dict_by_day["CHANNEL"]=data_file["CHANNEL"]
                dict_by_day["DEPTH"]=data_file["DEPTH"]

                # Get indices where data is at date d (for every array)
                indices_days= np.where(dates == d)[0]

                for key, array in data_file.items():
                    if key in ["DEPTH", "CHANNEL"] : 
                        continue

                    # Crop array to get only data related to date d
                    cropped_array = array[indices_days]

                    # Add data related to date d in dict
                    dict_by_day[key]=cropped_array

                # Save dict containing date d in pkl file
                with open(pkl_twenty_four_h, 'ab') as dest_file_day:
                    pickle.dump((d,    dict_by_day), dest_file_day)
                    
        # End of file
        except EOFError:
            print("End of file")
            break