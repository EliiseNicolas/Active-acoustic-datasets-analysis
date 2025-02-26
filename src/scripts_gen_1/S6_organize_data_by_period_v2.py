import os
import pickle
import sys
import numpy as np

"""
Script for splitting and saving data from a pickle (.pkl) file into separate period-wise files.

This script reads data from a source pickle file, organizes it by day, and saves the data for each day into four distinct pickle files: one for each of the following: day, sunset, sunrise, and night. If the pickle files already exist, they are deleted before being recreated with the split data.

Usage:
------
Run from the terminal with the following command:

    python script.py <source_pkl_file>

Arguments:
----------
- source_pkl_file (str) : Path to the source .pkl file containing the data splitted by date. The script will split the data by periods and create new files based on this source.

Outputs:
--------
- Four pickle files: one for each of the following categories:
    - <source_filename>_day.pkl
    - <source_filename>_sunset.pkl
    - <source_filename>_sunrise.pkl
    - <source_filename>_night.pkl

Example:
--------
To split data from a source pickle file:
    python script.py /path/to/source_file.pkl
"""

# Get args
src_path = sys.argv[1]
dest_path = sys.argv[2]

# Create 4 pkl files, one per day
src_basename = os.path.basename(src_path)
src_filename = os.path.splitext(src_basename)[0]
pkl_files = {
    1: os.path.join(dest_path, f"{src_filename}_day.pkl"),
    2: os.path.join(dest_path, f"{src_filename}_sunset.pkl"),
    3: os.path.join(dest_path, f"{src_filename}_sunrise.pkl"),
    4: os.path.join(dest_path, f"{src_filename}_night.pkl")
}


for file in pkl_files.values():
    if os.path.exists(file):
        os.remove(file)

# Open pkl src file (s0)
with open(src_path, 'rb') as pkl_file:
    # read data in pkl as stream
    while True:
        try:
            date, data_dict = pickle.load(pkl_file)

            # Organize data per day
            try : # Get days of data_dict
                days = data_dict["DAY"]
                unique_days = np.unique(days)
            except : 
                print("DAY not found in keys of dictionnary containing data")
                break
            
            # Split data and put data in every corresponding day_pickle file
            
            for day in unique_days :
                day = int(day)
                indices_days= np.where(days == day)[0]
                dict_day= {}
                dict_day["CHANNEL"]=data_dict["CHANNEL"]
                dict_day["DEPTH"]=data_dict["DEPTH"]
                dict_day["in_ROI"]=data_dict["in_ROI"]
                
                for key, array in data_dict.items():
                    if key in ["DEPTH", "CHANNEL", "in_ROI"] : 
                        continue
                    filtered_array = array[indices_days]
                    if key == "TIME" : 
                        hours = np.unique(np.array([dt.strftime("%H") for dt in filtered_array]))
                        # print(day, hours)
                    dict_day[key] = filtered_array
                    
                with open(pkl_files[day], 'ab') as p_file_day:
                    print(day, pkl_files[day], dict_day["TIME"])
                    pickle.dump((date,    dict_day), p_file_day)

        except EOFError:
            print("End of file.")
            break