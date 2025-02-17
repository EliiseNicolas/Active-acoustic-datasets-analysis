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
- source_pkl_file (str) : Path to the source .pkl file containing the original data. The script will split the data by periods and create new files based on this source.

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
s1_src_path = sys.argv[1]

# Create 4 pkl files, one per day
s0_filename, _ = os.path.splitext(s1_src_path) # get the name of s0 file without ".pkl" extension
pkl_day = s0_filename+ "_day.pkl"
pkl_sunset = s0_filename + "_sunset.pkl"
pkl_sunrise = s0_filename + "_sunrise.pkl"
pkl_night = s0_filename + "_night.pkl"
list_days_pkl=[pkl_day, pkl_sunset, pkl_sunrise, pkl_night]

for pickle_file in list_days_pkl : 
    print(pickle_file)
    if os.path.exists(pickle_file):
        os.remove(pickle_file)

# Open pkl src file (s0)
with open(s1_src_path, 'rb') as p_file:
    # read data in pkl as stream
    while True:
        try:
            data_title, data_file = pickle.load(p_file) 
            print("Batch chargé :", data_title)  
            print("Clés du batch :", list(data_file.keys()))

            # Organize data per day
            try : # Get days of data_file
                days = data_file["DAY"]
                unique_days = np.unique(days)
            except : 
                print("DAY not found in keys of dictionnary containing data")
                break
            
            # Split data and put data in every corresponding day_pickle file
            else : 
                print("Handling multiple days in one file.")
                for day in unique_days :
                    indices_days= np.where(days == day)[0]
                    dict_day= {}
                    dict_day["CHANNEL"]=data_file["CHANNEL"]
                    dict_day["DEPTH"]=data_file["DEPTH"]
                    
                    for key, array in data_file.items():
                        if key in ["DEPTH", "CHANNEL"] : 
                            continue
                        cropped_array = array[indices_days]
                        # print(cropped_array[0], array[indices_days[0]]) 
                        dict_day[key] = cropped_array
                    
                    with open(list_days_pkl[int(day-1)], 'ab') as p_file_day:
                        pickle.dump((data_title,    dict_day), p_file_day)

        except EOFError:
            print("End of file.")
            break