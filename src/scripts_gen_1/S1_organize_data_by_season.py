import pickle
import numpy as np
import os
import sys
from basic_functions import *

"""
Script for splitting and saving data from a pickle (.pkl) file into separate season-wise files.

This script reads data from a source pickle file, organizes it by season, and saves the data for each season 
into four distinct pickle files: one for each season (winter, spring, summer, and fall). 
If the pickle files already exist, they are deleted before being recreated with the split data.

Usage:
------
Run from the terminal with the following command:

    python script.py <source_pkl_file>

Arguments:
----------
- source_pkl_file (str) : Path to the source .pkl file containing the original data. 
The script will split the data by seasons and create new files based on this source.

Outputs:
--------
- Four pickle files: one for each of the following seasons:
    - <source_filename>_winter.pkl
    - <source_filename>_spring.pkl
    - <source_filename>_summer.pkl
    - <source_filename>_fall.pkl

Example:
--------
To split data from a source pickle file by season:
    python script.py /path/to/source_file.pkl
"""


# Get args
s0_src_path = sys.argv[1]

# Create 4 pkl files, one per season
s0_filename, _ = os.path.splitext(s0_src_path) # get the name of s0 file without ".pkl" extension
pkl_winter = s0_filename+ "_winter.pkl"
pkl_spring = s0_filename + "_spring.pkl"
pkl_summer = s0_filename + "_summer.pkl"
pkl_fall = s0_filename + "_fall.pkl"
list_seasons_pkl=[pkl_winter, pkl_spring, pkl_summer, pkl_fall]

# remove existing pkl file
for pickle_file in list_seasons_pkl : 
    if os.path.exists(pickle_file):
        os.remove(pickle_file)

# Open pkl src file (s0)
with open(s0_src_path, 'rb') as p_file:
    # read data in pkl as stream
    while True:
        try:
            data_title, data_file = pickle.load(p_file) 
            print("Batch chargé :", data_title)  
            print("Clés du batch :", list(data_file.keys()))

            # Organize data per season
            try : # Get seasons of data_file
                seasons = data_file["SEASON"]
                unique_seasons = np.unique(seasons)
            except : 
                print("SEASON not found in keys of dictionnary containing data")
                break
            
            # If data of current file recorded during only 1 season
            # Copy paste all data in corresponding season_pickle file
            if len(unique_seasons)==1 : 
                season = seasons[0]
                with open(list_seasons_pkl[season], 'ab') as p_file_season:
                    pickle.dump((data_title, data_file), p_file_season)

            # Else : current file recorded during several seasons
            # Split data and put data in every corresponding season_pickle file
            else : 
                print("Handling multiple seasons in one file.")
                for season in unique_seasons :
                    indices_season = np.where(seasons == season)[0]
                    dict_season = {}
                    dict_season["CHANNEL"]=data_file["CHANNEL"]
                    dict_season["DEPTH"]=data_file["DEPTH"]
                    for key, array in data_file.items():
                        if key in ["DEPTH", "CHANNEL"] : 
                            continue
                        cropped_array = array[indices_season]
                        # print(cropped_array[0], array[indices_season[0]]) #OK !
                        dict_season[key] = cropped_array

                    with open(list_seasons_pkl[season], 'ab') as p_file_season:
                        pickle.dump((data_title, dict_season), p_file_season)

        except EOFError:
            print("End of file")
            break