import pickle
import numpy as np
import os
import sys
from basic_functions import *

def split_and_save_data_by_season(src_path:str, dest_path:str)->None:
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
    # Create 4 pkl files, one per season
    if not dest_path.endswith("/"):
        dest_path += "/"
    src_basename = os.path.basename(src_path)
    src_filename = os.path.splitext(src_basename)[0]
    pkl_basename = dest_path + src_filename

    pkl_winter = pkl_basename + "_winter.pkl"
    pkl_spring = pkl_basename + "_spring.pkl"
    pkl_summer = pkl_basename + "_summer.pkl"
    pkl_fall = pkl_basename + "_fall.pkl"
    list_seasons_pkl = [pkl_winter, pkl_spring, pkl_summer, pkl_fall]

    # Remove existing pkl files if they exist
    for pickle_file in list_seasons_pkl:
        if os.path.exists(pickle_file):
            os.remove(pickle_file)

    # Open source pickle file
    with open(src_path, 'rb') as pkl_file:
        while True:
            try:
                # Load pkl file
                data_title, data_dict = pickle.load(pkl_file)

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
                    with open(list_seasons_pkl[season], 'ab') as p_file_season:
                        pickle.dump((data_title, data_dict), p_file_season)

                # Handle data for files recorded across multiple seasons
                else:
                    print("Handling multiple seasons in one file.")
                    for season in unique_seasons:
                        indices_season = np.where(seasons == season)[0]
                        dict_season = {"DEPTH": data_dict["DEPTH"], "in_ROI":data_dict["in_ROI"]}

                        # Prepare data for the specific season
                        for key, array in data_dict.items():
                            if key not in ["DEPTH", "in_ROI"]:
                                dict_season[key] = array[indices_season]

                        with open(list_seasons_pkl[season], 'ab') as p_file_season:
                            pickle.dump((data_title, dict_season), p_file_season)

            except EOFError:
                print("End of file")
                break


if __name__ == "__main__":
    # Get source pickle file path from command line argument
    src_path = sys.argv[1]
    dest_path = sys.argv[2]
    
    # Call the function to split and save the data
    split_and_save_data_by_season(src_path, dest_path)
