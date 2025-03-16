import os
import pickle as pkl
import sys
import numpy as np

def split_and_save_data_by_period(src_path, dest_path):
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
    # Get source file base name without extension
    if not dest_path.endswith("/"):
        dest_path += "/"
    src_basename = os.path.basename(src_path)
    src_filename = os.path.splitext(src_basename)[0]

    # Define pickle file paths for each period
    pkl_files = {
        1: os.path.join(dest_path, f"{src_filename}_day.pkl"),
        2: os.path.join(dest_path, f"{src_filename}_sunset.pkl"),
        3: os.path.join(dest_path, f"{src_filename}_sunrise.pkl"),
        4: os.path.join(dest_path, f"{src_filename}_night.pkl")
    }

    # Remove existing pickle files if they exist
    for file in pkl_files.values():
        if os.path.exists(file):
            os.remove(file)

    # Open the source pickle file and read the data in stream
    with open(src_path, 'rb') as pkl_file:
        print("hey")
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
                        if key in ["DEPTH", "in_ROI"]:
                            continue
                        filtered_array = array[indices_days]
                        dict_day[key] = filtered_array
                    
                    # Save the data for the specific day to the corresponding pkl file
                    with open(pkl_files[day], 'ab') as p_file_day:
                        pkl.dump((traj_title, dict_day), p_file_day)

            except EOFError:
                print("End of file.")
                break

if __name__ == "__main__":
    # Get source and destination paths from command line arguments
    src_path = sys.argv[1]
    dest_path = sys.argv[2]
    
    # Call the function to split and save the data
    split_and_save_data_by_period(src_path, dest_path)
