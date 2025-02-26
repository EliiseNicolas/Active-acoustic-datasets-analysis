import pickle as pkl
import numpy as np
import sys
from basic_functions import are_points_in_polygon
import os

"""
Script for labeling data based on whether points are inside a region of interest (ROI).

This script reads a source pickle file containing geospatial data, checks whether the coordinates (longitude and latitude) are inside a specified polygon (ROI), and labels the data accordingly. The labeled data is then saved in a new pickle file.

Usage:
------
Run from the terminal with the following command:

    python script.py <source_pkl_file> <destination_folder>

Arguments:
----------
- source_pkl_file (str) : Path to the source pickle file containing geospatial data (longitude, latitude, etc.).
- destination_folder (str) : Path where the output pickle file with labeled data will be saved.

Outputs:
--------
- A new pickle file containing labeled data, named:
    - <source_filename>_labelled_in_ROI.pkl

Labeling Criteria:
------------------
- The coordinates (longitude and latitude) are checked to see if they fall within the given region of interest (ROI).
- A new key `"in_ROI"` is added to each data dictionary with a boolean value indicating whether the points are inside the ROI.

Example:
--------
To label data based on the ROI:
    python script.py /path/to/source.pkl /path/to/destination/
"""


# Get args
src_path = sys.argv[1]   # src path
dest_path = sys.argv[2]   # dest path

# Create pkl file 
src_basename = os.path.basename(src_path)
src_filename = os.path.splitext(src_basename)[0]
pkl_labelled = dest_path + src_filename + "_labelled_in_ROI.pkl"

if os.path.exists(pkl_labelled):
        os.remove(pkl_labelled)
        print(f"{pkl_labelled} removed")
        

# Create pkl file 
src_basename = os.path.basename(src_path)
src_filename = os.path.splitext(src_basename)[0]
pkl_with_label_ROI = dest_path + src_filename + "_labelled.pkl"

with open(src_path, 'rb') as pkl_file : 
    while True : 
        try : 
            date, data_dict = pkl.load(pkl_file)
            lon = data_dict["LONGITUDE"]
            lat = data_dict["LATITUDE"]
            mask = are_points_in_polygon(lon, lat)
            if True in mask:
                print(date)
                data_dict["in_ROI"] = True  
            else:
                data_dict["in_ROI"] = False

            with open(pkl_with_label_ROI, 'ab') as dest_file:
                pkl.dump((date, data_dict), dest_file)

        except EOFError : 
            print("end of file")
            break 
    