import pickle as pkl
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

"""
Script for filtering and saving bathymetry data from a pickle (.pkl) file.

This script reads a source pickle file containing volume backscattering data (Sv), filters the data based on depth and NaN thresholds, and saves the valid data into a new pickle file. The filtering process ensures that the percentage of missing values remains below a specified threshold.

Usage:
------
Run from the terminal with the following command:

    python script.py <source_pkl_file> <destination_folder> [NaN_threshold] [depth_threshold]

Arguments:
----------
- source_pkl_file (str) : Path to the source pickle file containing bathymetry data.
- destination_folder (str) : Path where the filtered pickle file will be saved.
- NaN_threshold (float, optional) : Maximum allowed fraction of missing values per depth. Default is 0.5.
- depth_threshold (float, optional) : Depth limit for filtering the Sv data. Default is 1002.5.

Outputs:
--------
- A new pickle file containing filtered bathymetry data, named:
    - <source_filename>_filtered_bathymetry.pkl

Filtering criteria:
-------------------
- The Sv data is processed up to the specified depth threshold.
- Depths with more than the specified fraction of NaN values are considered invalid.
- If fewer than 5 depths exceed the NaN threshold, the data is saved; otherwise, it is discarded.

Example:
--------
To filter bathymetry data with default thresholds:
    python script.py /path/to/source.pkl /path/to/destination/

To specify custom thresholds:
    python script.py /path/to/source.pkl /path/to/destination/ 0.3 800
"""

# Get args
src_path = sys.argv[1]   # src path
dest_path = sys.argv[2]   # dest path
if len(sys.argv) > 3 :
    NaN_thr = sys.argv[3] 
    depth_thr = sys.argv[4]
else : 
    NaN_thr = 0.5
    depth_thr = 1002.5


# Create pkl file 
src_basename = os.path.basename(src_path)
src_filename = os.path.splitext(src_basename)[0]
pkl_24h_bathymetry_filtered = dest_path + src_filename + "_filtered_bathymetry.pkl"

if os.path.exists(pkl_24h_bathymetry_filtered):
        os.remove(pkl_24h_bathymetry_filtered)
        print(f"{pkl_24h_bathymetry_filtered} removed")
        
with open(src_path, 'rb') as pkl_file : 
    while True : 
        try : 
            # Load data
            date, data_dict = pkl.load(pkl_file)

            # Get volume backscattering
            Sv = data_dict["Sv"]
            if Sv.ndim > 2 : # Select only 18kHz channel
                Sv = Sv[:,:,0]
            depth = data_dict["DEPTH"]

            # Get threshold index
            index_thr = int(np.where(depth == depth_thr)[0])
            print(index_thr)
            assert(depth[index_thr]==depth_thr)
            
            # Select sv that should not be empty
            selected_sv = Sv[:,:index_thr]
            # print(selected_sv)
            n_samples, n_depth = selected_sv.shape

            # Count missing values
            print(selected_sv.shape)
            n_NaN = np.sum(np.isnan(selected_sv), axis=0)/n_samples
            
            # If missing values absent (perc_Nan > NaN_thr) for every depth : 
            count_NaN = 0 # number of depth where percentage of missing values abnormally high
            for d in range(n_depth) : 
                if n_NaN[d] >= NaN_thr : 
                    count_NaN += 1
            print(count_NaN)

            if count_NaN < 5 : 
                print("OK")
                # Save dict containing date d in pkl file
                with open(pkl_24h_bathymetry_filtered, 'ab') as dest_file_day:
                    pkl.dump((date,    data_dict), dest_file_day)
            else : 
                print("Not OK, file not saved")


        except EOFError : 
            print("end of file")
            break