import pickle as pkl
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

"""
Script for visualizing echogram data from a pickle (.pkl) file.

This script reads acoustic backscatter data (Sv) from a .pkl file, extracts 
time, depth, and signal values, and generates an echogram. If specified, the 
figure can be saved in a given directory.

Usage:
------
Run from the terminal with the following command:

    python script.py <pkl_file> [save] [destination_path]

Arguments:
----------
- pkl_file (str) : Path to the .pkl file containing acoustic data.
- save (bool, optional) : If "true", saves the echogram figure. Default: False.
- destination_path (str, optional) : Folder to save the figure if save=True.

Outputs:
--------
- Displays the echogram plot.
- Saves the echogram as a .png file if `save` is True.

Example:
--------
To display an echogram:
    python script.py /path/to/data.pkl

To save the echogram:
    python script.py /path/to/data.pkl true /path/to/save/
"""

# Get args
pkl_path = sys.argv[1]   # src path

if len(sys.argv) > 2 :
    save = sys.argv[2].lower() == "true" # Convert string to boolean
    dest_path = sys.argv[3]
    if not dest_path.endswith("/"):
        dest_path += "/"
else : 
    save = False
    dest_path=""

with open(pkl_path, 'rb') as pkl_file:
        # read data in .pkl as stream
        while True:
            try:
                # Load data from .pkl
                date, data_dict = pkl.load(pkl_file) 

                # Get Sv
                Sv = data_dict["Sv"] 
                if Sv.ndim >2 :
                    Sv = Sv[:, :, 0] # select channel 18kHz if several channels in Sv
                
                # Get depth
                depth = data_dict["DEPTH"]

                # Get time
                time = data_dict["TIME"]

                # Convert time as str
                time_str = np.array([dt.strftime("%Y : %m:%d %H:%M") for dt in time])

                # Select only 10 times to put as xlabels in plot
                tick_indices = np.linspace(0, len(time) - 1, num=10, dtype=int)
                tick_labels = time_str[tick_indices]
                time_indices = np.array([i.strftime("%Y : %m:%d %H:%M") for i in time[tick_indices]])

                # Plot
                plt.pcolormesh(time_str, -depth, 10 * np.log10(Sv.T), shading='auto', cmap='jet')

                # Labels and title
                plt.colorbar(label="Sv (dB re 1m⁻¹)")
                plt.xlabel("Time")
                plt.xticks(time_indices, tick_labels, rotation=90)
                plt.ylabel("Depth (m)")
                plt.title(f"Echogram of acoustic data recorded the {date} for 18 kHz channel")

                # Save in precised location dest_path
                if save : 
                    src_basename = os.path.basename(pkl_path)
                    src_filename = os.path.splitext(src_basename)[0]
                    fig_path = dest_path + src_filename + "_"+str(date)
                    i=0
                    # change filename if already in path
                    while os.path.exists(fig_path):
                        fig_path = fig_path+"_"+str((i))
                        i+=1

                    print(fig_path)
                    plt.savefig(f"{fig_path}.png", dpi=300, bbox_inches="tight")

                # Show echogram
                # plt.show()
                plt.clf()

            # End of file
            except EOFError :
                print("End of file")
                break