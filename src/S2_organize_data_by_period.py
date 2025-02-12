import os
import pickle
import sys
import numpy as np

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