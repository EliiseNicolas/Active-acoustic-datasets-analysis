import os
import pickle as pkl
import numpy as np
import sys

def filter_by_date(src_path, dest_path):
    """
    Filters data from a pickle file by detecting significant time gaps and saves the segmented data.

    Parameters:
    - src_path (str): Path to the source pickle file.
    - dest_path (str): Path where the filtered pickle file will be saved.

    Outputs:
    - A new pickle file containing filtered data segmented by detected time periods.
    """
    if not dest_path.endswith("/"):
        dest_path += "/"
    
    src_basename = os.path.basename(src_path)
    src_filename = os.path.splitext(src_basename)[0]
    pkl_date = os.path.join(dest_path, f"{src_filename}_by_date.pkl")
    
    if os.path.exists(pkl_date):
        os.remove(pkl_date)

    with open(src_path, 'rb') as pkl_file:
        while True:
            try:
                title, data_dict = pkl.load(pkl_file)
                print(title)

                time = data_dict["TIME"]
                tuple_dt_hour = np.array([(t, t.hour) for t in time])
                index_base = 0
                
                for i in range(len(tuple_dt_hour) - 1): 
                    d, h = tuple_dt_hour[i]
                    d_next, h_next = tuple_dt_hour[i + 1]
                    
                    if h_next > h + 4:  # Detecting time gap of at least 4 hours
                        index_thr = i + 1
                        dict_by_date = {"DEPTH": data_dict["DEPTH"]}
                        
                        for key, array in data_dict.items():
                            if key not in ["DEPTH", "in_ROI"]:
                                dict_by_date[key] = array[index_base:index_thr]
                        
                        with open(pkl_date, 'ab') as p_file_day:
                            pkl.dump((d, dict_by_date), p_file_day)
                        
                        index_base = index_thr

            except EOFError:
                print("End of file")
                break

if __name__ == "__main__":
    # Get source and destination paths from command line arguments
    src_path = sys.argv[1]
    dest_path = sys.argv[2]
    
    # Call the function to split and save the data
    filter_by_date(src_path, dest_path)