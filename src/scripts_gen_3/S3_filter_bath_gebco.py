from scipy.spatial import KDTree
import os
import sys
import numpy as np
import pickle as pkl
import xarray as xr

def filter_by_bathymetry_gebco(src_path, dest_path, gebco_path, bath_thr) :
    # Create pkl file
    if not dest_path.endswith("/"):
        dest_path += "/"
    src_basename = os.path.basename(src_path)
    src_filename = os.path.splitext(src_basename)[0]
    pkl_filename = dest_path + src_filename +"_bath.pkl"

    if os.path.exists(pkl_filename):
        os.remove(pkl_filename)
        print(f"{pkl_filename} removed")

    # open gebco
    gebco = xr.open_dataset(gebco_path)
    gebco_bath = gebco["elevation"].values
    gebco_lat = gebco["lat"].values
    gebco_lon = gebco["lon"].values

    with open(src_path, 'rb') as pkl_file : 
        while True :
            try : 
                name_traj, data_dict = pkl.load(pkl_file)
                print(name_traj)

                # Find bathymetry of data in data_dict
                latitude = data_dict["LATITUDE"]
                longitude = data_dict["LONGITUDE"]

                # find indexes of (lat, lon) in gebco_bath
                lat_tree = KDTree(gebco_lat[:, None])  # Create KDTree for latitude
                lon_tree = KDTree(gebco_lon[:, None])  # Create KDTree for longitude
                
                lat_closest_indexes= lat_tree.query(latitude[:, None])[1] 
                lon_closest_indexes = lon_tree.query(longitude[:, None])[1]

                # find bathymetry for each point of ds
                bath = gebco_bath[lat_closest_indexes, lon_closest_indexes]

                # Filter data_dict by bath
                dict_filtered = {"DEPTH": data_dict["DEPTH"], "BATH":bath}

                # find data where depth at least *bath_thr* meters
                mask =  bath < bath_thr
                mask_unique = np.unique(mask)

                bath_filtered = bath[mask]
                print(bath_filtered)
                
                # Prepare data for the specific season
                for key, array in data_dict.items():
                    if key not in ["DEPTH", "in_ROI"]:
                        dict_filtered[key] = array[mask]
                if False in mask_unique :
                    print("before filter : ", data_dict["Sv"].shape,  "after : ", dict_filtered["Sv"].shape)
                    
                with open(pkl_filename, 'ab') as pkl_filtered:
                    pkl.dump((name_traj, dict_filtered), pkl_filtered)
                

            except EOFError : 
                print("end of file")
                break

if __name__ == "__main__":
    src_path = sys.argv[1]
    dest_path = sys.argv[2]
    gebco_path = sys.argv[3]
    bath_thr = int(sys.argv[4])
    
    filter_by_bathymetry_gebco(src_path, dest_path, gebco_path, bath_thr)
    