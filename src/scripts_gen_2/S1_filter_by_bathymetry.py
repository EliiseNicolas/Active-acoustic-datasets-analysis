import pickle as pkl
import numpy as np
import sys
import os

def filter_and_save_bathymetry_data(src_path, dest_path, NaN_thr=0.5, depth_thr=1002.5):
    """
    Filters bathymetry data from a pickle file based on depth and NaN thresholds and saves the valid data.

    Parameters:
    - src_path (str): Path to the source pickle file containing bathymetry data.
    - dest_path (str): Path where the filtered pickle file will be saved.
    - NaN_thr (float, optional): Maximum allowed fraction of missing values per depth. Default is 0.5.
    - depth_thr (float, optional): Depth limit for filtering the Sv data. Default is 1002.5.

    Outputs:
    - A new pickle file containing filtered bathymetry data, named:
      <source_filename>_filtered_bathymetry.pkl
    """
    if not dest_path.endswith("/"):
        dest_path += "/"
    src_basename = os.path.basename(src_path)
    src_filename = os.path.splitext(src_basename)[0]
    pkl_filtered = os.path.join(dest_path, f"{src_filename}_filtered_bathymetry.pkl")

    if os.path.exists(pkl_filtered):
        os.remove(pkl_filtered)
        print(f"{pkl_filtered} removed")

    with open(src_path, 'rb') as pkl_file:
        while True:
            try:
                # Load data
                name_traj, data_dict = pkl.load(pkl_file)
                Sv = data_dict["Sv"]
                depth = data_dict["DEPTH"]

                index_thr = int(np.where(depth == depth_thr)[0])
                assert depth[index_thr] == depth_thr
                selected_sv = Sv[:, :index_thr]
                n_samples, n_depth = selected_sv.shape

                n_NaN = np.sum(np.isnan(selected_sv), axis=0) / n_samples
                count_NaN = sum(1 for d in range(n_depth) if n_NaN[d] >= NaN_thr)

                if count_NaN < 5:
                    print("OK")
                    with open(pkl_filtered, 'ab') as dest_file_day:
                        pkl.dump((name_traj, data_dict), dest_file_day)
                else:
                    print("Not OK, file not saved")

            except EOFError:
                print("End of file")
                break

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <source_pkl_file> <destination_folder> [NaN_threshold] [depth_threshold]")
        sys.exit(1)
    
    src_path = sys.argv[1]
    dest_path = sys.argv[2]
    NaN_thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    depth_thr = float(sys.argv[4]) if len(sys.argv) > 4 else 1002.5

    filter_and_save_bathymetry_data(src_path, dest_path, NaN_thr, depth_thr)
