#%% ------------------------ Imports
import os
import numpy as np
import datetime
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import pandas as pd
import shutil
from shapely.geometry import Point, Polygon
import pickle as pkl

#%% ------------------------ Extract time(float) and convert into datetime
def get_datetime_from_xr(dataset: xr.Dataset) -> np.ndarray:
    """
    Extracts and converts the TIME variable from an xarray Dataset object into 
    an array of human-readable datetime objects.

    Parameters:
    ----------
    dataset : xr.Dataset
        The xarray Dataset object containing the TIME variable.

    Returns:
    -------
    np.ndarray
        An array of datetime objects of shape (n, 1)
    """

    try:
        time_var = dataset['TIME'].values

    except KeyError:
        print("KeyError: Could not find the TIME variable in the dataset.")
        return None
    
    datetime_array = np.array(time_var.astype('datetime64[s]').tolist())
    
    return datetime_array



#%% -------------------------------------------------------------- Get seasons from datetime
def get_seasons_from_datetime(datetime_array:np.ndarray[datetime.datetime]) -> np.ndarray[int]: 
    """
    Get corresponding season of every datetime.datetime object in a np.ndarray

    Parameters:
    ----------
    datetime_array : np.ndarray[datetime.datetime]
        An array of shape (n, 1) containing datetime.datetime objects
        

    Returns:
    -------
    np.ndarray[int] of shape(n,) with n the number of samples in dataset
        A numpy array containing int representing the season(s) for each sample 
        Possible values are 0, 1, 2, 3 corresponding respectively to "winter", "spring", "summer", and "fall".
        Seasons are calculated based on SOUTH hemisphere seasons.
    """
    # Initialize seasons
    seasons=np.zeros(len(datetime_array), dtype=int)

    # Get seasons
    for i, datetime in enumerate(datetime_array) : 
        # Extract month and year from datetime format
        month = datetime.month

        # Add season in corresponding year of recording
        if month in [6, 7, 8]: # Winter 
            seasons[i]= 0
        elif month in [9, 10, 11]: # Spring
            seasons[i]= 1
        elif month in [12, 1, 2]: # Summer
            seasons[i]= 2
        elif month in [3, 4, 5]: # fall
            seasons[i]= 3

    return seasons

# %% ------------------------------------------ sample rate analysis

def get_sample_rate_cropped_files(file_path:str) :
    sample_rates = []
    with open(file_path, 'rb') as p_file:
        # read data in pkl as stream
        while True : 
            try:
                date, date_dict = pkl.load(p_file)
                print(date)
                mean_sample_rate = 0

                # Get datetime
                time = date_dict["TIME"]
                if (time.shape[0]==1) : 
                    continue
                
                for i in range(time.shape[0]-1) : 
                    mean_sample_rate+= (time[i+1] - time[i]).total_seconds()
                mean_sample_rate=mean_sample_rate/(time.shape[0]-1)

                if mean_sample_rate> 40 : 
                    print("unusual data : ", date, mean_sample_rate)
                
                sample_rates.append(mean_sample_rate)
            
            except EOFError:
                print("End of file.")
                break
        sample_rates = np.array(sample_rates)
        return sample_rates, np.mean(sample_rates), np.std(sample_rates)

#%%--------------------------
def get_enveloppe_convexe_into_xr(path:str="/home/elise/Documents/M1-BIM/S2/active_acoutics_analysis_sea_elephants/data/geographic_data/convex_hull.xlsx") : 
    """
    Loads and converts a convex hull (enveloppe convexe) geographic dataset 
    into an xarray dataset.

    Returns:
    -------
    xr.Dataset
        The convex hull dataset.
    """
    
    enveloppe_convexe = path
    df = pd.read_excel(enveloppe_convexe)
    df = df.rename(columns={'lon': 'LONGITUDE', 'lat': 'LATITUDE'})
    ds = xr.Dataset.from_dataframe(df)
    return ds

def get_enveloppe_convexe_into_list_tuple()-> List[Tuple[float]] : 
    """
    Loads and converts a convex hull (enveloppe convexe) geographic dataset 
    into a list of Tuple (points).

    Returns:
    -------
    List[Tuple[float]]
        The convex hull dataset.
    """
    
    enveloppe_convexe = "/home/elise/Documents/M1-BIM/S2/active_acoutics_analysis_sea_elephants/data/geographic_data/convex_hull.xlsx"
    df = pd.read_excel(enveloppe_convexe)
    l = []
    for lon, lat in zip(df["lon"], df["lat"]):
        point = (lon, lat)
        l.append(point)
    return l
    

#%% -------------------- Display trajectories 
def display_all_trajectories_folder(pkl_path:str, save:bool=False, dest_path:str="") :
    print("hey")
    plt.clf()
    # Create whole fig in which we will put every trajectories
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})

    # Add Title to figure
    src_basename = os.path.basename(pkl_path)
    src_filename = os.path.splitext(src_basename)[0]
    plt.title("trajectories " + src_filename)

    # Add coastlines to figure"
    ax.coastlines()

    # Add enveloppe to figure           
    enveloppe = get_enveloppe_convexe_into_xr()
    ax.scatter(enveloppe['LONGITUDE'].values, enveloppe['LATITUDE'].values, color='green', s=2, transform=ccrs.PlateCarree(), label="Convex hull")

    # Iterate on every trajectory in pkl_file
    with open(pkl_path, 'rb') as pkl_file : 
        while True : 
            try : 
                # Get data
                title, data_dict = pkl.load(pkl_file)
                longitude = data_dict["LONGITUDE"]
                latitude = data_dict["LATITUDE"]

                # Plot trajectory in figure
                ax.scatter(longitude, latitude, s=2, transform=ccrs.PlateCarree())

            # End of file
            except EOFError : 
                print("Enf of file")
                print(dest_path + "_trajectories" + src_filename + ".png")
                
                if save : 
                    plt.savefig(dest_path + "trajectories_" + src_filename + ".png", dpi=300, bbox_inches="tight")

                # Display figure
                plt.tight_layout()

                # plt.legend()
                plt.show()
                break


def are_points_in_polygon(longitude:np.ndarray[float], latitude:np.ndarray[float])-> np.ndarray[bool]:
    """
    Check if each point (longitude, latitude) is inside the given polygon.

    Parameters:
    ----------
    longitude : np.ndarray[float]
        Array containing the longitudes of the points.
    latitude : np.ndarray[float]
        Array containing the latitudes of the points.


    Returns:
    -------
    np.ndarray[bool]
        Boolean array indicating whether each point is inside the polygon (`True`) or not (`False`).
    """

    polygon = Polygon(get_enveloppe_convexe_into_list_tuple()) # The ROI
    points = [Point(lon, lat) for lon, lat in zip(longitude, latitude)]

    return np.array([polygon.contains(point) for point in points])



def get_mean_var_in_out_ROI(src_path:str, channel:int=0)-> Tuple[float, float]:
    mean_in_ROI, mean_out_ROI = np.zeros(240), np.zeros(240)
    std_in_ROI, std_out_ROI = np.zeros(240), np.zeros(240)
    count = 0

    with open(src_path, 'rb') as pkl_file : 
        while True :
            try : 
                _, data_dict = pkl.load(pkl_file)
                d = data_dict["DEPTH"]
                in_ROI = data_dict["in_ROI"]
                Sv = data_dict["Sv"]
                if Sv.ndim>2 : 
                    Sv = Sv[:,:,channel]

                if data_dict["in_ROI"]:
                    print("inROI")
                    if not (Sv.size == 0 or np.all(np.isnan(Sv))):
                        sv_masked = np.ma.masked_invalid(Sv)  # Masquer les NaN dans le tableau
                        mean_Sv = np.mean(sv_masked, axis=0)  # Calculer la moyenne en ignorant les NaN
                        
                        # if np.count_nonzero(~np.isnan(Sv)) > 2: 
                        #     std_value = np.nanstd(Sv, axis=0)
                        #     std_Sv += std_value
                    
                    assert (mean_Sv.shape == mean_in_ROI.shape)
                    # assert(std_Sv.shape == std_in_ROI.shape)

                    if in_ROI : 
                        mean_in_ROI+= mean_Sv
                        # std_in_ROI+= std_Sv

                    else : 
                        print(mean_Sv)
                        mean_out_ROI+=mean_Sv
                        # std_out_ROI+=std_Sv

                count+=1

            except EOFError :
                print("end of file")
                # Normalization
                mean_without_0_r = np.where(mean_in_ROI==0, np.nan, mean_in_ROI)
                mean_without_0_o = np.where(mean_out_ROI==0, np.nan, mean_out_ROI)
                mean_r = 10 * np.log10(mean_without_0_r / count)
                mean_o = 10 * np.log10(mean_without_0_o / count)
                # mean_in_ROI, mean_out_ROI = mean_in_ROI/count, mean_out_ROI/count
                # std_in_ROI, std_out_ROI = std_in_ROI/count, std_out_ROI/count
                # return d, (mean_in_ROI, std_in_ROI), (mean_out_ROI, std_out_ROI)
                return d, mean_r, mean_o


def boxplot_mean_std(depth: np.ndarray, inROI: Tuple[np.ndarray], outROI: Tuple[np.ndarray], title:str, save:bool=False, dest_path:str="") :
    mean_inROI, std_inROI = inROI
    mean_outROI, std_outROI = outROI
    
    # Création du graphique
    plt.figure(figsize=(10, 6))

    # Tracer le backscattering moyen avec les barres d'erreur (écart-type)
    plt.errorbar(mean_inROI, -depth, xerr=std_inROI, fmt='o', color='b', ecolor='r', capsize=3, label="Mean backscattering strenght in ROI")
    plt.errorbar(mean_outROI, -depth, xerr=std_outROI, fmt='o', color='g', ecolor='purple', capsize=3, label="Mean backscattering strenght out ROI")

    # plt.gca().invert_yaxis()
    # Ajouter des labels et un titre
    plt.ylabel("Depth (m)")
    plt.xlabel("Mean backscattering strength (dB re m⁻¹)")
    plt.title(title)
    plt.legend()

    # Afficher le graphique
    plt.grid(True)

    # Save
    if save : 
        fig_path = dest_path + "mean_Sv_by_depth_in_out_ROI_boxplot_"+title
        plt.savefig(f"{fig_path}.png", dpi=300, bbox_inches="tight")

    plt.show()


def plot_mean_backscatter(src_path, dest_path) :
    d, in_ROI, out_ROI = get_mean_var_in_out_ROI(src_path)
    src_basename = os.path.basename(src_path)
    src_filename = os.path.splitext(src_basename)[0]
    title = src_filename
    boxplot_mean_std(d, in_ROI, out_ROI, save=True, dest_path=dest_path, title=title)

    
