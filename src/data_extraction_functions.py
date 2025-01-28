"""
Wrote by Elise Nicolas
January 2025

This file is meant to provide basic functions to extract data from netCDF files
"""
#%% ------------------------ Imports
import netCDF4 as nc
from netCDF4 import Dataset
import numpy as np
import datetime
from typing import List


#%% ------------------------ Extract and convert time/date data
def get_datetime(dataset:Dataset, convert_datetime=True)->np.ndarray : 
    """
    Extracts and converts the TIME variable from a netCDF4 Dataset object into 
    an array of human-readable datetime strings in the format '%Y-%m-%d %H:%M:%S'.

    Parameters:
    ----------
    dataset : Dataset
        The netCDF4 Dataset object containing the TIME variable.
    convert_datetime : bool (default : True)
        If True : convert gregoriant time into array of datetime strings in the format '%Y-%m-%d %H:%M:%S'
        If False, the function returns the raw Gregorian times as computed by nc.num2date.

    Returns:
    -------
    np.ndarray
        If convert_datetime=True: Returns an array of datetime objects
        If convert_datetime=False: Returns an array of Gregorian datetime objects (from cftime).
    """

    try :
        time_var = dataset.variables['TIME'] # time since 1950-01-01 00:OO:OO UTC
        date_var = nc.num2date(time_var[:], units=time_var.units, calendar=getattr(time_var, 'calendar', 'standard')) # Convert in gregorian date format
        if convert_datetime : 
            date_var = np.array([date.strftime("%Y-%m-%d %H:%M:%S") for date in date_var]) # convert in %Y-%m-%d %H:%M:%S format
            date_var = np.array([datetime.datetime.strptime(dt, "%Y-%m-%d %H:%M:%S") for dt in date_var])
        return date_var
    except ValueError:
        print("ValueError: Could not find the TIME variable in the dataset.")
        return None

# Modifier : faire code de int pour chaque saison et retourner np.array of shape(n,) (vector)
def get_season_from_datetime(dataset:Dataset) -> List[str]: 
    """
    Extracts the season(s) from a dataset based on the datetime of each sample.

    This function assumes that the dataset contains a TIME variable which is 
    converted into a list of datetime objects. It then sorts the datetime 
    values and determines in which season each sample was recorded, returning
    a list of corresponding seasons.

    Parameters:
    ----------
    dataset : Dataset
        The netCDF4 Dataset object containing the TIME variable, which is 
        used to extract datetime values for each sample.

    Returns:
    -------
    List of len (n) with n the number of samples in dataset
        A numpy arrayy containing strings representing the season(s) for each sample as values. 
        Possible values are "winter", "spring", "summer", and "fall".
        Seasons are calculated based on north hemisphere seasons.
    """
    
    # Get datetime in '%Y-%m-%d %H:%M:%S' format
    datetime_list = get_datetime(dataset)

    # Initialisation 
    seasons=[]

    for datetime in datetime_list : 
        # Extract month and year from datetime format
        month = datetime.month
        season = ""

        # Add season in corresponding year of recording
        if month in [6, 7, 8]: # Winter 
             seasons.append("winter")
        elif month in [9, 10, 11]: # Spring
            seasons.append("spring")
        elif month in [12, 1, 2]: # Summer
            seasons.append("summer")
        elif month in [3, 4, 5]: # fall
            seasons.append("fall")

    return seasons

    # def get_seasons_and_periods_from_DAY


#%% -------------------------------------------- Extract channels 
def get_channels(dataset:Dataset)->np.ndarray : 
    try : 
        channel= dataset.variables["CHANNEL"][:]
        decoded_channel = np.char.decode(channel, 'utf-8')
        decoded_channel = [''.join(row).strip() for row in decoded_channel]  
        return decoded_channel
    except ValueError:
        print("ValueError: Could not find the CHANNEL variable in the dataset.")
        return None 

# %%
