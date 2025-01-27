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

    
def get_season(dataset:Dataset) -> List[str]: 
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
    List[str]
        A list of strings representing the season(s) for each sample. 
        Possible values are "winter", "spring", "summer", and "fall".
        The list will have the same length as the number of samples in the 
        dataset, with each element indicating the season of the corresponding 
        sample.
    """

    # Get datetime in '%Y-%m-%d %H:%M:%S' format
    datetime_list = get_datetime(dataset)

    # Sort datetime in increasing order
    sorted_datetime_objects = sorted(datetime_list)
    
    # Get season(s) in which data were recorded
    data_info = dict()
    for datetime in sorted_datetime_objects : 
        # Extract month and year from datetime format
        month = datetime.month
        year=datetime.year

        # Add year in dict
        if year not in data_info : 
            data_info[year] = []
        
        # Add season in corresponding year of recording
        if month in [12, 1, 2]: # Winter 
            if "winter" not in data_info[year] :
                data_info[year].append("winter")
        elif month in [3, 4, 5]: # Spring
            if "spring" not in data_info[year] : 
                data_info[year].append("spring")
        elif month in [6, 7, 8]: # Summer
            if "summer" not in data_info[year] : 
                data_info[year].append("summer")
        elif month in [9, 10, 11]: # fall
            if "fall" not in data_info[year] :
                data_info[year].append("fall")

    return data_info

# %%
