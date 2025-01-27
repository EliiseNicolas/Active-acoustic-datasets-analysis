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

def get_season(dataset : Dataset)->str : 
    # Get datetime in '%Y-%m-%d %H:%M:%S' format
    datetime_list = get_datetime(dataset)

    # Sort datetime in increasing order
    sorted_datetime_objects = sorted(datetime_list)
    
    # Get first and last sample and dertermine the season(s)
    start_time = sorted_datetime_objects[0]
    end_time = sorted_datetime_objects[len(sorted_datetime_objects)]

    print(start_time)

    