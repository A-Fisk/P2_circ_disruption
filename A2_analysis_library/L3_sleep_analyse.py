# Script to define functions for analysing sleep

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pathlib
import sys
sys.path.append('..')
from A2_analysis_library.L1_preprocessing import remove_object_col

def sleep_process(data, window=4):
    """
    Function to score the PIR based activity data as sleep

    :param data: pandas dataframe to be sleep scored
    :param window: length of window, default =4 which is 40 seconds
    :return: dataframe of sleep
    """
    # if >40 seconds (4 rows) of inactivity score as 1

    rolling_sum_data = data.rolling(window).sum()

    scored_data = rolling_sum_data == 0

    return scored_data.astype(int)

# Function to take input file name and save sleep_df in the same directory
def sleep_create_file(file_path, index_col=0):
    """
    Function to process a csv and save a sleep scored csv in the same directory

    :param file_path: string - path to file name
    :param index_col: int - column to use to create index in read_csv
    """

    # grab the directory from the file name
    # read the file as a dataframe
    # remove object columns and save them
    # perform the sleep_processing
    # add object columns back in
    # save the sleep dataframe as a new csv in the directory

    path = pathlib.Path(file_path)

    directory = path.parent

    old_file_name = path.name

    data = pd.read_csv(file_path,
                       index_col=index_col)

    # remove object columns

    data, columns = remove_object_col(data, return_cols=True)

    # sleep process the data

    sleep_df = sleep_process(data)

    # add object columns back in

    for col in columns:

        sleep_df[col.name] = col

    # save as new df

    new_file_name = directory.joinpath(old_file_name[:-4] + "_sleep_.csv")

    sleep_df.to_csv(new_file_name)

    # return columns so can print them as a progress indicator

    return columns

# Function to get hourly sum of sleep data
def create_hourly_sum(data, index_col=-1):
    """
    function that takes in a datetimeindex indexed pandas dataframe of PIR sleep scored data
    Returns as resampled into hourly bins, including the labels

    :param data:
    :param index_col:
    :return:
    """

    # resample the data with sum method
    # resample the index column with the first method
    # add index col back into the summed df
    # return the df

    df_hourly_sum = data.resample("H").sum()

    df_start = data.resample("H").first()

    # grab the column name from the original dataframe to put in the hourly sum

    col_name = data.iloc[:, index_col].name

    df_hourly_sum[col_name] = df_start.iloc[:, index_col]

    return df_hourly_sum

# Function to save the csv to a specified directory
def save_sleep_csv_file(data, destination_dir, file_name):
    """
    Function to save the dataframe as a csv in the specified directory
    :param data:
    :param destination_dir:
    :param file_name:
    :return:
    """

    # Define where to put the file and the file name then save the file there

    destination_directory = destination_dir

    file_name_to_use = file_name

    destination = pathlib.Path(destination_directory, file_name_to_use)

    data.to_csv(destination)

# Function to pipeline the creating of the hourly sum then saving it to a specified directory
def hourly_sleep_save_file_pipeline(read_file_name, destination_dir):
    """
    Function to read the sleep file from the directory, process it into hourly sum, and then save it in specified
    directory

    :param read_file_name:
    :param destination_dir:
    :param save_file_name:
    :return:
    """

    # Read the file in as a dataframe
    # process it for hourly sum
    # create save file_name and destination
    # save it there

    # read file

    data = pd.read_csv(read_file_name,
                       index_col=0,
                       parse_dates=True)

    # process it for hourly sum

    data_hourly = create_hourly_sum(data)

    # create name and save

    file_name_only = read_file_name.parts[-1]

    save_sleep_csv_file(data_hourly, destination_dir, file_name_only)

# Function to plot data and save to file
def simple_plot(data, destination_dir='.', file_name="test.svg", savefig=False, showfig=True):
    """
    Function take in pandas dataframe, plot it as subplots, and then save to specified place
    :param data:
    :param destination_dir:
    :param file_name:
    :return:
    """

    # create the destination to save as

    destination_directory = destination_dir

    file_name_to_use = file_name

    destination = pathlib.Path(destination_directory, file_name_to_use)

    # plot the file

    no_rows = len(data.columns)

    fig, ax = plt.subplots(nrows=no_rows, sharey=True)

    for axis, col in enumerate(data.columns):

        ax[axis].plot(data[col])

    if showfig:

        plt.show()

    if savefig:

        plt.savefig(destination, dpi=500)
