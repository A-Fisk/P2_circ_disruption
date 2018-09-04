# Script to define functions for analysing sleep

import pandas as pd
import pathlib
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
