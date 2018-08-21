import pandas as pd
import numpy as np
import pathlib

####################################################################
# Sleep processing functions


# function to process whole data file and create output sleep scored

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
    # perform the sleep_processing
    # save the sleep dataframe as a new csv in the directory

    path = pathlib.PurePosixPath(file_path)

    directory = path.parent

    old_file_name = path.name

    data = pd.read_csv(file_path,
                       index_col=index_col)

    sleep_df = sleep_process(data)

    new_file_name = directory.joinpath(old_file_name[:-4] + "_sleep_.csv")

    sleep_df.to_csv(new_file_name)

######################################################################
# Generally useful functions for pre-processing data


# function to remove column if object

def remove_object_col(data):
    """
    Function to check the data type in each column and drop it if it is an object
    Does not distinguish between float, int, strings
    :param data: pandas dataframe to check
    :return: pandas dataframe without object columns
    """

    # Check each column type
    # drop the columns that are objects
    # return the dataframe

    for column in data.columns:

        column_data = data.loc[:, column]

        if column_data.dtype == 'O':

            data = data.drop(column, axis=1)

    return data


# Function to split dataframe into periods based on label_column
def separate_by_condition(data, label_col=-1):
    """
    Function to separate activity data based upon the condition defined by a label column.
    e.g. separate into "Baseline", "Disrupted", "Post_Baseline"
    :param data: Dataframe to split, requires label column
    :param label_col: int, which column to select based upon, default -1
    :return: list of dataframes, length of list determined by number of unique labels
    """

    # select the unique values in the label column
    # slice the data based upon the label column values
    # append to list and return list of separated dataframes

    unique_conditions = data.iloc[:, label_col].unique()

    list_of_dataframes_by_condition = []

    for condition in unique_conditions:

        temporary_sliced_data = data[data.iloc[:, label_col] == condition]

        list_of_dataframes_by_condition.append(temporary_sliced_data)

    return list_of_dataframes_by_condition




######################################################################

# function for processing of data further


# split by internal period

# Write function to split given dataframe into columns based on days, for a single column at a time
# Require column number as input
def split_data_by_period(data, animal_number, period=None):
    """
    Function to take a dataframe indexed by time and split it into columns based on a specified period
    Takes dataframe and selects single column which it then splits by the set period and returns a dataframe with
    each column as a new day
    :param data: time indexed pandas dataframe
    :param animal_number: column which will be selected to split
    :param period: period to be split by, in the format of "%H %T" - Default = 24H 0T
    :return: Dataframe Indexed by real time through the period, with each column being subsequent period, for a single
    column
    """

    if not period:

        period = "24H 0T"

    # Create index to slice by
    # Slice by the consecutive days
    # concatenate the list together into a new dataframe
    # use a new index for the new dataframe

    start, end = data.index[0], data.index[-1]

    list_of_days_index = pd.date_range(start=start, end=end, freq=period)

    data_by_day_list = []

    animal_label = data.columns[animal_number]

    for day_start, day_end in zip(list_of_days_index[:-1], list_of_days_index[1:]):

        day_data = data.loc[day_start:day_end, animal_label]

        data_by_day_list.append(day_data)

    # append the final day as well

    final_day_start = list_of_days_index[-1]

    final_day_data = data.loc[final_day_start:, animal_label]

    data_by_day_list.append(final_day_data)

    # before putting into large dataframe, need to alter the index

    values_by_day_list = []

    for day in data_by_day_list:

        values = day.reset_index().iloc[:,1]

        values_by_day_list.append(values)

    split_dataframe = pd.concat(values_by_day_list,
                                axis=1)

    # Now to create new index for the dataframe

    old_frequency_seconds = 86400 / len(split_dataframe)

    int_seconds = int(old_frequency_seconds)

    miliseconds = round((old_frequency_seconds - int_seconds) * 1000)

    new_index_frequency = str(int_seconds) + "S " + str(miliseconds) + "ms"

    new_index = pd.timedelta_range(start = '0S',
                                   freq=new_index_frequency,
                                   periods=len(split_dataframe))

    split_dataframe.index=new_index

    # Now to set the column numbers to be subsequent days

    days = len(values_by_day_list)

    split_dataframe.columns = range(days)

    # Now set the name of dataframe to be PIR name

    split_dataframe.name = animal_label

    return split_dataframe
