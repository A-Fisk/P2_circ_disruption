
# This script contains functions which are useful for preprocessing of PIR data

# function to remove column if object

def remove_object_col(data, return_cols = False):
    """
    Function to check the data type in each column and drop it if it is an object
    Does not distinguish between float, int, strings
    :param data: pandas dataframe to check
    :param return_cols: Boolean - default False, returns columns as a list
    :return: pandas dataframe without object columns
    :return: if return_cols is true, returns list of dropped columns
    """

    # Check each column type
    # drop the columns that are objects
    # return the dataframe

    dropped_cols = []

    for column in data.columns:

        column_data = data.loc[:, column]

        if column_data.dtype == 'O':

            current_col = data.loc[:,column]

            dropped_cols.append(current_col)

            data = data.drop(column, axis=1)

    if return_cols:

        return data, dropped_cols

    else:

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


