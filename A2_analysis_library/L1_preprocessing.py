import pandas as pd


# This script contains functions which are useful for preprocessing of PIR data

# function to remove column if object

def remove_object_col(data, return_cols=False):
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
            current_col = data.loc[:, column]

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


# Function to read files in as a pandas dataframe in standard way
def read_file_to_df(file_name):
    """
    function to take given csv file name and turn it into a df
    :param file_name:
    :return:
    """

    # quick error handling to see if is a csv file

    if file_name.suffix != ".csv":
        raise ValueError("Not a csv file")

    df = pd.read_csv(file_name,
                     index_col=0,
                     parse_dates=True)

    return df


# Function to check subdirectory and create if doesn't exist
def create_subdir(input_directory, subdir_name):
    """
    Function takes in Path object of input_directory and string of subdir name, adds them together, checks if it
    exists and if it doesn't creates new directory

    :param input_directory:
    :param subdir_name:
    :return:
    """

    # create path name
    # check if exists
    # create it if it doesn't exist
    # return path name

    sub_dir_path = input_directory / subdir_name

    if not sub_dir_path.exists():
        sub_dir_path.mkdir()

    return sub_dir_path


# Function to create file_name_path
def create_file_name_path(directory, file, save_suffix):
    """
    Simple function to put together directory, file.stem, and suffix to create path
    :param directory:
    :param file:
    :param save_suffix:
    :return:
    """

    # combine directory, file stem and save suffix
    file_path = directory / (file.stem + save_suffix)

    return file_path


# Create save_pipeline class with objects for saving csv and plots depending on the method used
class SaveObjectPipeline:
    """
    Class object for saving data to a file. Main object used for processing data and saving it to a directory
    Separate methods for saving dataframes to csv files and for creating and saving plots to a file

    init method globs all the files in the input_directory and creates a df_list with dataframes of all the files in
    the input_directory

    Takes the arguments initially of
    Input_directory - place to search for files to process
    Subdir_name - name for subdirectory to be created in input_directory to hold new files
    search_suffix - default .csv, name to glob for files in input_directory

    """

    # init method to create attributes
    def __init__(self, input_directory, search_suffix=".csv"):

        self.processed_list = []

        self.input_directory = input_directory

        self.search_suffix = search_suffix

        # create the file list by globbing for the search suffix
        self.file_list = list(self.input_directory.glob("*" + self.search_suffix))

        # read all the dfs into a list
        self.df_list = []

        for file in self.file_list:
            temp_df = read_file_to_df(file)

            self.df_list.append(temp_df)

    # method for saving a csv file
    def save_csv_file(self, function_name, subdir_name, save_suffix):
        """
        Method that applies a defined function to all the dataframes in the df_list and saves them to the subdir that
        is also created

        :param function_name:
        :param save_suffix:
        :param subdir_name
        :return:
        """

        # create the subdirectory
        # For every df in the list
        # apply the function
        # create the name to save it
        # save the df there
        # Save to a processed list so can use for plotting

        subdir_path = create_subdir(self.input_directory, subdir_name=subdir_name)

        for df, file in zip(self.df_list, self.file_list):
            temp_df = function_name(df)

            file_name_path = create_file_name_path(subdir_path, file, save_suffix)

            temp_df.to_csv(file_name_path)

            self.processed_list.append(temp_df)

    # method for saving a plot
    def create_plot(self, function_name, subdir_name, data_list=None, save_suffix='.png', showfig=False,
                    savefig=True, dpi=300):
        """
        Method to take each df and apply given plot function and save to file
        Default parameters of showfig = False and savefig = True but can be changed

        :type save_suffix: str
        :param save_suffix:
        :param dpi:
        :param function_name:
        :param showfig:
        :param savefig:
        :return:
        """

        # create subdir
        # for every df in the list
        # create the save name
        # remove the object col
        # apply the plotting function, passing savefig and showfig arguments to the function and the path to save name

        # define the data list
        if data_list is None:
            data_list = self.processed_list

        # create the subdir
        subdir_path = create_subdir(self.input_directory, subdir_name=subdir_name)

        # loop through the dfs and pass to plotting function (saves by default))
        for df, file in zip(data_list, self.file_list):
            file_name_path = create_file_name_path(subdir_path, file, save_suffix)

            temp_df = remove_object_col(df, return_cols=False)

            function_name(temp_df,
                          file_name_path,
                          showfig=showfig,
                          savefig=savefig,
                          dpi=dpi)
