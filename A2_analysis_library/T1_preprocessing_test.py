# This script tests the preprocessing steps

import unittest
import numpy as np
import pandas as pd
import pathlib
import L1_preprocessing

np.random.seed(42)

# define my test class

class tests_preprocessing(unittest.TestCase):
    """
    Class to test preprocessing
    Sets up dataframe with following parameters
    2 non-object columns
    index as pd.timestamped
    final column for periods has 3 separate values
    save df to a "./test.csv"

    Tests remove object column, separate by condition, and read from df in turn
    """

    def setUp(self):
        # Create lists with correct types of data
        # turn into a Dataframe

        # lists with specific data types

        col_1_list = np.random.rand(100)

        col_2_list = np.random.randint(0, 1, size=100, dtype=int)

        col_3_list = np.random.randint(0, 1, size=100, dtype=int).astype("O")

        col_4_list = np.random.randint(10, 20, size=100, dtype=int).astype("O")

        list_of_columns = [col_1_list,
                           col_2_list,
                           col_3_list,
                           col_4_list]

        # assigning the names and data types to use

        dtypes_to_use = {"0":"float",
                         "1":"int",
                         "2":"object",
                         "3":"object",
                         "periods":"object"}

        col_names = ["0", "1", "2", "3", "index"]

        # create the index

        self.start = '2018-01-01 00:00:00'

        index_1 = pd.date_range(start=self.start,
                                freq='10S',
                                periods=len(col_1_list))

        # Turn these lists into a dataframe
        # with the correct column names and dtypes for the test

        test_data_df = pd.DataFrame(list_of_columns).T

        test_data_df.index = index_1

        test_data_df.columns = col_names[:-1]

        # create the final index column as different values for sep condition test

        test_data_df["periods"] = 100

        test_data_df.iloc[:31,-1] = 5
        test_data_df.iloc[31:61, -1] = 50
        test_data_df.iloc[61:, -1] = 75

        # set the data types

        test_data_df_typed = test_data_df.astype(dtype = dtypes_to_use)

        self.test_data_df = test_data_df_typed

        # save it to a csv file.
        self.save_str = pathlib.Path("./test.csv")

        # error with pathlib objects to save_csv. string input fixes
        self.test_data_df.to_csv(str(self.save_str))

    # first test for remove_object_col function
    def test_remove_object_col_function(self):

        removed_col_data = L1_preprocessing.remove_object_col(self.test_data_df)

        number_remaining_columns = len(removed_col_data.columns)

        self.assertEqual(2, number_remaining_columns)

    # second test check separate by condition on df
    def test_separate_by_condition(self):
        separated_list = L1_preprocessing.separate_by_condition(self.test_data_df,
                                                                label_col=-1)

        number_of_separate_conditions = len(separated_list)

        self.assertEqual(3, number_of_separate_conditions)

    def test_read_file_to_df_funct(self):
        """
        Create df with index and 10 rows
        Save that to a csv file
        Read that csv file
        Check the index 0 is correct
        Check index 0 dtype is a timestamp
        Check length of df is correct.
        :return:
        """
        # check reads file correctly
        # use the function to read the file.
        df_testread = L1_preprocessing.read_file_to_df(self.save_str)

        # check index [0] is correct
        first_index = (df_testread.index[0])

        self.assertEqual(str(first_index), self.start)

        # check index is a timestamp.
        self.assertEqual(type(first_index), pd.Timestamp)

        # check length is correct
        self.assertEqual(len(df_testread), 100)

    def tearDown(self):
        # remove the file we created
        import os
        os.remove(str(self.save_str))


if __name__ == "__main__":
    unittest.main()


# TODO test for following:
# TODO create subdir
# TODO create file name path
# TODO SaveObjectPipeline - save csv file and create plot


