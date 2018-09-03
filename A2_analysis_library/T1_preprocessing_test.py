# This script tests the preprocessing steps

import unittest
import numpy as np
import pandas as pd
import L1_preprocessing

# define my test class

class tests_preprocessing(unittest.TestCase):

    def setUp(self):
        # Create lists with correct types of data
        # turn into a Dataframe

        np.random.seed(42)

        # lists with specific data types

        col_1_list = np.random.rand(100)

        col_2_list = np.random.randint(0, 1, size=100, dtype=int)

        col_3_list = np.random.randint(0, 1, size=100, dtype=int).astype("O")

        col_4_list = np.random.randint(10, 20, size=100, dtype=int).astype("O")

        list_of_columns = [col_1_list,
                           col_2_list,
                           col_3_list,
                           col_4_list]

        dtypes_to_use = {"0":"float",
                         "1":"int",
                         "2":"object",
                         "3":"object"}

        col_names = ["0", "1", "2", "3"]

        # Turn these lists into a dataframe
        # with the correct column names and dtypes for the test

        test_data_df = pd.DataFrame(list_of_columns).T

        test_data_df.columns = col_names

        test_data_df_typed = test_data_df.astype(dtype = dtypes_to_use)

        self.test_data_df = test_data_df_typed

    # first test for remove_object_col function
    def test_remove_object_col_function(self):

        removed_col_data = L1_preprocessing.remove_object_col(self.test_data_df)

        number_remaining_columns = len(removed_col_data.columns)

        self.assertEqual(2, number_remaining_columns)

# first test for remove_object_col function

if __name__ == "__main__":
    unittest.main()
