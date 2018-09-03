# This script tests the preprocessing steps

import unittest
import os
import 1_preprocessing

# define my test class

class 1_tests_preprocessing(unittest.TestCase):

    def setUp(self):
        pass

    # first test for remove_object_col function
    def test_remove_object_col_function(self):

        test_data = pd.read_csv(os.path.join(os.getcwd(), 'T1_test_data', '1_remove_object_col_testdata.csv'))

        removed_col_data = remove_object_col(test_data)

        self.assertEqual(2, len(removed_col_data.columns()))

# first test for remove_object_col function

if __name__ == "__main__":
    unittest.main()
