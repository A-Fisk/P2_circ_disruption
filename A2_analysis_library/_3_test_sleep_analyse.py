# Testing module for sleep analysis

import unittest
import numpy as np
import pandas as pd
import _3_sleep_analyse

# test sleep processing

class test_sleep_processing(unittest.TestCase):

    def setUp(self):
        # generate data for sleep processing
        # list of 100's with known number of 4x0s

        high_no_list = [100 for x in range(100)]

        self.test_data_series = pd.Series(high_no_list)

        self.test_data_series[10:14] = 0

        self.test_data_series[57:61] = 0

        self.test_data_series[82:86] = 0

    def test_sleep_processing(self):
        # test whether picks up correct number of sleep episodes

        sleep_scored_data = _3_sleep_analyse.sleep_process(self.test_data_series)

        sleep_scored_sum = sleep_scored_data.sum()

        self.assertEqual(3, sleep_scored_sum)

if __name__ == "__main__":
    unittest.main()



