# Function to test the output of the split by period function

import pandas as pd

import sys

sys.path.append('../')

import P2_analysis_lib as library


file_path = '/Users/angusfisk/Documents/1_PhD_files/2_Test_data/2_PIR_Exp_test_data/6C.csv'

test_data = pd.read_csv(file_path,
                        index_col=0,
                        parse_dates=True)

split_data = library.split_data_by_period(test_data, 1)

print(split_data.head())
