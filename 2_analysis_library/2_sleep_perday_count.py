# Function to test the output of the split by period function

import pandas as pd

import sys

import matplotlib
import matplotlib.pyplot as plt

sys.path.append('../')

import P2_analysis_lib as library

file_path = '/Users/angusfisk/Documents/1_PhD_files/2_Test_data/2_PIR_Exp_test_data/6C_sleep_.csv'

test_data = pd.read_csv(file_path,
                        index_col=0,
                        parse_dates=True)

test_data_removed_cols = library.remove_object_col(test_data)

first_animal_split_standarddays = library.split_data_by_period(test_data, 2)

count = library.sleep_count(first_animal_split_standarddays)

plt.plot(count)

plt.show()
