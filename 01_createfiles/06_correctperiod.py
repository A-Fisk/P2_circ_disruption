# find period max and correct the file for that period
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import sys
sys.path.insert(0, "/Users/angusfisk/Documents/01_PhD_files/"
                    "07_python_package/actiPy")
import actiPy.preprocessing as prep
import actiPy.waveform as wave

# define the input directoryies
activity_dir = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/"
                              "01_projects/P2_Circ_Disruption_paper_chapt2/"
                              "01_data_files/01_activity/00_clean")
sleep_dir = activity_dir.parents[1] / "02_sleep/00_clean"
input_list = [activity_dir, sleep_dir]

# define the save directories
activity_save = activity_dir.parents[2] / \
                '03_analysis_outputs/01_activity/'
sleep_save = activity_save.parents[0] / '02_sleep/'
output_list = [activity_save, sleep_save]

# define the subdirectory in the save directory to create and save in
subdir_name = '04_waveform/01_24hr'


file = sorted(activity_dir.glob("*.csv"))[-1]
df = prep.read_file_to_df(file, index_col=[0, 1])

# perform periodogram - LS from astropy easiest

from astropy.stats import LombScargle



# to define frequency per unit of observation, need to calculate
# how many cycles per unit time
# define boundaries of search
low_time = "20H"
high_time = "30H"
base_time = "10S"

from actiPy.preprocessing import _drop_level_decorator
import seaborn as sns
sns.set()

@_drop_level_decorator
def _period_df(data,
               animal_no: int=0,
               low_time: str="20H",
               high_time: str="30H",
               base_time: str="10S",
               base_unit: str="s"):
    """
    Applies Lombscargle periodogram for given data
    :param data:
    :param low_time:
    :param high_time:
    :param base_time:
    :return:
    """
    
    # create x and y values of observations
    time = np.linspace(0, len(data), len(data))
    y = data.iloc[:, animal_no]
    
    # convert all times into the same units (seconds)
    base_secs = pd.Timedelta(base_time).total_seconds()
    low_secs = pd.Timedelta(low_time).total_seconds()
    high_secs = pd.Timedelta(high_time).total_seconds()

    # frequency is number of 1/ cycles per base = base / cycles
    low_freq = base_secs / low_secs
    high_freq = base_secs / high_secs
    frequency = np.linspace(high_freq, low_freq, 1000)

    # find the LombScargle power at each frequency point
    power = LombScargle(time, y).power(frequency)

    # create index of timedeltas for dataframe
    index = pd.to_timedelta((1/frequency), unit=base_unit) * base_secs

    # create df out of the power values
    power_df = pd.DataFrame(power, index=index)
    
    return power_df


grouped_dict = {}
for animal, label in enumerate(df.columns[:-1]):
    grouped_periods = df.groupby(level=0).apply(_period_df,
                                                animal_no=animal,
                                                reset_level=False)
    grouped_dict[label] = grouped_periods
    
power_df = pd.concat(grouped_dict, axis=1)

def idxmax_level(data,
                 level_drop: int=0):
    """
    Drops the given level and returns idx max
    :param data:
    :param level_drop:
    :return:
    """
    data.index = data.index.droplevel(level_drop)
    max_values = data.idxmax()
    
    return max_values

max_values = power_df.groupby(level=0).apply(idxmax_level)

# split each animal based on the period for each time
# need to do in a loop as groupby can't pass in new period value each time

vals = df.index.get_level_values(0).unique()
split_dict = {}
# get the split based on internal period for each animal
for animal_no, column in enumerate(df.columns[:-1]):
    
    temp_dict = {}
    
    # for each animal get the split based on the correct period
    for val_no, val in enumerate(vals):
        period = max_values.iloc[val_no, animal_no]
        data_to_split = df.loc[val]
    
        temp_df = prep.split_dataframe_by_period(data_to_split,
                                                 drop_level=False,
                                                 reset_level=False,
                                                 animal_number=animal_no,
                                                 period=period)
        print(val, len(temp_df.columns))
        temp_dict[val] = temp_df
    
    animal_split = pd.concat(temp_dict)
    split_dict[column] = animal_split
    
fully_split = pd.concat(split_dict, axis=1)



power_agg = animal_split.aggregate(np.mean, axis=1)



