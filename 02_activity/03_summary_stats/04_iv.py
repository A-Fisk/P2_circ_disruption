# Script to calculate IV for all files animals and conditions
# find period max and correct the file for that period
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()
import pathlib
import sys
sys.path.insert(0, "/Users/angusfisk/Documents/01_PhD_files/"
                    "07_python_package/actiPy")
import actiPy.preprocessing as prep
import actiPy.analysis as als

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
subdir_name = '03_summarystats/01_intradayvar'

# define the keywords for reading in the file
init_kwargs = {
    "subdir_name": subdir_name,
    "func": (prep, "read_file_to_df"),
    "index_col": [0, 1],
    "header": [0]
}

# define the keywords to process the file
process_kwargs = {
    "function": (als, "intradayvar"),
    "create_df": True,
    "savedfcsv": True,
    "set_name": False,
}

# define plotting kwargs
plot_kwargs = {
    "set_name_title": False,
    "set_file_title": False,
    "showfig": False,
    "savefig": True,
    "figsize": (10, 10)
}


for input, save in zip(input_list, output_list):

    # copy the kwargs
    curr_init = init_kwargs
    curr_process = process_kwargs
    curr_plot = plot_kwargs

    # modify the init and plot kwargs if necessary
    curr_init["input_directory"] = input
    curr_init["save_directory"] = save

    # process all the files
    iv_obj = prep.SaveObjectPipeline(**curr_init)
    iv_obj.process_file(**curr_process)
    
    df = iv_obj.processed_df

    curr_plot["fname"] = iv_obj.processed_df_filename.with_suffix(".png")
    disrupted = als.normalise_to_baseline(df)

    als.catplot(disrupted, **curr_plot)



#
# # calculate IV
# # ratio of variance of first derivative to total variance
# iv = als.iv_by_group(df)
#
#
# norm_to_baseline = iv - iv.iloc[0]
# plot = norm_to_baseline.iloc[:, :-1].unstack().reset_index()
#
# sns.catplot(data=plot, x="light_period", y=0)
# sns.pointplot(data=plot, x="light_period", y=0, capsize=0.2)
#

