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
                              "01_projects/01_thesisdata/02_circdis/"
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

# define the keywords for reading in the file
init_kwargs = {
    "subdir_name": subdir_name,
    "func": (prep, "read_file_to_df"),
    "index_col": [0, 1],
    "header": [0]
}

# define the keywords to plot the file
plot_kwargs = {
    "function": (wave, "plot_wave_from_df"),
    "remove_col": False,
    "showfig": False,
    "savefig": False,
    "figsize": (15,10),
    "ignore_index": False,
}

# apply process to both activity and sleep
for input, save in zip(input_list, output_list):

    # copy the kwargs
    curr_init = init_kwargs
    curr_plot = plot_kwargs

    # modify the init and plot kwargs if necessary
    curr_init["input_directory"] = input
    curr_init["save_directory"] = save

    # process all the files
    wave_object = prep.SaveObjectPipeline(**curr_init)
    wave_object.create_plot(**curr_plot)

