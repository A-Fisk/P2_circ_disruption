import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import sys
import numpy as np
sys.path.insert(0, "/Users/angusfisk/Documents/01_PhD_files/"
                    "07_python_package/actiPy")
import actiPy.preprocessing as prep
import actiPy.episodes as ep

# define the input directoryies
activity_dir = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/"
                              "01_projects/P2_Circ_Disruption_paper_chapt2/"
                              "01_data_files/01_activity/01_episodes")
sleep_dir = activity_dir.parents[1] / "02_sleep/01_episodes"
input_list = [activity_dir, sleep_dir]

# define the save directories
activity_save = activity_dir.parents[2] / '03_analysis_outputs/01_activity'
sleep_save = activity_save.parents[0] / '02_sleep'
output_list = [activity_save, sleep_save]

# define the subdirectory in the save directory to create and save in
subdir_name = '03_episodes'

# define the keywords for reading in the file
init_kwargs = {
    "subdir_name": subdir_name,
    "func": (prep, "read_file_to_df"),
    "index_col": [0, 1],
    "header": [0]
}

# define the keywords to process the file
plot_kwargs = {
    "function": (ep, "episode_histogram"),
    "logy": False,
    "logx": True,
    "remove_col": False,
    "showfig": False,
    "savefig": True,
    "xlabel": "Episode Duration (seconds)",
    "figsize": (10,5),
    "bins": np.geomspace(10, 3600, 10),
    "clip": True,
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
    ep_object = prep.SaveObjectPipeline(**curr_init)
    ep_object.create_plot(**curr_plot)
    
