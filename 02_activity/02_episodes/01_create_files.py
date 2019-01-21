# Script for creating episode files

import pathlib
import sys
sys.path.insert(0, "/Users/angusfisk/Documents/01_PhD_files/"
                    "07_python_package/actiPy")
import actiPy.preprocessing as prep
import actiPy.episodes as ep
import actiPy.actogram_plot as act

# define the input directoryies
activity_dir = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/"
                              "01_projects/01_thesisdata/02_circdis/"
                              "01_data_files/01_activity/00_clean")
sleep_dir = activity_dir.parents[1] / "02_sleep/00_clean"
input_list = [activity_dir, sleep_dir]

# define the save directories
activity_save = activity_dir.parent
sleep_save = sleep_dir.parent
output_list = [activity_save, sleep_save]

# define the subdirectory in the save directory to create and save in
subdir_name = '01_episodes'

# define the keywords for reading in the file
init_kwargs = {
    "subdir_name": subdir_name,
    "func": (prep, "read_file_to_df"),
    "index_col": [0, 1],
    "header": [0]
}

# define the keywords to process the file
process_kwargs = {
    "function": (ep, "episode_find_df"),
    "savecsv": True,
}

# apply process to both activity and sleep
for input, save in zip(input_list, output_list):
    
    # copy the kwargs
    curr_init = init_kwargs
    curr_process = process_kwargs
    
    # modify the init and plot kwargs if necessary
    curr_init["input_directory"] = input
    curr_init["save_directory"] = save
    
    # process all the files
    ep_object = prep.SaveObjectPipeline(**curr_init)
    ep_object.process_file(**curr_process)

