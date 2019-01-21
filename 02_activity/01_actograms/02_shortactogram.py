# Script for creating short actograms

import pathlib
import sys
sys.path.insert(0, "/Users/angusfisk/Documents/01_PhD_files/"
                    "07_python_package/actiPy")
import actiPy.preprocessing as prep
import actiPy.actogram_plot as act

# define the input directoryies
activity_dir = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/"
                              "01_projects/01_thesisdata/02_circdis/"
                              "01_data_files/01_activity/00_clean")
sleep_dir = activity_dir.parents[1] / "02_sleep/00_clean"
input_list = [activity_dir, sleep_dir]

# define the save directories
activity_save = activity_dir.parents[2] / "03_analysis_outputs/01_activity"
sleep_save = activity_save.parent / "02_sleep"
output_list = [activity_save, sleep_save]

# define the subdirectory in the save directory to create and save in
subdir_name = '02_short_actograms'

# define the keywords for reading in the file
init_kwargs = {
    "subdir_name": subdir_name,
    "func": (prep, "read_file_to_df"),
    "index_col": [0, 1],
    "header": [0]
}

# define the keywords to plot the file
plot_kwargs = {
    "function": (act, "actogram_plot_all_cols"),
    "LDR": -1,
    "set_file_title": True,
    "showfig": False,
    "period": "24H",
    "savefig": True,
    "figsize": (10, 10)
}

# apply plotting to both activity and sleep
for input, save in zip(input_list, output_list):
    
    curr_init = init_kwargs
    curr_plot = plot_kwargs
    # modify the init and plot kwargs if necessary
    curr_init["input_directory"] = input
    curr_init["save_directory"] = save
    
    if "sleep" in input.parent.stem:
        print(input.stem)
        curr_plot["ylim"] = [0, 1.5]
        
    short_act_object = prep.SaveObjectPipeline(**curr_init)
    short_act_object.create_plot(**curr_plot)
