import pandas as pd
import sys
import pathlib
import os
sys.path.append(pathlib.Path("..","..",'A2_analysis_library', '.'))
from A2_analysis_library.L3_sleep_analyse import hourly_sleep_save_file_pipeline


# glob all the files we want
# define the output directory
# put through pipeline

# can update to be part of pipeline later

# glob the files

file_dir_string = "/Users/angusfisk/Documents/_deprec_PHD_Data/Experiment_Data/Rough/"

file_directory = pathlib.Path(file_dir_string)

file_list = list(file_directory.glob("**/*sleep_.csv"))

# define output directory

output_dir = pathlib.Path("/Users/angusfisk/Documents/1_PhD_files/1_Projects/P2_Circ_Disruption_paper_chapt2"
                          "/2_analysis_outputs/1_sleep_hourly/")

for read_file, save_file in zip(file_list, output_names):

    hourly_sleep_save_file_pipeline(read_file, output_dir)
