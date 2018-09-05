import pandas as pd
import sys
import pathlib
import os
sys.path.append(pathlib.Path("..","..",'A2_analysis_library', '.'))
from A2_analysis_library.L3_sleep_analyse import sleep_process, process_all_files_in_dir

# create script to process all files in dir

input_dir = pathlib.Path("/Users/angusfisk/Documents/1_PhD_files/1_Projects/P2_Circ_Disruption_paper_chapt2" \
                        "/03_data_files")

save_suffix = "_sleep.csv"

subdir_name = "sleep_csvs"

process_all_files_in_dir(input_directory=input_dir,
                         function_name = sleep_process,
                         save_suffix=save_suffix,
                         subdir_name=subdir_name)


# TODO Turn sleep create file into dealing with apply processing and deal with object cols