import pandas as pd
import sys
import pathlib
sys.path.append(pathlib.Path("..","..",'A2_analysis_library', '.'))
from A2_analysis_library.L3_sleep_analyse import simple_plot, process_all_files_in_dir

# define input dir
# define save suffix
# define subdir

input_dir = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/01_projects/P2_Circ_Disruption_paper_chapt2/03_data_files/01_sleep_csvs/01_hourly_sum")

save_suffix = "plot.png"

subdir_name = "01_plots"

process_all_files_in_dir(input_directory=input_dir,
                         function_name=simple_plot())


# TODO create function to create save file name/path given name and suffix
# TODO create function to save csv file
# TODO create function to save plots
# TODO update process all files function to be able to call save plots and save csvs
# TODO update process all files function to accept **kwargs
