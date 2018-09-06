import sys
import pathlib
sys.path.append(pathlib.Path("..","..",'A2_analysis_library', '.'))
from A2_analysis_library.L3_sleep_analyse import sleep_create_df, process_all_files_in_dir

# define input directory, save suffix, and the directory to put it in.

input_dir = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/01_Projects/P2_Circ_Disruption_paper_chapt2" \
                        "/03_data_files")

save_suffix = "_sleep.csv"

subdir_name = "sleep_csvs"

process_all_files_in_dir(input_directory=input_dir,
                         function_name = sleep_create_df,
                         save_suffix=save_suffix,
                         subdir_name=subdir_name)


