import sys
import pathlib
sys.path.append(pathlib.Path("..","..",'A2_analysis_library', '.'))
from A2_analysis_library.L3_sleep_analyse import process_all_files_in_dir, create_hourly_sum

# define input director
# define subdir name
# define save suffix

input_dir = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/01_projects/P2_Circ_Disruption_paper_chapt2/03_data_files/01_sleep_csvs")

save_suffix = '_hourly.csv'

subdir_name = "01_hourly_sum"

process_all_files_in_dir(input_directory=input_dir,
                         function_name=create_hourly_sum,
                         save_suffix=save_suffix,
                         subdir_name=subdir_name)
