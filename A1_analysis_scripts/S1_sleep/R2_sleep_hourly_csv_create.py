import sys
import pathlib
sys.path.append(pathlib.Path("..","..",'A2_analysis_library', '.'))
from A2_analysis_library.L1_preprocessing import save_object_pipeline
from A2_analysis_library.L3_sleep_analyse import create_hourly_sum

# define input director
# define subdir name
# define save suffix

input_dir = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/01_projects/P2_Circ_Disruption_paper_chapt2/03_data_files/01_sleep_csvs")

save_suffix = '_hourly.csv'

subdir_name = "01_hourly_sum"

sleep_dfs = save_object_pipeline(input_directory=input_dir,
                                 subdir_name=subdir_name)

sleep_dfs.save_csv_file(create_hourly_sum,
                        save_suffix=save_suffix)
