import sys
import pathlib
sys.path.append(pathlib.Path("..","..",'A2_analysis_library', '.'))
from A2_analysis_library.L1_preprocessing import save_object_pipeline
from A2_analysis_library.L3_sleep_analyse import sleep_create_df

# define input directory, save suffix, and the directory to put it in.

input_dir = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/01_Projects/P2_Circ_Disruption_paper_chapt2" \
                        "/03_data_files")

save_suffix = "_sleep.csv"

subdir_name = "01_sleep_csvs"

raw_data_files = save_object_pipeline(input_directory=input_dir,
                                      subdir_name=subdir_name)

raw_data_files.save_csv_file(sleep_create_df,
                             save_suffix=save_suffix)

