import sys
print (sys.path)
import pathlib
from A2_analysis_library.L1_preprocessing import SaveObjectPipeline
from A2_analysis_library.L3_sleep_analyse import create_hourly_sum, simple_plot

# define input director
# define subdir name
# define save suffix

input_dir = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/01_projects/P2_Circ_Disruption_paper_chapt2/03_data_files/01_sleep_csvs")

save_suffix_csv = '_hourly.csv'

subdir_name_csv = "01_hourly_sum"

save_suffix_plot = "_hourly_plot.png"

subdir_name_plot = subdir_name_csv + "/01_plots"

sleep_dfs = SaveObjectPipeline(input_directory=input_dir)

sleep_dfs.save_csv_file(function_name=create_hourly_sum,
                        subdir_name=subdir_name_csv,
                        save_suffix=save_suffix_csv)

sleep_dfs.create_plot(function_name=simple_plot
                      subdir_name=subdir_name_plot,
                      save_suffix=save_suffix_plot)


