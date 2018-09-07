import sys
import pathlib
sys.path.append(pathlib.Path("..","..",'A2_analysis_library', '.'))
from A2_analysis_library.L1_preprocessing import save_object_pipeline
from A2_analysis_library.L3_sleep_analyse import simple_plot

# define input dir
# define save suffix
# define subdir

input_dir = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/01_projects/P2_Circ_Disruption_paper_chapt2/03_data_files/01_sleep_csvs/01_hourly_sum")

save_suffix = "plot.png"

subdir_name = "01_plots"

hourly_sleep_files = save_object_pipeline(input_directory=input_dir,
                                          subdir_name=subdir_name)

hourly_sleep_files.create_plot(function_name=simple_plot,
                               save_suffix=save_suffix)

