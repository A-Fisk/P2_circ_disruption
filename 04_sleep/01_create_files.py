# script to create actograms and save for all data files
# in input dir

import pathlib
import sys
sys.path.insert(0, "/Users/angusfisk/Documents/01_PhD_files/07_python_package/actigraphy_analysis")
import actigraphy_analysis.preprocessing as prep
import actigraphy_analysis.episodes as ep
import actigraphy_analysis.sleep_process as sleep

input_directory = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/"
                                "01_projects/P2_Circ_Disruption_paper_chapt2/"
                                "03_data_files")
save_directory = input_directory
subdir_name = "02_sleep"

sleep_object = prep.SaveObjectPipeline(input_directory=input_directory,
                                       save_directory=save_directory)
sleep_object.process_file(module=sleep,
                          function_name="create_scored_df",
                          subdir_name=subdir_name,
                          savecsv=True)
sleep_object.process_file(module=sleep,
                          function_name="alter_file_name",
                          subdir_name=subdir_name,
                          savecsv=False,
                          suffix="_sleep_data",
                          object_list=sleep_object.processed_file_list)
