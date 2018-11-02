# script to create actograms and save for all data files
# in input dir

import pathlib
import sys
sys.path.insert(0, "/Users/angusfisk/Documents/01_PhD_files/07_python_package/actigraphy_analysis")
import actigraphy_analysis.preprocessing as prep
import actigraphy_analysis.episodes as ep

input_directory = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/"
                                "01_projects/P2_Circ_Disruption_paper_chapt2/"
                                "03_data_files")
save_directory = input_directory / "../03_data_files"
subdir_name = "01_episodes"

episode_object = prep.SaveObjectPipeline(input_directory=input_directory,
                                         save_directory=save_directory)
episode_object.process_file(module=ep,
                            function_name="create_episode_df",
                            subdir_name=subdir_name,
                            savecsv=True)

