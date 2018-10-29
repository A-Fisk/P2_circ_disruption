# script to create actograms and save for all data files
# in input dir

import pathlib
import sys
sys.path.insert(0, "/Users/angusfisk/Documents/01_PhD_files/07_python_package/actigraphy_analysis")
import actigraphy_analysis.preprocessing as prep
import actigraphy_analysis.actogram_plot as act

input_directory = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/"
                                "01_projects/P2_Circ_Disruption_paper_chapt2/"
                                "03_data_files")
save_directory = input_directory / "../02_analysis_outputs"
subdir_name = '01_long_actograms'

fig_object = prep.SaveObjectPipeline(input_directory=input_directory,
                                     save_directory=save_directory)

fig_object.create_plot(function_name="actogram_plot_all_cols",
                       module=act,
                       data_list=fig_object.df_list,
                       subdir_name=subdir_name,
                       savefig=True,
                       save_suffix=".png",
                       figsize=(10,10))

