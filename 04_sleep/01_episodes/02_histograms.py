# script to create actograms and save for all data files
# in input dir

import pathlib
import sys
import numpy as np
sys.path.insert(0, "/Users/angusfisk/Documents/01_PhD_files/07_python_package/actigraphy_analysis")
import actigraphy_analysis.preprocessing as prep
import actigraphy_analysis.episodes as ep

input_directory = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/"
                                "01_projects/P2_Circ_Disruption_paper_chapt2/"
                                "03_data_files/02_sleep/01_episodes")
save_directory = input_directory.parents[2] / "02_analysis_outputs/04_sleep"
subdir_name = "01_episode_hist"

episode_object = prep.SaveObjectPipeline(input_directory=input_directory,
                                         save_directory=save_directory)
bins = np.linspace(0,12000,21)
episode_object.create_plot(function_name="ep_hist_conditions_from_df",
                           module=ep,
                           subdir_name=subdir_name,
                           data_list=episode_object.df_list,
                           save_suffix='.png',
                           remove_col=False,
                           savefig=True,
                           figsize=(15,10),
                           bins=bins,
                           xtitle="Episode duration (seconds)",
                           logy=True)
