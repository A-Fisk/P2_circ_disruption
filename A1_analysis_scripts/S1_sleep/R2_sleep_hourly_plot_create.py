import pandas as pd
import sys
import pathlib
import os
sys.path.append(pathlib.Path("..","..",'A2_analysis_library', '.'))
from A2_analysis_library.L3_sleep_analyse import simple_plot

file_directory = pathlib.Path("/Users/angusfisk/Documents/1_PhD_files/1_Projects/P2_Circ_Disruption_paper_chapt2"
                          "/2_analysis_outputs/1_sleep_hourly/")

file_list = list(file_directory.glob("**/*sleep_.csv"))

# define output directory

output_dir = pathlib.Path("/Users/angusfisk/Documents/1_PhD_files/1_Projects/P2_Circ_Disruption_paper_chapt2"
                          "/2_analysis_outputs/1_sleep_hourly/1_plots/")

data_list = []

for file in file_list:

    df = pd.read_csv(file, index_col=0, parse_dates=True)

    data_list.append(df)

file_list_names = []

for file in file_list:

    temp_file_name = file.parts[-1][:-4] + ".png"

    file_list_names.append(temp_file_name)

for df, file_name in zip(data_list, file_list_names):

    simple_plot(df, destination_dir = output_dir, file_name = file_name, savefig=True, showfig=False)

