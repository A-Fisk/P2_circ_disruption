# Function to create actogram plots
# What do I need?
# requires split period function
# requires pipeline object
# What are the steps to create?
# Take input data, apply split
# period function, then plot either as an imshow
# or as a multiple subplots

import sys
import pathlib
from A2_analysis_library.L1_preprocessing import SaveObjectPipeline, split_dataframe_by_period
from A2_analysis_library.L5_actogram_create import actogram_plot

# Define which bit of data we are going to play with
# by defining the input directory first
#

input_dir = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/01_projects"
                         "/P2_Circ_Disruption_paper_chapt2/03_data_files")

# create the object
actogram_object = SaveObjectPipeline(input_dir=input_dir)
