import pathlib
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
sys.path.insert(0, "/Users/angusfisk/Documents/01_PhD_files/"
                   "07_python_package/actigraphy_analysis")
import actigraphy_analysis.preprocessing as prep
import actigraphy_analysis.actogram_plot as act
import actigraphy_analysis.episodes as ep

# read the first file in from the input directory
input_dir = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/"
                         "01_Projects/P2_Circ_Disruption_paper_chapt2"
                         "/03_data_files")
file_name = list(input_dir.glob("*.csv"))[0]
data = prep.read_file_to_df(file_name)
save_name = pathlib.Path('./filter_length_<10s.png')

episodes = ep.create_episode_df(data,
                                min_length="1S")

data_col = episodes.iloc[:,0]
lengths = []
filter_range = list(range(0,30))
for x in filter_range:
    temp_len = len(data_col[data_col>x])
    lengths.append(temp_len)

len_series = pd.Series(data=lengths,
                       index=filter_range)
    
fig, ax = plt.subplots()
ax.plot(len_series)
ax.set_xlabel("Filter duration in seconds")
ax.set_ylabel("Number of activity episodes")
ax.set_title(file_name.stem)
xticks = np.linspace(filter_range[0], filter_range[-1], len(filter_range))
ax.set(xticks=xticks,
       xticklabels=filter_range)
fig.suptitle("Effect of filter length on activity episdoes")
# plt.savefig(save_name)




