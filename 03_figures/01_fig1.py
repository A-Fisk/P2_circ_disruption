# start with imports
import pathlib
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.dates as mdates
import sys
sys.path.insert(0, "/Users/angusfisk/Documents/01_PhD_files/"
                    "07_python_package/actiPy")
import actiPy.preprocessing as prep
import actiPy.actogram_plot as aplot
import actiPy.periodogram as per
import actiPy.waveform as wave

# Figure 1
# actograms as subplots

# import the files we are going to read

# define constants
index_cols = [0, 1]
idx = pd.IndexSlice
save_fig = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/"
                        "01_projects/01_thesisdata/02_circdis/"
                        "03_analysis_outputs/03_figures/01_fig1.png")

# get the file names
activity_dir = pathlib.Path('/Users/angusfisk/Documents/01_PhD_files/'
                            '01_projects/01_thesisdata/02_circdis/'
                            '01_data_files/01_activity/00_clean')

sleep_dir = pathlib.Path('/Users/angusfisk/Documents/01_PhD_files/'
                         '01_projects/01_thesisdata/02_circdis/'
                         '01_data_files/02_sleep/00_clean')

activity_filenames = sorted(activity_dir.glob("*.csv"))
activity_files = [x.stem for x in activity_filenames]
sleep_filenames = sorted(sleep_dir.glob("*.csv"))
sleep_files = [x.stem for x in sleep_filenames]

# import into lists
activity_dfs = [prep.read_file_to_df(x, index_col=index_cols)
                for x in activity_filenames]
sleep_dfs = [prep.read_file_to_df(x, index_col=index_cols)
             for x in sleep_filenames]

activity_df = pd.concat(dict(zip(activity_files, activity_dfs)))
sleep_df = pd.concat(dict(zip(activity_files, sleep_dfs)))

##### Plotting - each group

# plotting constants
conditions = activity_df.index.get_level_values(0).unique()
sleep_ylim = [0, 1.5]
hspace = 0.5
anim_dict = {
    conditions[0]: 0,
    conditions[1]: 5,
    conditions[2]: 3,
    conditions[3]: 3
}
actogram_kwargs = {
    "drop_level": True,
    "set_file_title": False,
    "linewidth": 0.1,
}

# initialise figure
fig = plt.figure()

# create the left column for activity
activity_grid = gs.GridSpec(nrows=len(conditions), ncols=1, figure=fig,
                            right=0.4, hspace=hspace)
activity_axes = [plt.subplot(x) for x in activity_grid]

# loop through and add activity actograms to column
for condition, axis in zip(conditions, activity_axes):

    # select right number for each conditions
    data = activity_df.loc[condition]
    animal = anim_dict[condition]
    
    # remove labels from subplot before adding in actogram
    axis.set(yticks=[],
             xticks=[],
             title=condition)
    
    # plot actogram
    aplot._actogram_plot_from_df(data, animal, fig=fig, subplot=axis,
                                 timeaxis=True, **actogram_kwargs)

    # tidy spacing labels and axis


# create the right column for sleep
sleep_grid = gs.GridSpec(nrows=len(conditions), ncols=1, figure=fig,
                         left=0.6, hspace=hspace)
sleep_axes = [plt.subplot(x) for x in sleep_grid]

# same thing for sleep. Loop through conditions, select data, remove axis, plot
for condition, axis_1 in zip(conditions, sleep_axes):
    
    # grab the data
    sleep_data = sleep_df.loc[condition]
    animal = anim_dict[condition]
    
    # remove labels
    axis_1.set(yticks=[],
               xticks=[],
               title=condition)
    
    # plot
    aplot._actogram_plot_from_df(sleep_data, animal, fig=fig, subplot=axis_1,
                                 timeaxis=True,
                                 ylim=sleep_ylim, **actogram_kwargs)
    


fig.set_size_inches(8.27, 11.69)

plt.savefig(save_fig, dpi=1000)

plt.close('all')
