# Figure 1 - representative actograms of each condition

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

# define constants
index_cols = [0, 1]
idx = pd.IndexSlice
save_fig = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/"
                        "01_projects/01_thesisdata/02_circdis/"
                        "03_analysis_outputs/03_figures/01_fig1.png")

# import the files we are going to read ########################################
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

##### Plotting - ###############################################################

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
ac_panels = {
    conditions[0]: "A",
    conditions[1]: "B",
    conditions[2]: "C",
    conditions[3]: "D"
}
sl_panels = {
    conditions[0]: "E",
    conditions[1]: "F",
    conditions[2]: "G",
    conditions[3]: "H"

}
actogram_kwargs = {
    "drop_level": True,
    "set_file_title": False,
    "linewidth": 0.1,
    "day_label_size": 6,
    "ylabelpos": (0.1, 0.5),
    "xlabelpos": (0.5, 0.1),
    "title": "Double plotted actograms of activity and sleep"
}
timelabelsize = 5

# initialise figure
fig = plt.figure()
plt.rcParams['patch.force_edgecolor'] = False

# Plot activity actograms on left column
activity_grid = gs.GridSpec(nrows=len(conditions),
                            ncols=1,
                            figure=fig,
                            right=0.45,
                            left=0.15,
                            hspace=hspace)
activity_axes = [plt.subplot(x) for x in activity_grid]

# Plot a representative animal from each protocol on each axis
for condition, axis in zip(conditions, activity_axes):
    data = activity_df.loc[condition]
    animal = anim_dict[condition]
    
    # Plot on a clean axis
    axis.set(yticks=[],
             xticks=[])
    axis.text(-0.2, # panel text
              1.1,
              ac_panels[condition],
              transform=axis.transAxes)
    axis.text(-0.3, # condition text
              0.5,
              condition,
              transform=axis.transAxes,
              rotation=90)
    afig, acax = aplot._actogram_plot_from_df(data,
                                              animal,
                                              fig=fig,
                                              subplot=axis,
                                              timeaxis=True,
                                              **actogram_kwargs)
    
    acax.tick_params(axis="x", which='major', labelsize=timelabelsize)
    
    
# Plot sleep actograms on the right column
sleep_grid = gs.GridSpec(nrows=len(conditions),
                         ncols=1,
                         figure=fig,
                         left=0.55,
                         right=0.85,
                         hspace=hspace)
sleep_axes = [plt.subplot(x) for x in sleep_grid]

# Plot same animal sleep actogram on right column
for condition, axis_1 in zip(conditions, sleep_axes):
    sleep_data = sleep_df.loc[condition]
    animal = anim_dict[condition]
    
    # plot on clean axis
    axis_1.set(yticks=[],
               xticks=[])
    axis_1.text(-0.2, # panel text
                1.1,
                sl_panels[condition],
                transform=axis_1.transAxes)
    sfig, sax  = aplot._actogram_plot_from_df(sleep_data,
                                              animal,
                                              fig=fig,
                                              subplot=axis_1,
                                              timeaxis=True,
                                              ylim=sleep_ylim,
                                              **actogram_kwargs)
    
    # tidy timelabels
    sax.tick_params(axis="x", which='major', labelsize=timelabelsize)
    
    
# Add in further labels
type_label_level = 1.1
type_label_xval = 0.35
fig.text(
    type_label_xval,
    type_label_level,
    "Activity",
    transform=activity_axes[0].transAxes
)
fig.text(
    type_label_xval,
    type_label_level,
    "Sleep",
    transform=sleep_axes[0].transAxes
)

# save final version
fig.set_size_inches(8.27, 11.69)
plt.savefig(save_fig, dpi=1000)

plt.close('all')
