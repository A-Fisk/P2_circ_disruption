# Figure 4. Episode histogram
# Baseline-disrupted for each animal with separate rows for each condition
# animals separate for now
# Final will be scatter plot of median length per day vs count per day with
# hue as baseline vs disrupted. Sep column for each condition

#### Imports
import pathlib
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import seaborn as sns
sns.set()
import sys
sys.path.insert(0, "/Users/angusfisk/Documents/01_PhD_files/"
                    "07_python_package/actiPy")
import actiPy.preprocessing as prep
import actiPy.analysis as als
import actiPy.periodogram as per
import actiPy.waveform as wave

#### CONSTANTS
INDEX_COLS = [0, 1]
idx = pd.IndexSlice
SAVE_FIG = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/"
                        "01_projects/01_thesisdata/02_circdis/"
                        "03_analysis_outputs/03_figures/04_fig4.png")
LDR_COL = -1
COL_NAMES = ["Data type", "Condition", "Section", "Animal", "Measurement"]
def longform(df,
             col_names: list):
    new_df = df.stack().reset_index().drop("index", axis=1)
    new_df.columns = col_names
    
    return new_df


#### Step 1 Read in the data
#  import the files to read

# get the file names
activity_dir = pathlib.Path('/Users/angusfisk/Documents/01_PhD_files/'
                            '01_projects/01_thesisdata/02_circdis/'
                            '01_data_files/01_activity/01_episodes')

sleep_dir = pathlib.Path('/Users/angusfisk/Documents/01_PhD_files/'
                         '01_projects/01_thesisdata/02_circdis/'
                         '01_data_files/02_sleep/01_episodes')
activity_filenames = sorted(activity_dir.glob("*.csv"))
sleep_filenames = sorted(sleep_dir.glob("*.csv"))

# import into lists
activity_dfs = [prep.read_file_to_df(x, index_col=INDEX_COLS)
                for x in activity_filenames]
activity_names = [x.name for x in activity_dfs]
sleep_dfs = [prep.read_file_to_df(x, index_col=INDEX_COLS)
             for x in sleep_filenames]
sleep_names = [x.name for x in sleep_dfs]

# turn into a dict
activity_dict = dict(zip(activity_names, activity_dfs))
sleep_dict = dict(zip(sleep_names, sleep_dfs))

# turn into multi-index
activity_df = pd.concat(activity_dict)
sleep_df = pd.concat(sleep_dict)

# remove LDR
ldr_label = activity_df.columns[LDR_COL]
activity_df.drop(ldr_label, axis=1, inplace=True)
sleep_df.drop(ldr_label, axis=1, inplace=True)

# put into a large df
data_dict = {
    "activity": activity_df,
    "sleep": sleep_df
}
all_data = pd.concat(data_dict)

#### Step 2 remove post-baseline
data_type = all_data.index.get_level_values(0).unique()
conditions = all_data.index.get_level_values(1).unique()
sections = all_data.index.get_level_values(2).unique()

removed_data = all_data.loc[idx[:, :, :sections[-2]], :]

long_removed = longform(removed_data, col_names=COL_NAMES)

#### Step 3 Calculate length per day and count per day

# select each day - resample using median
removed_data.resample("D", level=3).median()
median_data = removed_data.groupby(level=[0, 1, 2]
                                  ).resample("D", level=3
                                             ).median()
# count each day
above_zero = (removed_data > 0).astype(bool)
count_data = above_zero.groupby(level=[0, 1, 2]
                                ).resample("D", level=3
                                           ).sum()

median_long = longform(median_data, col_names=COL_NAMES)
count_long = longform(count_data, col_names=COL_NAMES)

scatter_data = median_long.copy()
scatter_data.rename({"Measurement": "Median"}, axis=1, inplace=True)
scatter_data["Count"] = count_long.iloc[:, -1]


######## Stats ##########

# start with just saving to send to snp.
save_dir = SAVE_FIG.parent / "00_csvs"
length_csv = save_dir / "04_fig4_length.csv"
long_removed.to_csv(length_csv)
scatter_csv = save_dir / "04_fig4_scatter.csv"
scatter_data.to_csv(scatter_csv)

#### Step 4 Plot all together

# plotting constants
nocols = 2
norows = 5
condition_col = COL_NAMES[1]
section_col = COL_NAMES[2]
measurement_col = COL_NAMES[-1]
animal_col = COL_NAMES[-2]
data_type_col = COL_NAMES[0]
bins = np.geomspace(10, 3600, 10)
label_size = 8

# initialise figure
# fig, ax = plt.subplots(nrows=norows, ncols=nocols)
fig = plt.figure()
upper_grid = gs.GridSpec(nrows=4, ncols=2, figure=fig, top=0.85, bottom=0.5,
                         hspace=0)

histogram_axes = []
for row in range(4):
    col_axes = []
    for col in range(2):
        add_ax = plt.subplot(upper_grid[row, col])
        col_axes.append(add_ax)
    histogram_axes.append(col_axes)
    
histogram_axes_array = np.array(histogram_axes)

data = long_removed
data_types = data[data_type_col].unique()

sections = data[section_col].unique()
animals = data[animal_col].unique()

### Plot histograms

# loop through the data types
for col_no, data_type in enumerate(data_types):
    
    # select the column
    axis_column = histogram_axes_array[:, col_no]

    # select just the data type
    # need to grab data type here as pir_data_sleep
    data_sep_type = data[data[data_type_col] == data_type]
    conditions = data_sep_type[condition_col].unique()

    # loop through all the conditions
    for row_no, condition in enumerate(conditions):
        
        curr_ax = axis_column[row_no]

        # select the data
        curr_data = data[data[condition_col] == condition]
        baseline_data = curr_data[curr_data[section_col]
                                    == sections[0]]
        disrupted_data = curr_data[curr_data[section_col]
                                        == sections[1]]

        ax1 = curr_ax
        ax1.hist([baseline_data[measurement_col],
                  disrupted_data[measurement_col]],
                  # alpha=0.5,
                  color=["k", 'b'],
                  bins=bins, density=True)
        # ax1.hist(disrupted_data[measurement_col], alpha=0.3, color="b",
        #          bins=bins, density=True)
        
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        ax1.tick_params(axis='both', which='major', labelsize=label_size)
        
        # remove the axis label
        if condition != conditions[-1]:
            ax1.set_xticklabels(ax1.get_xticklabels(), visible=False)
        else:
            ax1.set_xlabel("Bout duration, log secs", size=label_size)
            
        if condition == conditions[0]:
            ax1.set_title("Log histogram of bout duration", size=label_size)
            
        ax1.text(0.8, 0.9, condition, transform=ax1.transAxes,
                 size=label_size)
        
fig.text(0.07, 0.69, "Normalised density", rotation=90, size=label_size)
fig.text(0.5, 0.69, "Normalised density", rotation=90, size=label_size)

### Plot scatterplot

# create plots
lower_grid = gs.GridSpec(nrows=1, ncols=2, top=0.40, bottom=0.1,
                         figure=fig)
activity_ax = plt.subplot(lower_grid[:, 0])
sleep_ax = plt.subplot(lower_grid[:, 1])

scatter_array = np.array([activity_ax, sleep_ax])

median_col = scatter_data.columns[-2]
count_col = scatter_data.columns[-1]

# loop through the data types
for sep_col, data_type in enumerate(data_types):
    
    curr_scatter_axis = scatter_array[sep_col]
    
    # select the data
    data_sep_type = scatter_data[scatter_data[data_type_col] == data_type]
    ymin = 0
    ymax = data_sep_type[median_col].max() * 0.8
    xmin = 0
    xmax = data_sep_type[count_col].max()
    
    # create different conditions plots
    inner_grid_scatter = gs.GridSpecFromSubplotSpec(nrows=1,
                                                    ncols=4,
                                                    subplot_spec=curr_scatter_axis,
                                                    wspace=0,
                                                    hspace=0)

    # grab the conditions to avoid sleep/activity missing
    conditions = data_sep_type[condition_col].unique()
    
    # loop through the conditions
    for condition, grid in zip(conditions, inner_grid_scatter):
        
        # select the data
        condition_scatter = data_sep_type[
            data_sep_type[condition_col] == condition
        ]
        baseline_data = condition_scatter[condition_scatter[section_col] ==
                                          sections[0]]
        disrupted_data = condition_scatter[condition_scatter[section_col] ==
                                           sections[1]]
        
        # create the new axis
        ax2 = plt.Subplot(fig, grid)
        fig.add_subplot(ax2)
        
        # plot the scatterplot on the axis
        # sns.scatterplot(x=count_col, y=median_col, hue=section_col,
        #                 data=condition_scatter, ax=ax2, alpha=0.5,
        #                 legend=False)
        
        # try with a kdeplot instead
        sns.kdeplot(baseline_data[count_col], baseline_data[median_col],
                    shade=False, shade_lowest=False, cmap="Greys",
                    ax=ax2, alpha=0.8)
        sns.kdeplot(disrupted_data[count_col], disrupted_data[median_col],
                    shade=False, shade_lowest=False, cmap="Blues",
                    ax=ax2, alpha=0.8)
        
        # set x and y axis
        ax2.set(ylim=[ymin, ymax],
                xlim=[xmin, xmax],
                facecolor='w')
        ax2.tick_params(axis='both', which='major', labelsize=label_size)

        # remove the labels
        if condition != conditions[0]:
            ax2.set_yticks([])
            ax2.set_yticklabels(ax2.get_yticklabels(), visible=False)
        ax2.set_ylabel("")
        ax2.set_xlabel("")
        
        # set the title to be each condition
        ax2.set_title(condition, size=label_size, rotation=45, va="top")
    
    # remove curr axis labels
    curr_scatter_axis.set_yticks([])
    curr_scatter_axis.set_xticks([])

    # set axis label
    curr_scatter_axis.set_xlabel("Number of episodes per day", size=label_size)
    curr_scatter_axis.set_ylabel("Median length of episodes per day, secs",
                                 size=label_size)
    curr_scatter_axis.yaxis.set_label_coords(-0.15, 0.5)
    curr_scatter_axis.xaxis.set_label_coords(0.5, -0.15)

# add in activity and sleep columns
fig.text(0.25, 0.9, "Activity", size=label_size)
fig.text(0.75, 0.9, "Sleep", size=label_size)

# create the legend
legend_lines = [Line2D([0], [0], color='w', alpha=0.8,
                        marker='o', markersize=10, label="Baseline",
                       markerfacecolor='0.5'),
                Line2D([0], [0], color='w', alpha=0.4, marker='o',
                        markersize=10, label="Disrupted",
                       markerfacecolor='b')]
fig.legend(handles=legend_lines, loc=(0.87, 0.9), fontsize=label_size,
           frameon=False)
fig.legend(handles=legend_lines, loc=(0.87, 0.42), fontsize=label_size,
           frameon=False)

fig.suptitle("Bout duration under different disrupting light cycles")
fig.set_size_inches(8.27, 11.69)

plt.savefig(SAVE_FIG, dpi=600)

plt.close('all')