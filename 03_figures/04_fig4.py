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

# initialise figure
fig, ax = plt.subplots(nrows=norows, ncols=nocols)

data = long_removed
data_types = data[data_type_col].unique()

sections = data[section_col].unique()
animals = data[animal_col].unique()

### Plot histograms

# loop through the data types
for col_no, data_type in enumerate(data_types):
    
    # select the column
    axis_column = ax[:, col_no]

    # select just the data type
    # need to grab data type here as pir_data_sleep
    data_sep_type = data[data[data_type_col] == data_type]
    conditions = data_sep_type[condition_col].unique()

    # loop through all the conditions
    for row_no, condition in enumerate(conditions):
        
        curr_ax = axis_column[row_no]

        # select the data
        curr_data = data[data[condition_col] == condition]

        # add in subplots for each PIR
        inner_grid = gs.GridSpecFromSubplotSpec(nrows=1,
                                                ncols=6,
                                                subplot_spec=curr_ax,
                                                wspace=0,
                                                hspace=0)

        for animal, grid in zip(animals, inner_grid):
            
            # select the data for just that PIR
            animal_data =  curr_data[curr_data[animal_col] == animal]
            baseline_data = animal_data[animal_data[section_col]
                                        == sections[0]]
            disrupted_data = animal_data[animal_data[section_col]
                                            == sections[1]]

            # create the new subplot
            ax1 =plt.Subplot(fig, grid)
            fig.add_subplot(ax1)
            
            # plot the animal data on the new subplot
            # baseline and disrupted on the same axis with lower alpha
            ax1.hist(baseline_data[measurement_col], alpha=0.5, color="k",
                     bins=bins, density=True)
            ax1.hist(disrupted_data[measurement_col], alpha=0.5, color="b",
                     bins=bins, density=True)
            
            ax1.set_yscale('log')
            ax1.set_xscale('log')

### Plot scatterplot
scatter_row = ax[-1, :]

curr_scatter_axis = scatter_row[0]

median_col = scatter_data.columns[-2]
count_col = scatter_data.columns[-1]

# loop through the data types
for sep_col, data_type in enumerate(data_types):
    
    curr_scatter_axis = scatter_row[sep_col]
    
    # select the data
    data_sep_type = scatter_data[scatter_data[data_type_col] == data_type]
    ymin = data_sep_type[median_col].min()
    ymax = data_sep_type[median_col].max() * (2/3)
    xmin = data_sep_type[count_col].min()
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
                    shade=True, shade_lowest=False, cmap="Greys",
                    ax=ax2, alpha = 0.5)
        sns.kdeplot(disrupted_data[count_col], disrupted_data[median_col],
                    shade=True, shade_lowest=False, cmap="Blues",
                    ax=ax2, alpha = 0.5)
        
        # set x and y axis
        ax2.set(ylim=[ymin, ymax],
                xlim=[xmin, xmax])




plt.close('all')


t20_test_data = long_removed[long_removed[condition_col] == "t20_pir_data"]
t20_median = t20_test_data[measurement_col]
t20_median.plot()

t20_upmedian = median_data.loc[idx[:, "ll_pir_data"], :]
#
#
# fig1, ax1 = plt.subplots()
#
# condition = conditions[2]
# # select the data
# condition_scatter = data_sep_type[
#     data_sep_type[condition_col] == condition
# ]
# baseline_data = condition_scatter[condition_scatter[section_col] ==
#                                   sections[0]]
# disrupted_data = condition_scatter[condition_scatter[section_col] ==
#                                    sections[1]]
#
# sns.kdeplot(baseline_data[count_col], baseline_data[median_col],
#             shade=True, shade_lowest=False, cmap="Greys",
#             ax=ax1, alpha = 0.5)
# sns.kdeplot(disrupted_data[count_col], disrupted_data[median_col],
#             shade=True, shade_lowest=False, cmap="Blues",
#             ax=ax1, alpha = 0.5)
