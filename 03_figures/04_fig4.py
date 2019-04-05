# Figure 4. Do these conditions cause fragmentation?

# Imports
import sys
import pathlib
import pingouin as pg
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

sns.set()

sys.path.insert(0, "/Users/angusfisk/Documents/01_PhD_files/"
                   "07_python_package/actiPy")
import actiPy.preprocessing as prep
import actiPy.periodogram as per

# CONSTANTS
INDEX_COLS = [0, 1]
idx = pd.IndexSlice
SAVE_FIG = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/"
                        "01_projects/01_thesisdata/02_circdis/"
                        "03_analysis_outputs/03_figures/04_fig4.png")
LDR_COL = -1
COL_NAMES = ["Data_type", "Protocol", "Time", "Animal", "Measurement"]
save_csv_dir = SAVE_FIG.parent / "00_csvs/04_fig4"

def longform(df,
             col_names: list,
             drop_index: bool=True):
    if drop_index:
        new_df = df.stack().reset_index().drop("index", axis=1)
    else:
        new_df = df.stack().reset_index()
    new_df.columns = col_names

    return new_df

def norm_base_mean(protocol_df, baseline_str: str = "Baseline"):
    base_values = protocol_df.loc[idx[:, :, baseline_str], :]
    normalise_value = base_values.mean().mean()
    normalised_df = (protocol_df / normalise_value) * 100
    return normalised_df


# Step 1 Read in the data
activity_dir = pathlib.Path('/Users/angusfisk/Documents/01_PhD_files/'
                            '01_projects/01_thesisdata/02_circdis/'
                            '01_data_files/01_activity/01_episodes')

sleep_dir = pathlib.Path('/Users/angusfisk/Documents/01_PhD_files/'
                         '01_projects/01_thesisdata/02_circdis/'
                         '01_data_files/02_sleep/01_episodes')
activity_filenames = sorted(activity_dir.glob("*.csv"))
sleep_filenames = sorted(sleep_dir.glob("*.csv"))
activity_dfs = [prep.read_file_to_df(x, index_col=INDEX_COLS)
                for x in activity_filenames]
activity_names = [x.name for x in activity_dfs]
sleep_dfs = [prep.read_file_to_df(x, index_col=INDEX_COLS)
             for x in sleep_filenames]
sleep_names = [x.name for x in sleep_dfs]
activity_dict = dict(zip(activity_names, activity_dfs))
sleep_dict = dict(zip(sleep_names, sleep_dfs))
activity_df = pd.concat(activity_dict)
sleep_df = pd.concat(sleep_dict)

# remove LDR
ldr_label = activity_df.columns[LDR_COL]
activity_df.drop(ldr_label, axis=1, inplace=True)
sleep_df.drop(ldr_label, axis=1, inplace=True)

# Combine into a single df
data_dict = {
    "activity": activity_df,
    "sleep": sleep_df
}
all_data = pd.concat(data_dict)

# get the periods
period_dir = pathlib.Path(
    "/Users/angusfisk/Documents/01_PhD_files/01_projects/01_thesisdata/"
    "02_circdis/03_analysis_outputs/03_figures/00_csvs/03_fig3/03_periods"
)
period_files = sorted(period_dir.glob("*.csv"))
activity_periods = pd.to_timedelta(
    pd.read_csv(
        period_files[0],
        index_col=[0, 1]
    ).stack()
).unstack(
    level=2
)
sleep_periods = pd.to_timedelta(
    pd.read_csv(
        period_files[1],
        index_col=[0, 1]
    ).stack()
).unstack(
    level=2
)


# Step 2 remove post-baseline
data_type = all_data.index.get_level_values(0).unique()
conditions = all_data.index.get_level_values(1).unique()
sections = all_data.index.get_level_values(2).unique()
removed_data = all_data.loc[idx[:, :, :sections[-2]], :]

# relabel animal names
removed_rename = removed_data.groupby(
    level=[0, 1, 2]
).apply(
    prep.label_anim_cols,
    level_index=1
)

long_removed = longform(removed_rename, col_names=COL_NAMES)

# Split by period
activity_split = prep.split_list_with_periods(
    name_df=activity_periods,
    df_list=activity_dfs
)
sleep_split = prep.split_list_with_periods(
    name_df=sleep_periods,
    df_list=sleep_dfs
)
split_dict = {
    "activity": activity_split,
    "sleep": sleep_split
}
split_df = pd.concat(split_dict)

# Step 3 Calculate mean length per day and count per day
mean_data = split_df.groupby(
    level=[0, 1, 2, 3]
).mean( # mean of each day
).mean( # mean of all days for each animal
    axis=1
).unstack(
    level=3
)

median_data = all_data.groupby( # median length per day
    level=[0, 1, 2]
).resample(
    "D",
    level=3
).mean()
median_anim_mean = median_data.groupby( # mean median length for each animal
    level=[0, 1, 2]
).mean()
median_norm = median_anim_mean.groupby( # normalise animal means to baseline
    level=[0, 1]
).apply(
    norm_base_mean
)
med_relabel = median_norm.groupby( # relabel for stats
    level=[0, 1, 2]
).apply(
    prep.label_anim_cols,
    level_index=1
)

# count each day
above_zero = (all_data > 0).astype(bool) # Turn each value to a bool
count_data = above_zero.groupby( # count where there is an episode
    level=[0, 1, 2]
).resample(
    "D",
    level=3
).sum()
count_anim_mean = count_data.groupby( # mean of number per day for each animal
    level=[0, 1, 2]
).mean()
count_norm = count_anim_mean.groupby( # normalise to baseline mean
    level=[0, 1]
).apply(
    norm_base_mean
)
count_relabel = count_norm.groupby( # relabel for stats
    level=[0, 1, 2]
).apply(
    prep.label_anim_cols,
    level_index=1
)

# tidy and put into a single df
med_cols = COL_NAMES.copy()
med_cols[-1] = "Mean"
median_long = longform(med_relabel, col_names=med_cols, drop_index=False)
count_cols = COL_NAMES.copy()
count_cols[-1] = "Count"
count_long = longform(count_relabel, col_names=count_cols, drop_index=False)
scatter_data = median_long.copy()
scatter_data["Count"] = count_long.iloc[:, -1]

######## Stats ##########

# start with just saving to send to snp.
save_dir = SAVE_FIG.parent / "00_csvs"
length_csv = save_dir / "04_fig4_length.csv"
long_removed.to_csv(length_csv)
scatter_csv = save_dir / "04_fig4_scatter.csv"
scatter_data.to_csv(scatter_csv)


# activity episodes?
ac_mask = scatter_data.iloc[:, 0] == "activity"
activity_data = scatter_data[ac_mask]
sl_mask = scatter_data.iloc[:, 0] == 'sleep'
sleep_data = scatter_data[sl_mask]

median = "Mean"
count = "Count"
within = COL_NAMES[2]
between = COL_NAMES[1]
subjects = COL_NAMES[3]

act_test_dir = save_csv_dir / "01_activity"
sl_test_dir = save_csv_dir / "02_sleep"

# 1. First main question - is the duration of episodes significantly affected
# by time and by protocol
# ANOVA on median length per day

med_csv = "01_median.csv"
ac_med_file = act_test_dir / med_csv
ac_med_rm = pg.mixed_anova(dv=median,
                           within=within,
                           between=between,
                           subject=subjects,
                           data=activity_data)
pg.print_table(ac_med_rm)
ac_med_rm.to_csv(ac_med_file)

sl_med_file = sl_test_dir / med_csv
sl_med_rm = pg.mixed_anova(dv=median,
                           within=within,
                           between=between,
                           subject=subjects,
                           data=sleep_data)
pg.print_table(sl_med_rm)
sl_med_rm.to_csv(sl_med_file)

# post hoc test for sleep med
protocols = sleep_data[between].unique()
ph_dict = {}
for protocol in protocols:
    protocol_mask = sleep_data[between] == protocol
    protocol_df = sleep_data[protocol_mask]

    ph = pg.pairwise_tukey(dv=median,
                           between=within,
                           data=protocol_df)
    print(protocol)
    pg.print_table(ph)

    ph_dict[protocol] = ph
ph_df = pd.concat(ph_dict)
ph_name = sl_test_dir / "03_med_ph.csv"
ph_df.to_csv(ph_name)

# 2. Are the number of episodes different?

count_csv = "02_count.csv"
ac_ct_file = act_test_dir / count_csv
ac_count_rm = pg.mixed_anova(dv=count,
                             within=within,
                             between=between,
                             subject=subjects,
                             data=activity_data)
pg.print_table(ac_count_rm)
ac_count_rm.to_csv(ac_ct_file)

sl_ct_file = sl_test_dir / count_csv
sl_count_rm = pg.mixed_anova(dv=count,
                             within=within,
                             between=between,
                             subject=subjects,
                             data=sleep_data)
pg.print_table(sl_count_rm)
sl_count_rm.to_csv(sl_ct_file)


######### Step 4 Plot all together#############################################

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
data = long_removed
data_types = data[data_type_col].unique()
sections = data[section_col].unique()
animals = data[animal_col].unique()
marker_types = [count_cols[-1], med_cols[-1]]
med_count_data = {
    marker_types[0]: count_long,
    marker_types[1]: median_long
}

# point/swarm plot
dodge = 0.5
capsize = 0.2
errwidth = 1
sem = 68
marker_size = 3

# initialise figure
fig = plt.figure()

count_grid = gs.GridSpec(
    nrows=1,
    ncols=2,
    figure=fig,
    top=0.9,
    bottom=0.7
)
count_axes = [plt.subplot(x) for x in count_grid]
median_grid = gs.GridSpec(
    nrows=1,
    ncols=2,
    figure=fig,
    top=0.7,
    bottom=0.5
)
median_axes = [plt.subplot(x) for x in median_grid]

# plot point plot and swarm plot of each data on the the axis
med_count_axes = [count_axes, median_axes]
for marker, type_axis in zip(marker_types, med_count_axes):
    data = med_count_data[marker]
    for curr_ax, data_type in zip(type_axis, data_types):
        curr_data = data.query("%s =='%s'"%(data_type_col, data_type))
        
        sns.pointplot(
            x=condition_col,
            y=marker,
            hue=section_col,
            data=curr_data,
            ax=curr_ax,
            join=False,
            capsize=capsize,
            errwidth=errwidth,
            dodge=dodge,
            ci=sem,
        )
        sns.swarmplot(
            x=condition_col,
            y=marker,
            hue=section_col,
            data=curr_data,
            ax=curr_ax,
            dodge=dodge,
            size=marker_size
        )
        legend = curr_ax.legend()
        legend.remove()



upper_grid = gs.GridSpec(
    nrows=4,
    ncols=2,
    figure=fig,
    top=0.85,
    bottom=0.5,
    hspace=0
)

histogram_axes = []
for row in range(4):
    col_axes = []
    for col in range(2):
        add_ax = plt.subplot(upper_grid[row, col])
        col_axes.append(add_ax)
    histogram_axes.append(col_axes)

histogram_axes_array = np.array(histogram_axes)


# Plot histograms

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

# Plot scatterplot

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
    inner_grid_scatter = gs.GridSpecFromSubplotSpec(
        nrows=1, ncols=4, subplot_spec=curr_scatter_axis, wspace=0, hspace=0)

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
        sns.scatterplot(baseline_data[count_col],
                        baseline_data[median_col],
                        # shade=True,
                        # shade_lowest=True,
                        # cmap="Greys",
                        ax=ax2,
                        alpha=1)
        sns.scatterplot(disrupted_data[count_col],
                        disrupted_data[median_col],
                        # shade=False,
                        # shade_lowest=False,
                        # cmap="Blues",
                        ax=ax2,
                        alpha=0.5)

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

        print(condition, baseline_data.describe())

    # remove curr axis labels
    curr_scatter_axis.set_yticks([])
    curr_scatter_axis.set_xticks([])

    # set axis label
    curr_scatter_axis.set_xlabel("Number of episodes per day", size=label_size)
    curr_scatter_axis.set_ylabel("Mean length of episodes per day, secs",
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
