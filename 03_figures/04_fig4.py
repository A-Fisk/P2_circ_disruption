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
import actiPy.plots as aplot

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
periods_dict = {
    "activity": activity_periods,
    "sleep": sleep_periods
}
periods_df = pd.concat(periods_dict).stack()


# Step 2 calculate histogram of episode durations
# Getting mean of
bins = [(x*60) for x in [0, 1, 10, 60, 60000000]]
hist_bin_cols = ["0-1", "1-10", "10-60", ">60"]
def hist_vals(test_data, bins, hist_cols, **kwargs):
    hist = np.histogram(test_data, bins, **kwargs)
    hist_vals = pd.DataFrame(hist[0], index=hist_cols)
    return hist_vals

# correct for period
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
hist_input = split_df.stack().reorder_levels([0, 1, 2, 3, 5, 4]).sort_index()
hist_data = hist_input.groupby(
    level=[0, 1, 2, 3, 4]
).apply(
    hist_vals,
    bins=bins,
    hist_cols = hist_bin_cols,
).unstack(
    level=4
)
hist_anim_mean = hist_data.mean(axis=1).unstack(level=-2)
hist_label = hist_anim_mean.groupby(
    level=[0, 1, 2, 3]
).apply(
    prep.label_anim_cols,
    level_index=1
)
hist_cols = COL_NAMES.copy()
hist_cols[-1] = "Duration"
hist_cols.append("Number of Episodes")
long_hist = hist_label.stack(
).reorder_levels(
    [0, 1, 2, 4, 3]
).reset_index()
long_hist.columns = hist_cols

# Step 3 Calculate mean length per day and count per day

# going to resample by each animals period
# will need manual groupby fx to do so with each animal havign diff period
def manual_mean_groupby(curr_data,
                        mean: bool=True,
                        sum: bool=False):
    types = curr_data.index.get_level_values(0).unique()
    protocols = curr_data.index.get_level_values(1).unique()
    times = curr_data.index.get_level_values(2).unique()
    animals = curr_data.index.get_level_values(3).unique()
    resampled_data_dict = {}
    for type in types:
        type_df = curr_data.loc[type]
        type_periods = periods_df.loc[type]
        type_dict = {}
        for protocol in protocols:
            protocol_df = type_df.loc[protocol]
            protocol_periods = type_periods.loc[protocol]
            protocol_dict = {}
            for time in times:
                time_df = protocol_df.loc[time]
                time_periods = protocol_periods.loc[time]
                time_dict = {}
                for animal in animals:
                    animal_df = time_df.loc[animal]
                    animal_period = time_periods.loc[animal]
                    if mean:
                        anim_day_df = animal_df.resample(animal_period).mean()
                    if sum:
                        anim_day_df = animal_df.resample(animal_period).sum()
                    time_dict[animal] = anim_day_df
                time_df_resampled = pd.concat(time_dict)
                protocol_dict[time] = time_df_resampled
            protocol_df_resampled = pd.concat(protocol_dict)
            type_dict[protocol] = protocol_df_resampled
        type_df_resampled = pd.concat(type_dict)
        resampled_data_dict[type] = type_df_resampled
    resampled_data = pd.concat(resampled_data_dict)

    return resampled_data

# get mean data in right format
stacked_all = all_data.stack().reorder_levels([0, 1, 2, 4, 3])
mean_data = manual_mean_groupby(
    stacked_all,
    mean=True,
    sum=False
).unstack(
    level=3
)
mean_anim_mean = mean_data.groupby(
    level=[0, 1, 2]
).mean()
median_norm = mean_anim_mean.groupby( # normalise animal means to baseline
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

# get sum data in right format
above_zero = (all_data > 0).astype(
    bool
).stack().reorder_levels([0, 1, 2, 4, 3])
count_data = manual_mean_groupby(
    above_zero,
    mean=False,
    sum=True
).unstack(
    level=3
)
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
long_hist.to_csv(length_csv)
scatter_csv = save_dir / "04_fig4_scatter.csv"
scatter_data.to_csv(scatter_csv)


# activity episodes?
ac_mask = scatter_data.iloc[:, 0] == "activity"
activity_data = scatter_data[ac_mask]
sl_mask = scatter_data.iloc[:, 0] == 'sleep'
sleep_data = scatter_data[sl_mask]

median = "Mean"
count = "Count"
time_col = COL_NAMES[2]
protocol_col = COL_NAMES[1]
subjects = COL_NAMES[3]
type_col = COL_NAMES[0]

act_test_dir = save_csv_dir / "01_activity"
sl_test_dir = save_csv_dir / "02_sleep"

act_ph_dict = {}
sl_ph_dict = {}

# 1. Are the number of episodes different?

count_csv = "01_count.csv"
print(count_csv)
ac_ct_file = act_test_dir / count_csv
ac_count_rm = pg.mixed_anova(dv=count,
                             within=time_col,
                             between=protocol_col,
                             subject=subjects,
                             data=activity_data)
print('Activity')
pg.print_table(ac_count_rm)
ac_count_rm.to_csv(ac_ct_file)
ph_count_act = prep.tukey_pairwise_ph(
    activity_data,
    hour_col=protocol_col,
    dep_var=count,
    protocol_col=time_col
)
ph_name = act_test_dir / "04_count_ph.csv"
ph_count_act.to_csv(ph_name)
act_ph_dict[count] = ph_count_act

sl_ct_file = sl_test_dir / count_csv
sl_count_rm = pg.mixed_anova(dv=count,
                             within=time_col,
                             between=protocol_col,
                             subject=subjects,
                             data=sleep_data)
print('Sleep')
pg.print_table(sl_count_rm)
sl_count_rm.to_csv(sl_ct_file)
ph_count_sl = prep.tukey_pairwise_ph(
    sleep_data,
    hour_col=protocol_col,
    dep_var=count,
    protocol_col=time_col
)
ph_name = sl_test_dir / "04_count_ph.csv"
ph_count_sl.to_csv(ph_name)
sl_ph_dict[count] = ph_count_sl

# 2. Are the duration of episodes different?

med_csv = "02_mean.csv"
print(med_csv)
ac_med_file = act_test_dir / med_csv
ac_med_rm = pg.mixed_anova(dv=median,
                           within=time_col,
                           between=protocol_col,
                           subject=subjects,
                           data=activity_data)
print('Activity')
pg.print_table(ac_med_rm)
ac_med_rm.to_csv(ac_med_file)
ph_med_ac = prep.tukey_pairwise_ph(
    activity_data,
    hour_col=protocol_col,
    dep_var=median,
    protocol_col=time_col
)
ph_name = act_test_dir / "03_med_ph.csv"
ph_med_ac.to_csv(ph_name)
act_ph_dict[median] = ph_med_ac

sl_med_file = sl_test_dir / med_csv
sl_med_rm = pg.mixed_anova(dv=median,
                           within=time_col,
                           between=protocol_col,
                           subject=subjects,
                           data=sleep_data)
print('Sleep')
pg.print_table(sl_med_rm)
sl_med_rm.to_csv(sl_med_file)

# Significant differences on ANOVA so post hoc Tukeys test
ph_med_sl = prep.tukey_pairwise_ph(
    sleep_data,
    hour_col=protocol_col,
    dep_var=median,
    protocol_col=time_col
)
ph_name = sl_test_dir / "03_med_ph.csv"
ph_med_sl.to_csv(ph_name)
sl_ph_dict[median] = ph_med_sl


# Q3 What duration of episodes are affected?
duration_col = hist_cols[-2]
no_ep_col = hist_cols[-1]
hist_ph_dir = save_csv_dir / "03_histogram"

test_hist = long_hist.set_index([hist_cols[0], hist_cols[1]])

def groupby_ph_test(test_hist):
    print(test_hist.iloc[0].index)
    hist_ph = prep.tukey_pairwise_ph(
        test_hist,
        hour_col=duration_col,
        dep_var=no_ep_col,
        protocol_col=time_col
    )
    return hist_ph

hist_ph = test_hist.groupby(
    level=[0, 1]
).apply(
    groupby_ph_test
)
hist_ph_name = hist_ph_dir / "01_hist_ph.csv"
hist_ph.to_csv(hist_ph_name)

######### Step 4 Plot all together#############################################

# plotting constants
# looping/selecting data constants
condition_col = COL_NAMES[1]
section_col = COL_NAMES[2]
measurement_col = COL_NAMES[-1]
animal_col = COL_NAMES[-2]
data_type_col = COL_NAMES[0]
data = long_hist
data_types = data[data_type_col].unique()
conditions = data[condition_col].unique()
sections = data[section_col].unique()
animals = data[animal_col].unique()
duration_col = hist_cols[-2]
no_ep_col = hist_cols[-1]
marker_types = [count_cols[-1], med_cols[-1]]
med_count_data = {
    marker_types[0]: count_long,
    marker_types[1]: median_long
}

# sig constants
sig_ylevel_disrupt = 0.9
sig_ylevel_recovery = 0.95
ph_med_count_dict = dict(zip(data_types, [act_ph_dict, sl_ph_dict]))
minus_sigval = 0.3
plus_sigval = 0.3

# Tidy constants
sleep_hist_ylim = [0, 120]
act_hist_ylim = [0, 375]
mean_ylim = [25, 275]
xlabelsize = 10
ylabelsize = 10
tick_label_size = 8
panel_size = 8
panel_xpos = -0.17
panel_ypos = 0.9
marker_size = 3
mean_count_ylabel_dict = dict(
    zip(
        marker_types,
        ["No. Episodes", "Mean Duration"]
    )
)
count_med_panels = {
    marker_types[0]: {
        data_types[0]: "A",
        data_types[1]: "G",
    },
    marker_types[1]: {
        data_types[0]: "B",
        data_types[1]: "H",
    },
}
hist_panels = [
    ["C", "D", "E", "F"],
    ["I", "J", "K", "L"]
]


# point/swarm plot
dodge = 0.5
capsize = 0.2
errwidth = 1
sem = 68


# initialise figure
fig = plt.figure()

count_grid = gs.GridSpec(
    nrows=1,
    ncols=2,
    figure=fig,
    top=0.9,
    bottom=0.75
)
count_axes = [plt.subplot(x) for x in count_grid]
median_grid = gs.GridSpec(
    nrows=1,
    ncols=2,
    figure=fig,
    top=0.75,
    bottom=0.6
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
        
        # add in vertical lines 
        curr_ax.axvline(0.5, color='k', ls="--")
        curr_ax.axvline(1.5, color='k', ls="--")
        curr_ax.axvline(2.5, color='k', ls="--")


        # Tidy axis
        if marker == marker_types[1]:
            curr_ax.set_ylim(mean_ylim)
        curr_ax.tick_params(
            axis='both',
            which='major',
            labelsize=tick_label_size
        )
        curr_ax.set(xlabel="")
        if data_type == data_types[0]:
            curr_ax.set_ylabel(
                mean_count_ylabel_dict[marker],
                fontsize=ylabelsize
            )
        if data_type == data_types[1]:
            curr_ax.set_ylabel("")
        curr_panel = count_med_panels[marker][data_type]
        curr_ax.text(
            panel_xpos,
            panel_ypos,
            curr_panel,
            transform=curr_ax.transAxes,
            fontsize=panel_size
        )
        
        # Significance lines
        
        # Get y vals, xvals where sig, plot
        ycoor_disrupt = aplot.sig_line_coord_get(
            curr_ax=curr_ax,
            sig_line_ylevel=sig_ylevel_disrupt
        )
        ycoord_recovery = aplot.sig_line_coord_get(
            curr_ax=curr_ax,
            sig_line_ylevel=sig_ylevel_recovery
        )
        
        # xvals needs dict to lookup
        xtick_dict = aplot.get_xtick_dict(
            curr_ax=curr_ax
        )
        curr_ph_df = ph_med_count_dict[data_type][marker]
        disrupt_xvals = aplot.sig_locs_get(
            df=curr_ph_df,
            index_level2val=0
        )
        recovery_xvlas = aplot.sig_locs_get(
            df=curr_ph_df,
            index_level2val=1
        )
        aplot.draw_sighlines(
            yval=ycoor_disrupt,
            sig_list=disrupt_xvals,
            label_loc_dict=xtick_dict,
            minus_val=minus_sigval,
            plus_val=0,
            curr_ax=curr_ax,
            color="C1"
        )
        aplot.draw_sighlines(
            yval=ycoord_recovery,
            sig_list=recovery_xvlas,
            label_loc_dict=xtick_dict,
            minus_val=minus_sigval,
            plus_val=plus_sigval,
            curr_ax=curr_ax,
            color="C2"
        )
        
        

# Plot histograms
# Create axes
hist_grid = gs.GridSpec(
    nrows=4,
    ncols=2,
    figure=fig,
    top=0.55,
    bottom=0.1,
    hspace=0,
)
hist_axes = [plt.subplot(x) for x in hist_grid]
act_axes = hist_axes[::2]
sl_axes = hist_axes[1::2]
both_hist_axes = [act_axes, sl_axes]
# loop through the data types
for col_no, data_type in enumerate(data_types):
    
    # select the axis list
    curr_ax_list = both_hist_axes[col_no]
    
    # select just the data type
    curr_data_type = long_hist.query("%s == '%s'"%(data_type_col, data_type))

    # loop through all the conditions
    for row_no, condition in enumerate(conditions):
        

        # select the data and axis
        curr_hist_ax = curr_ax_list[row_no]
        curr_data = curr_data_type.query("%s == '%s'"%(condition_col,
                                                       condition))
        
        sns.barplot(
            x=duration_col,
            y=no_ep_col,
            hue=section_col,
            data=curr_data,
            ax=curr_hist_ax,
            # join=False,
            dodge=dodge,
            capsize=capsize,
            errwidth=errwidth,
            ci=sem
        )
        sns.swarmplot(
            x=duration_col,
            y=no_ep_col,
            hue=section_col,
            data=curr_data,
            ax=curr_hist_ax,
            dodge=dodge,
            size=marker_size,
            color='k'
        )
        curr_legend = curr_hist_ax.legend()
        curr_legend.remove()
        
        # Tidy axis
        if data_type == data_types[0]:
            curr_hist_ax.set_ylim(act_hist_ylim)
            curr_hist_ax.set_ylabel(
                conditions[row_no],
                fontsize=ylabelsize
            )
        if data_type == data_types[1]:
            curr_hist_ax.set_ylim(sleep_hist_ylim)
            curr_hist_ax.set_ylabel("")
        curr_hist_ax.tick_params(
            axis='both',
            which='major',
            labelsize=tick_label_size
        )
        curr_hist_ax.set_xlabel(
            "Episode Duration, minutes",
            fontsize=xlabelsize
        )
        curr_panel = hist_panels[col_no][row_no]
        curr_hist_ax.text(
            panel_xpos,
            panel_ypos,
            curr_panel,
            transform=curr_hist_ax.transAxes,
            fontsize=panel_size
        )
        
        
        # Significance
        # Get y vals, xvals where sig, plot
        ycoor_disrupt = aplot.sig_line_coord_get(
            curr_ax=curr_hist_ax,
            sig_line_ylevel=sig_ylevel_disrupt
        )
        ycoord_recovery = aplot.sig_line_coord_get(
            curr_ax=curr_hist_ax,
            sig_line_ylevel=sig_ylevel_recovery
        )
        
        # xvals needs dict to lookup
        xtick_dict = aplot.get_xtick_dict(
            curr_ax=curr_hist_ax
        )
        curr_ph_df = hist_ph.loc[idx[data_type, condition], :]
        disrupt_xvals = aplot.sig_locs_get(
            df=curr_ph_df,
            index_level2val=0
        )
        recovery_xvlas = aplot.sig_locs_get(
            df=curr_ph_df,
            index_level2val=1
        )
        aplot.draw_sighlines(
            yval=ycoor_disrupt,
            sig_list=disrupt_xvals,
            label_loc_dict=xtick_dict,
            minus_val=minus_sigval,
            plus_val=0,
            curr_ax=curr_hist_ax,
            color="C1"
        )
        aplot.draw_sighlines(
            yval=ycoord_recovery,
            sig_list=recovery_xvlas,
            label_loc_dict=xtick_dict,
            minus_val=minus_sigval,
            plus_val=plus_sigval,
            curr_ax=curr_hist_ax,
            color="C2"
        )
# set the legend
handles, legends = curr_ax.get_legend_handles_labels()
fig.legend(
    handles=handles[:3],
    loc=(0.87, 0.91),
    fontsize=tick_label_size,
    markerscale=0.5
)

# Set titles around plot
fig.suptitle(
    "Effect of light conditions on episodes of activity and sleep"
)
type_label_xval = 0.5
type_label_yval = 1.1
col_title_size = ylabelsize
fig.text(
    type_label_xval,
    type_label_yval,
    "Activity",
    transform=count_axes[0].transAxes,
    fontsize=col_title_size,
    ha='center'
)
fig.text(
    type_label_xval,
    type_label_yval,
    "Sleep",
    transform=count_axes[1].transAxes,
    fontsize=col_title_size,
    ha='center'
)
fig.text(
    0.02,
    0.75,
    "Percentage of baseline mean, mean +/-SEM",
    rotation=90,
    fontsize=ylabelsize,
    va='center'
)
fig.text(
    0.02,
    0.325,
    "Number of episodes, mean +/-SEM",
    rotation=90,
    fontsize=ylabelsize,
    va='center'
)

fig.set_size_inches(8.27, 11.69)
plt.savefig(SAVE_FIG, dpi=600)

plt.close('all')
