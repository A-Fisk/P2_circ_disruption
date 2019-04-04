# Figure 2. Mean activity and sleep over 24 Circadian hours

import pingouin as pg
import statsmodels.formula.api as smf
import statsmodels.stats.anova as smav
import actiPy.plots as aplot
import actiPy.waveform as wave
import actiPy.periodogram as per
import actiPy.preprocessing as prep
import sys
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import matplotlib
import pathlib
import pandas as pd
idx = pd.IndexSlice
sys.path.insert(0, "/Users/angusfisk/Documents/01_PhD_files/"
                   "07_python_package/actiPy")


def norm_base_mean(protocol_df, baseline_str: str = "Baseline"):
    base_values = protocol_df.loc[idx[:, baseline_str], :]
    normalise_value = base_values.mean()[0]
    normalised_df = (protocol_df / normalise_value) * 100
    return normalised_df


# Step 1: Read in files to analyse
index_cols = [0, 1]
save_fig = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/"
                        "01_projects/01_thesisdata/02_circdis/"
                        "03_analysis_outputs/03_figures/02_fig2.png")
activity_dir = pathlib.Path('/Users/angusfisk/Documents/01_PhD_files/'
                            '01_projects/01_thesisdata/02_circdis/'
                            '01_data_files/01_activity/00_clean')

sleep_dir = pathlib.Path('/Users/angusfisk/Documents/01_PhD_files/'
                         '01_projects/01_thesisdata/02_circdis/'
                         '01_data_files/02_sleep/00_clean')
activity_filenames = sorted(activity_dir.glob("*.csv"))
sleep_filenames = sorted(sleep_dir.glob("*.csv"))
activity_dfs = [prep.read_file_to_df(x, index_col=index_cols)
                for x in activity_filenames]
sleep_dfs = [prep.read_file_to_df(x, index_col=index_cols)
             for x in sleep_filenames]
activity_hourly = [prep._resample(x) for x in activity_dfs]  # turn into dfs
sleep_hourly = [prep._resample(x) for x in sleep_dfs]

# Step 2: Split into circadian days
# find internal period
activity_dict = {}
for df in activity_dfs:
    periods = df.groupby(level=0).apply(per.get_period)
    name = df.name
    activity_dict[name] = periods
activity_periods = pd.concat(activity_dict)

sleep_dict = {}
for df in sleep_dfs:
    periods = df.groupby(level=0).apply(per.get_period)
    name = df.name
    sleep_dict[name] = periods
sleep_periods = pd.concat(sleep_dict)

# set all baseline periods to be 24 hours
for df in [activity_periods, sleep_periods]:
    df.loc[idx[:, "Baseline"], :] = pd.Timedelta("1D")

# Split into days of given period length
activity_split = prep.split_list_with_periods(
    name_df=activity_periods,
    df_list=activity_dfs
)
sleep_split = prep.split_list_with_periods(
    name_df=sleep_periods,
    df_list=sleep_dfs
)

# Step 3: Find mean hourly activity

# find group means of individual animal means
activity_mean_raw = wave.group_mean_df(activity_split)
sleep_mean_raw = wave.group_mean_df(sleep_split)
# normalise
activity_mean = activity_mean_raw.groupby(level=0).apply(norm_base_mean)
sleep_mean = sleep_mean_raw.groupby(level=0).apply(norm_base_mean)

########## Stats ##########

# tidy data so can be fitted
col_names = ['Protocol', 'Time', 'Animal', 'Hour', "Mean"]
protocols = activity_split.index.get_level_values(0).unique()


def clean_df(df, col_names):
    temp_mean = df.mean(axis=1)
    temp_mean_h = temp_mean.groupby(level=[0, 1, 2]
                                    ).resample("H",
                                               level=3).mean()
    tidy_df = temp_mean_h.reset_index()
    tidy_df.columns = col_names

    return tidy_df


tidy_activity = clean_df(activity_split, col_names)
tidy_sleep = clean_df(sleep_split, col_names)

# fit two way repeated measures ANOVA to activity and sleep
# for each protocol - three way ANOVA?
dep_var = col_names[-1]
subject = col_names[2]
within = [col_names[3], col_names[1]]
between = col_names[0]
protocols = activity_mean.index.get_level_values(0).unique()
hours = tidy_activity["Hour"].unique()

save_test_dir = save_fig.parent / "00_csvs/02_fig2"

# 1. Do we have different effects of protocolxtimexhour on activity?
# 3 way anova


# testing activity
act_test_dir = save_test_dir / "01_activity"

three_way_model = smf.ols(
    "Mean ~ C(Hour)*C(Time)*C(Protocol)",
    tidy_activity
).fit()
three_way_table = smav.anova_lm(three_way_model)
three_way_name = act_test_dir / "01_threewayanova.csv"
three_way_table.to_csv(three_way_name)
print(three_way_model.summary())

curr_data = tidy_activity

# post hoc 2 way ANOVAs
ph_df_dict_act = {}
for protocol in protocols:
    mask = curr_data["Protocol"] == protocol
    protocol_df = curr_data[mask]
    two_way_model = smf.ols("Mean ~ C(Hour)*C(Time)",
                            protocol_df
                            ).fit()
    two_way_df = smav.anova_lm(two_way_model, typ=2)
    print(protocol)
    print(two_way_model.summary())

    # post hoc tukeys
    ph_dict = {}
    for hour in hours:
        hour_mask = protocol_df["Hour"] == hour
        hour_df = protocol_df[hour_mask]
        ph_t = pg.pairwise_tukey(dv=dep_var,
                                 between="Time",
                                 data=hour_df)
        print(hour)
        pg.print_table(ph_t)
        ph_dict[hour] = ph_t
    ph_df = pd.concat(ph_dict)
    ph_df_dict_act[protocol] = ph_df

    curr_test_dir = act_test_dir / protocol
    ph_test_name = curr_test_dir / "02_posthoc.csv"
    ph_df.to_csv(ph_test_name)
    two_way_name = curr_test_dir / "01_anova.csv"
    two_way_df.to_csv(two_way_name)
    # to csv


# testing sleep
sleep_test_dir = save_test_dir / "02_sleep"

three_way_model = smf.ols("Mean ~ C(Hour)*C(Time)*C(Protocol)",
                          tidy_sleep
                          ).fit()
three_way_table = smav.anova_lm(three_way_model)
three_way_name = sleep_test_dir / "01_threewayanova.csv"
three_way_table.to_csv(three_way_name)
print(three_way_model.summary())

curr_data = tidy_sleep

# post hoc 2 way ANOVAs
ph_df_dict_sleep = {}
for protocol in protocols:
    mask = curr_data["Protocol"] == protocol
    protocol_df = curr_data[mask]
    two_way_model = smf.ols("Mean ~ C(Hour)*C(Time)",
                            protocol_df
                            ).fit()
    two_way_df = smav.anova_lm(two_way_model, typ=2)
    print(protocol)
    print(two_way_model.summary())

# post hoc tukeys
    ph_dict = {}
    for hour in hours:
        hour_mask = protocol_df["Hour"] == hour
        hour_df = protocol_df[hour_mask]
        ph_t = pg.pairwise_tukey(dv=dep_var,
                                 between="Time",
                                 data=hour_df)
        print(hour)
        pg.print_table(ph_t)
        ph_dict[hour] = ph_t
    ph_df = pd.concat(ph_dict)
    ph_df_dict_sleep[protocol] = ph_df

    curr_test_dir = sleep_test_dir / protocol
    ph_test_name = curr_test_dir / "02_posthoc.csv"
    ph_df.to_csv(ph_test_name)
    two_way_name = curr_test_dir / "01_anova.csv"
    two_way_df.to_csv(two_way_name)
    # to csv

    # to csv of two way
# to csv

# temporary - save data to have a think about how to do this - SNP wants to
# help
save_dir = save_fig.parent / "00_csvs/02_fig2.csv"
stats_data = pd.concat([tidy_activity, tidy_sleep], sort=False)
stats_data.to_csv(save_dir)

########### Plot #########################################################

# plotting constants
activity_conditions = activity_mean.index.get_level_values(0).unique()
sleep_conditions = sleep_mean.index.get_level_values(0).unique()
both_conditions = [activity_conditions, sleep_conditions]
sections = activity_mean.index.get_level_values(1).unique()
cols = activity_mean.columns
min_30 = pd.Timedelta("30m")
ph_df_both = [ph_df_dict_act, ph_df_dict_sleep]
sig_val = 0.05
sig_indexlevel_ph_df = 0
sig_yvals = [300, 175]
p_val_col = "p-tukey"
xfmt = mdates.DateFormatter("%H:%M:%S")
hspace = 0.5
xfontsize = 8
panels = [["A", "B", "C", "D"], ["E", "F", "G", "H"]]
mainlabelsize = 12
panelsize = 10
dark_index = pd.DatetimeIndex(
    start="2010-01-01 12:00:00",
    end="2010-01-02 02:00:00",
    freq="H"
)
start_index = pd.Timestamp('2010-01-01 00:00:00')
end_index = pd.Timestamp("2010-01-02 00:00:00")
sns.set_style(
    "darkgrid",
    # {"xtick.bottom": "True",
    #  "ytick.left": "True",
    #  "axes.spines.bottom": "False",
    #  "axes.spines.left": "False"}
)


# Create figure and axes
fig = plt.figure()
activity_grid = gs.GridSpec(
    nrows=len(activity_conditions),
    ncols=1,
    figure=fig,
    right=0.45,
    left=0.15,
    hspace=hspace
)
activity_axes = [plt.subplot(x) for x in activity_grid]
sleep_grid = gs.GridSpec(
    nrows=len(sleep_conditions),
    ncols=1,
    figure=fig,
    right=0.85,
    left=0.55,
    hspace=hspace
)
sleep_axes = [plt.subplot(x) for x in sleep_grid]
both_axes = [activity_axes, sleep_axes]

# Plot data on correct axis
for col, df in enumerate([activity_mean, sleep_mean]):
    axis_column = both_axes[col]  # Select right axis column
    condition = both_conditions[col]
    panels_col = panels[col]

    # Plot conditions on each axis
    for condition_no, condition_label in enumerate(condition):
        curr_ax = axis_column[condition_no]

        # Plot each section on the same axis
        for section in sections:
            mean_data = df.loc[idx[condition_label, section], cols[0]]
            sem_data = df.loc[idx[condition_label, section], cols[1]]

            curr_ax.plot(
                mean_data,
                label=section
            )
            curr_ax.fill_between(
                mean_data.index,
                (mean_data - sem_data),
                (mean_data + sem_data),
                alpha=0.5
            )

            # Set the background to indicate lights
            curr_ax.fill_between(
                dark_index,
                # mean_data.between_time("12:00:00", "23:59:00").index,
                500,
                alpha=0.2,
                color='0.5'
            )

        # Fix the axes
        ylim = [0, 350]
        if col == 1:
            ylim = [0, 200]
        curr_ax.set(
            xlim=[start_index, end_index],
            ylim=ylim
        )
        curr_ax.xaxis.set_major_formatter(xfmt)
        for label in curr_ax.get_xticklabels():
            label.set_ha('right')
            label.set_rotation(30)
            label.set_fontsize(xfontsize)
        for ylabel in curr_ax.get_yticklabels():
            ylabel.set_fontsize(xfontsize)

        # Add in extra text
        if col == 0:
            curr_ax.text(
                -0.3,
                0.5,
                condition_label,
                transform=curr_ax.transAxes,
                rotation=90,
                fontsize=mainlabelsize
            )
        curr_ax.text(
            -0.2,
            1.1,
            panels_col[condition_no],
            transform=curr_ax.transAxes,
            fontsize=panelsize,
        )

        # get xvalues where significant
        ph_df_curr = ph_df_both[col]
        ph_df_protocol = ph_df_curr[condition_label]
        sig_disrupt = aplot.sig_locs_get(
            ph_df_protocol,
            index_level2val=0
        ) + min_30
        sig_recovery = aplot.sig_locs_get(
            ph_df_protocol,
            index_level2val=1
        ) + min_30
        # sig_mask = ph_df_protocol.loc[:, p_val_col] < sig_val
        # ph_sig_times = ph_df_protocol[sig_mask
        #                ].loc[idx[:, sig_indexlevel_ph_df], :
        #                ].index.get_level_values(0)

        print(condition_label)
        # sig_yval_curr = sig_yvals[col]
        sig_yval_disrupt = aplot.sig_line_coord_get(
            curr_ax,
            0.9
        )
        sig_yval_recovery = aplot.sig_line_coord_get(
            curr_ax,
            0.95
        )
        for xval in sig_disrupt:
            xvals = aplot.get_xval_dates(
                xval,
                minus_val=min_30,
                plus_val=min_30,
                curr_ax=curr_ax
            )
            curr_ax.axhline(
                sig_yval_disrupt,
                xvals[0],
                xvals[1],
                color='C1'
            )
        for xval in sig_recovery:
            xvals = aplot.get_xval_dates(
                xval,
                minus_val=min_30,
                plus_val=min_30,
                curr_ax=curr_ax
            )
            curr_ax.axhline(
                sig_yval_recovery,
                xvals[0],
                xvals[1],
                color='C2'
            )
        # # draw line at points of significance
        # for xval in ph_sig_times:
        #     print(xval)
        #     hxvals = [(xval - min_30), (xval + min_30)]
        #     hxvals_shift = [x + min_30 for x in hxvals]
        #     hxvals_num = [mdates.date2num(x) for x in hxvals_shift]
        #     hxvals_transformed = curr_ax.transLimits.transform(
        #         [(hxvals_num[0], 0),
        #          (hxvals_num[1], 0)])
        #     hxvals_trans_xvals = hxvals_transformed[:, 0]
        #     curr_ax.axhline(sig_yval_curr,
        #                     xmin=hxvals_trans_xvals[0],
        #                     xmax=hxvals_trans_xvals[1])


both_axes[1][0].legend(
    loc=(1.05, 0.65),
    prop={"size": xfontsize},
)

# Add in figure text
type_label_level = 1.1
type_label_xval = 0.5
fig.text(
    type_label_xval,
    type_label_level,
    "Activity",
    transform=activity_axes[0].transAxes,
    fontsize=mainlabelsize,
    ha='center'
)
fig.text(
    type_label_xval,
    type_label_level,
    "Sleep",
    transform=sleep_axes[0].transAxes,
    fontsize=mainlabelsize,
    ha='center'
)
fig.suptitle(
    "Activity and sleep profiles under different protocols",
    fontsize=mainlabelsize
)
fig.text(
    0.5,
    0.05,
    "Circadian time, hours",
    fontsize=mainlabelsize,
    ha='center'
)
fig.text(
    0.02,
    0.5,
    "Percentage of mean of baseline day +/- SEM",
    rotation=90,
    fontsize=mainlabelsize,
    va='center'
)

fig.set_size_inches(8.27, 11.69)
plt.savefig(save_fig, dpi=600)

plt.close('all')
