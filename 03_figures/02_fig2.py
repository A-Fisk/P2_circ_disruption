# creating figure 2
# requires Mean waveforms for activity and sleep
# per hour with +/- SEM

# start with imports
import pathlib
import pandas as pd
import numpy as np
import pingouin as pg
import statsmodels.regression.mixed_linear_model as lm
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.dates as mdates
import sys

sys.path.insert(0, "/Users/angusfisk/Documents/01_PhD_files/"
                   "07_python_package/actiPy")
import actiPy.preprocessing as prep
import actiPy.periodogram as per
import actiPy.waveform as wave

# import the files we are going to read

def norm_base_mean(protocol_df, baseline_str: str = "Baseline"):
    base_values = protocol_df.loc[idx[:, baseline_str], :]
    normalise_value = base_values.mean()[0]
    normalised_df = (protocol_df / normalise_value) * 100
    return normalised_df

# define constants
index_cols = [0, 1]
idx = pd.IndexSlice
save_fig = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/"
                        "01_projects/01_thesisdata/02_circdis/"
                        "03_analysis_outputs/03_figures/02_fig2.png")

# get the file names
activity_dir = pathlib.Path('/Users/angusfisk/Documents/01_PhD_files/'
                            '01_projects/01_thesisdata/02_circdis/'
                            '01_data_files/01_activity/00_clean')

sleep_dir = pathlib.Path('/Users/angusfisk/Documents/01_PhD_files/'
                         '01_projects/01_thesisdata/02_circdis/'
                         '01_data_files/02_sleep/00_clean')

activity_filenames = sorted(activity_dir.glob("*.csv"))
sleep_filenames = sorted(sleep_dir.glob("*.csv"))

# import into lists
activity_dfs = [prep.read_file_to_df(x, index_col=index_cols)
                for x in activity_filenames]
sleep_dfs = [prep.read_file_to_df(x, index_col=index_cols)
             for x in sleep_filenames]

activity_hourly = [prep._resample(x) for x in activity_dfs]
sleep_hourly = [prep._resample(x) for x in sleep_dfs]

# find the hourly means for each condition
# need to find internal period and correct for that
# for disrupted and post_baseline times

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

# split based on period
activity_split = prep.split_list_with_periods(name_df=activity_periods,
                                              df_list=activity_dfs)

sleep_split = prep.split_list_with_periods(name_df=sleep_periods,
                                           df_list=sleep_dfs)

# find mean and sem
# want to find the mean for each individual animal then take the mean
# of the aggregate animal means and sem of them

# find means for each individual animal per condition/section
activity_mean_raw = wave.group_mean_df(activity_split)
sleep_mean_raw = wave.group_mean_df(sleep_split)

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
import statsmodels.stats.anova as smav
import statsmodels.formula.api as smf
import pingouin as pg


# testing activity
act_test_dir = save_test_dir / "01_activity"

three_way_model = smf.ols("Mean ~ C(Hour)*C(Time)*C(Protocol)",
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

# plot

# set values to iterate through to select right part of the data
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

# create figures
fig, ax = plt.subplots(nrows=len(activity_conditions),
                       ncols=2,
                       sharex=True)

# set font size
plt.rcParams.update({"font.size": 5})
# matplotlib.rc("ylabel", labelsize=3)
# matplotlib.rc("")

# loop through activity/sleep, conditions, sections
for col, df in enumerate([activity_mean, sleep_mean]):
    axis_column = ax[:, col]
    condition = both_conditions[col]
    
    # loop through conditions
    for condition_no, condition_label in enumerate(condition):
        curr_ax = axis_column[condition_no]
        
        # select the data and plot on the correct axis
        for section in sections:
            # select the data
            mean_data = df.loc[idx[condition_label, section], cols[0]]
            sem_data = df.loc[idx[condition_label, section], cols[1]]
            
            # plot the mean +/- sem on the subplot
            curr_ax.plot(mean_data, label=section)
            
            curr_ax.fill_between(mean_data.index, (mean_data - sem_data),
                                 (mean_data + sem_data), alpha=0.5)
            
            # set a grey background
            curr_ax.fill_between(mean_data.between_time("12:00:00",
                                                        "23:59:00").index,
                                 500,
                                 alpha=0.2, color='0.5')
        
        # set limits
        ylim = [0, 350]
        if col == 1:
            ylim = [0, 200]
        curr_ax.set(xlim=[mean_data.index[0],
                          mean_data.index[-1]],
                    ylim=ylim)
        
        # fix the ticks
        curr_ax.set_yticks([])
        curr_ax.set_xticks([])
        xfmt = mdates.DateFormatter("%H:%M:%S")
        curr_ax.xaxis.set_major_formatter(xfmt)
        
        # set the title for each condition
        curr_ax.set_title(condition_label)

        # get xvalues where significant
        ph_df_curr = ph_df_both[col]
        ph_df_protocol = ph_df_curr[condition_label]
        sig_mask = ph_df_protocol.loc[:, p_val_col] < sig_val
        ph_sig_times = ph_df_protocol[sig_mask
                       ].loc[idx[:, sig_indexlevel_ph_df], :
                       ].index.get_level_values(0)

        print(condition_label)
        sig_yval_curr = sig_yvals[col]
        # draw line at points of significance
        for xval in ph_sig_times:
            print(xval)
            hxvals = [(xval - min_30), (xval + min_30)]
            hxvals_shift = [x + min_30 for x in hxvals]
            hxvals_num = [mdates.date2num(x) for x in hxvals_shift]
            hxvals_transformed = curr_ax.transLimits.transform(
                [(hxvals_num[0], 0),
                 (hxvals_num[1], 0)])
            hxvals_trans_xvals = hxvals_transformed[:, 0]
            curr_ax.axhline(sig_yval_curr,
                            xmin=hxvals_trans_xvals[0],
                            xmax=hxvals_trans_xvals[1])


ax[0, 1].legend()

# set the size of the plot and text
fig.autofmt_xdate()
fig.set_size_inches(8.27, 11.69)
fig.subplots_adjust(hspace=0.5)
fig.suptitle("Mean activity and sleep under different disruption. +/- sem",
             fontsize=10)
fig.text(0.45, 0.05, "Circadian time (hours)", fontsize=10)

plt.savefig(save_fig, dpi=600)

plt.close('all')
