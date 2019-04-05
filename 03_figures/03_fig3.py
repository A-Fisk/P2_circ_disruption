# Script to create figure 3
# percentage change from baseline for IV, IS, Qp, others?

# start with standard imports
import pathlib
import pandas as pd
import numpy as np
import pingouin as pg
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
import actiPy.plots as aplot

###### Step 1
#  define constants
index_cols = [0, 1]
idx = pd.IndexSlice
save_fig = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/"
                        "01_projects/01_thesisdata/02_circdis/"
                        "03_analysis_outputs/03_figures/03_fig3.png")
save_csv_dir = save_fig.parent / "00_csvs/"
LDR_COL = -1
col_names = ["protocol", "time", "animal", "measurement"]

def longform(df,
             col_names: list):
    new_df = df.stack().reset_index()
    new_df.columns = col_names
    
    return new_df


def norm_base_mean(protocol_df, baseline_str: str = "Baseline"):
    base_values = protocol_df.loc[idx[:, baseline_str], :]
    normalise_value = base_values.mean().mean()
    normalised_df = (protocol_df / normalise_value) * 100
    return normalised_df


###### Step 2 tidy data
#  import the files to read
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
activity_names = [x.name for x in activity_dfs]
sleep_dfs = [prep.read_file_to_df(x, index_col=index_cols)
             for x in sleep_filenames]
sleep_names = [x.name for x in sleep_dfs]
activity_dict = dict(zip(activity_names, activity_dfs))
sleep_dict = dict(zip(sleep_names, sleep_dfs))
activity_df_ldr = pd.concat(activity_dict)
sleep_df_ldr = pd.concat(sleep_dict)

# remove LDR
ldr_label = activity_df_ldr.columns[LDR_COL]
activity_df = activity_df_ldr.drop(ldr_label, axis=1)
sleep_df = sleep_df_ldr.drop(ldr_label, axis=1)

######## Step 3 Calculate IV

activity_iv_raw = als.intradayvar(activity_df, level=[0, 1])
sleep_iv_raw = als.intradayvar(sleep_df, level=[0, 1])
iv_cols = col_names.copy()
iv_cols[-1] = "Intraday Variability"

# normalise to mean
activity_iv_norm = activity_iv_raw.groupby(level=0).apply(norm_base_mean)
sleep_iv_norm = sleep_iv_raw.groupby(level=0).apply(norm_base_mean)

# fix animal columns for stats
ac_iv_label = activity_iv_norm.groupby(level=0).apply(prep.label_anim_cols)
sl_iv_label = sleep_iv_norm.groupby(level=0).apply(prep.label_anim_cols)

# tidy for plotting
activity_iv = longform(ac_iv_label, col_names=iv_cols)
sleep_iv = longform(sl_iv_label, col_names=iv_cols)

####### Step 4 Calculate Periodogram power

# calculate the enright periodogram
activity_qp = activity_df.groupby(
    level=[0, 1]
).apply(
    per._enright_periodogram,
    level=[0,1]
)
sleep_qp = sleep_df.groupby(
    level=[0, 1]
).apply(
    per._enright_periodogram,
    level=[0, 1]
)

# calculate LS periodogram
activity_power = activity_df.groupby(
    level=[0, 1]
).apply(
    per.get_period,
    return_power=True,
    return_periods=False,
    drop_lastcol=False
)
activity_periods = activity_df.groupby(
    level=[0, 1]
).apply(
    per.get_period,
    return_power=False,
    return_periods=True,
    drop_lastcol=False
)
sleep_power = sleep_df.groupby(
    level=[0, 1]
).apply(
    per.get_period,
    return_power=True,
    return_periods=False,
    drop_lastcol=False
)
sleep_periods = sleep_df.groupby(
    level=[0, 1]
).apply(
    per.get_period,
    return_power=False,
    return_periods=True,
    drop_lastcol=False

for df in activity_power, sleep_power:
    df.columns = df.columns.droplevel(1)

# calculate max periodogram power
activity_max_power_raw = activity_power.groupby(level=[0, 1]).max()
sleep_max_power_raw = sleep_power.groupby(level=[0, 1]).max()

# normalise to baseline mean
activity_max_norm = activity_max_power_raw.groupby(
    level=0
).apply(
    norm_base_mean
)
sleep_max_norm = sleep_max_power_raw.groupby(
    level=0
).apply(
    norm_base_mean
)

# Tidy into longform for stats and plotting
power_cols = col_names.copy()
power_cols[-1] = "Periodogram Power"
ac_max_power_label = activity_max_norm.groupby(
    level=0
).apply(
    prep.label_anim_cols
)
sl_max_power_label = sleep_max_norm.groupby(
    level=0
).apply(
    prep.label_anim_cols
)
activity_max_power = longform(ac_max_power_label, col_names=power_cols)
sleep_max_power = longform(sl_max_power_label, col_names=power_cols)

####### Step 5 Calculate IS
# Calculate IS

# get max values from enright periodogram
ac_qp_max = activity_qp.groupby(level=[0, 1]).max()
sl_qp_max = sleep_qp.groupby(level=[0, 1]).max()
# calculate number of data points to normalise
activity_samples = activity_df.groupby(level=[0, 1]).count()
sleep_samples = sleep_df.groupby(level=[0, 1]).count()

# divide max by number of samples to get IS
activity_is_raw = ac_qp_max / activity_samples
sleep_is_raw = sl_qp_max / sleep_samples

# normalise to baseline mean
activity_is_norm = activity_is_raw.groupby(level=0).apply(norm_base_mean)
sleep_is_norm = sleep_is_raw.groupby(level=0).apply(norm_base_mean)

# Tidy into longform data for stats and plotting
is_cols = col_names.copy()
is_cols[-1] = "Interday Stability"
ac_is_label = activity_is_norm.groupby(level=0).apply(prep.label_anim_cols)
sl_is_label = sleep_is_norm.groupby(level=0).apply(prep.label_anim_cols)
activity_is = longform(ac_is_label, col_names=is_cols)
sleep_is = longform(sl_is_label, col_names=is_cols)

######### Step 6 Calculate light phase activity

# for non-free running
# mask where lights are high
# sum where lights are high
# sum total
# tada value
def light_phase_activity_nfreerun(
        test_df,
        ldr_label: str="LDR",
        ldr_val: float=150
):
    light_mask = test_df.loc[:, ldr_label] > ldr_val
    light_data = test_df[light_mask]
    light_sum = light_data.sum()
    total_sum = test_df.sum()
    light_phase_activity = light_sum / total_sum
    
    return light_phase_activity

activity_nfree_light = activity_df_ldr.groupby(
    level=[0, 1]
).apply(
    light_phase_activity_nfreerun
)
sleep_nfree_light = sleep_df_ldr.groupby(
    level=[0, 1]
).apply(
    light_phase_activity_nfreerun
)

# for free running Time
# split by period
# sum 0-12 / 0-24
# tada value
# set all baseline periods to be 24 hours
for df in [activity_periods, sleep_periods]:
    df.loc[idx[:, "Baseline"], :] = pd.Timedelta("1D")
activity_split = prep.split_list_with_periods(
    name_df=activity_periods,
    df_list=activity_dfs
)
sleep_split = prep.split_list_with_periods(
    name_df=sleep_periods,
    df_list=sleep_dfs
)

def light_phase_activity_freerun(test_df,
                                 start_light = "2010-01-01 00:00:00",
                                 end_light = "2010-01-01 12:00:00"):
    light_data = test_df.loc[idx[:, :, :, start_light:end_light], :]
    light_sum = light_data.sum()
    total_sum = test_df.sum()
    light_phase_activity = light_sum / total_sum
    
    return light_phase_activity


activity_free_light = activity_split.groupby(
    level=[0, 1, 2]
).apply(
    light_phase_activity_freerun
).mean(axis=1).unstack(level=2)
sleep_free_light = sleep_split.groupby(
    level=[0, 1, 2]
).apply(
    light_phase_activity_freerun
).mean(axis=1).unstack(level=2)

# Add free running for LL and T20 Disrupted to the final df
activity_lightphase = activity_nfree_light.drop(ldr_label, axis=1)
ll_values = activity_free_light.loc[idx["LL", "Disrupted"], :]
activity_lightphase.loc[idx["LL", "Disrupted"], :].update(ll_values)
t20_values = activity_free_light.loc[idx["T20", "Disrupted"], :]
activity_lightphase.loc[idx["T20", "Disrupted"], :].update(t20_values)

sleep_lightphase = sleep_nfree_light.drop(ldr_label, axis=1)
ll_values = sleep_free_light.loc[idx["LL", "Disrupted"], :]
sleep_lightphase.loc[idx["LL", "Disrupted"], :].update(ll_values)
t20_values = sleep_free_light.loc[idx["T20", "Disrupted"], :]
sleep_lightphase.loc[idx["T20", "Disrupted"], :].update(t20_values)

# Get into correct format

# norm base mean
activity_lightphase_norm = activity_lightphase.groupby(
    level=0
).apply(
    norm_base_mean
)
sleep_lightphase_norm = sleep_lightphase.groupby(
    level=0
).apply(
    norm_base_mean
)

# label animal cols
ac_lp_norm_label = activity_lightphase_norm.groupby(
    level=0
).apply(
    prep.label_anim_cols
)
sl_lp_norm_label = sleep_lightphase_norm.groupby(
    level=0
).apply(
    prep.label_anim_cols
)

# longform
lp_cols = col_names.copy()
lp_cols[-1] = "Lightphase Activity"
ac_lp_tidy = longform(ac_lp_norm_label, col_names=lp_cols)
sl_lp_tidy = longform(sl_lp_norm_label, col_names=lp_cols)

########## Step 7 Relative amplitude

# Get individual animal mean hourly activity
activity_split_hourly = activity_split.groupby(
    level=[0, 1, 2]
).resample(
    "H", level=3
).mean()
activity_mean_wave = activity_split_hourly.mean(axis=1).unstack(level=2)
sleep_split_hourly = sleep_split.groupby(
    level=[0, 1, 2]
).resample(
    "H", level=3
).mean()
sleep_mean_wave = sleep_split_hourly.mean(axis=1).unstack(level=2)

def relative_amplitude(test_df):
    hourly_max = test_df.max()
    hourly_min = test_df.min()
    hourly_diff = hourly_max - hourly_min
    hourly_sum = hourly_max + hourly_min
    relative_amplitude = hourly_diff / hourly_sum
    
    return relative_amplitude

# calculate Relative amplitude
activity_ra = activity_mean_wave.groupby(
    level=[0, 1]
).apply(
    relative_amplitude
)
sleep_ra = sleep_mean_wave.groupby(
    level=[0, 1]
).apply(
    relative_amplitude
)

# norm base mean
ac_ra_norm = activity_ra.groupby(
    level=0
).apply(
    norm_base_mean
)
sl_ra_norm = sleep_ra.groupby(
    level=0
).apply(
    norm_base_mean
)

# label animal cols
ac_ra_norm_label = ac_ra_norm.groupby(
    level=0
).apply(
    prep.label_anim_cols
)
sl_ra_norm_label = sl_ra_norm.groupby(
    level=0
).apply(
    prep.label_anim_cols
)

# longform
ra_cols = col_names.copy()
ra_cols[-1] = "Relative Amplitude"
ac_ra_tidy = longform(ac_ra_norm_label, col_names=ra_cols)
sl_ra_tidy = longform(sl_ra_norm_label, col_names=ra_cols)

####### Step 8 Calculate total sleep and activity

# calculate total sleep and activity
ac_total_days = activity_split.groupby(
    level=[0, 1, 2]
).sum()
ac_total = ac_total_days.mean(axis=1).unstack(level=2)
sl_total_days = sleep_split.groupby(
    level=[0, 1, 2]
).sum()
sl_total = sl_total_days.mean(axis=1).unstack(level=2)

# norm base mean
ac_total_norm = ac_total.groupby(
    level=0
).apply(
    norm_base_mean
)
sl_total_norm = sl_total.groupby(
    level=0
).apply(
    norm_base_mean
)

# label animal cols
ac_total_norm_label = ac_total_norm.groupby(
    level=0
).apply(
    prep.label_anim_cols
)
sl_total_norm_label = sl_total_norm.groupby(
    level=0
).apply(
    prep.label_anim_cols
)

# longform
tot_cols = col_names.copy()
tot_cols[-1] = "Total Activity/Sleep"
ac_tot_tidy = longform(ac_total_norm_label, col_names=tot_cols)
sl_tot_tidy = longform(sl_total_norm_label, col_names=tot_cols)


####### Stats #######
subject = col_names[2]
within = col_names[1]
between = col_names[0]
protocols = ac_iv_label.index.get_level_values(0).unique()
ac_marks = [
    activity_iv,
    activity_max_power,
    activity_is,
    ac_lp_tidy,
    ac_ra_tidy,
    ac_tot_tidy
]
sl_marks = [
    sleep_iv,
    sleep_max_power,
    sleep_is,
    sl_lp_tidy,
    sl_ra_tidy,
    sl_tot_tidy
]
data_types = ["01_activity", "02_sleep"]
mark_cols = [
    iv_cols,
    power_cols,
    is_cols,
    lp_cols,
    ra_cols,
    tot_cols
]
save_test_dir = save_csv_dir / "03_fig3"
save_dir_names = [
    '01_iv',
    '02_per_power',
    '03_is',
    '04_lightphase',
    '05_rel_amp',
    '06_total'
]

# loop through all the markers, save all the output files
ph_df_type_dict = {}
# loop through and apply rm to each marker
for mark_list, data_type in zip([ac_marks, sl_marks], data_types):
    save_testtype_dir = save_test_dir / data_type
    ph_df_markers_dict = {}
    for no, mark_df in enumerate(mark_list):
        cols = mark_cols[no]
        curr_dep_var = cols[-1]
        curr_rm = pg.mixed_anova(dv=curr_dep_var,
                                 within=within,
                                 subject=subject,
                                 between=between,
                                 data=mark_df)
        print(curr_dep_var)
        pg.print_table(curr_rm)
        ph_dict = {}
# loop through the post hoc tests too
        for protocol in protocols:
            mask = mark_df[between] == protocol
            post_hoc_data = mark_df[mask]
            curr_ph = pg.pairwise_tukey(dv=curr_dep_var,
                                        between=within,
                                        data=post_hoc_data)
            print(protocol)
            pg.print_table(curr_ph)
# save the tests into a dict
            ph_dict[protocol] = curr_ph
        
        test_name = save_dir_names[no]
        curr_save_dir = save_testtype_dir / test_name
        
        ph_savename = curr_save_dir / '02_posthoc.csv'
        ph_df = pd.concat(ph_dict)
        ph_df_markers_dict[curr_dep_var] = ph_df
        ph_df.to_csv(ph_savename)
        
        anova_filename = curr_save_dir / "01_anova.csv"
        curr_rm.to_csv(anova_filename)
    
    ph_df_markers_df = pd.concat(ph_df_markers_dict)
    ph_df_type_dict[data_type] = ph_df_markers_df

ph_df_bothtypes = pd.concat(ph_df_type_dict)

# get all together for export for SNP
processed_data_list = [activity_max_power, activity_iv, activity_is,
                       sleep_max_power, sleep_iv, sleep_is]
processed_data = pd.concat(processed_data_list, sort=False)
save_dfcsv = save_csv_dir / "03_fig3.csv"
processed_data.to_csv(save_dfcsv)

###### Step 6 Plot
# plot all on the same figure

# constants
nocols = 2
condition_col = col_names[0]
section_col = col_names[1]
animal_col = col_names[-2]
sem = 68
activity_list = ac_marks
sleep_list = sl_marks
data_type_list = [activity_list, sleep_list]
sections = activity_iv[section_col].unique()
capsize = 0.2
errwidth = 1
dodge = 0.5
marker_size = 3
col_title_size = 13
label_size = 10
data_sep_value = 0.3
sig_val = 0.05
sig_col = "p-tukey"
sig_line_ylevel_disrupt = 0.9
sig_line_ylevel_recovery = 0.95

# create figure with right number of rows and cols
fig, ax = plt.subplots(nrows=len(activity_list), ncols=nocols)

# set default font size
plt.rcParams.update({"font.size": 10})

# loop throuh sleep/activity in different columns
for col_no, data_list in enumerate(data_type_list):
    axis_column = ax[:, col_no]
    
    # loop through each measurement type for each row
    for row_no, data in enumerate(data_list):
        
        measurement_col = data.columns[-1]
        curr_ax = axis_column[row_no]
        conditions = data[condition_col].unique()
        
        # plot using seaborn
        sns.pointplot(x=condition_col,
                      y=measurement_col,
                      hue=section_col,
                      data=data,
                      ax=curr_ax,
                      join=False,
                      capsize=capsize,
                      errwidth=errwidth,
                      dodge=dodge,
                      ci=sem)
        
        sns.swarmplot(x=condition_col,
                      y=measurement_col,
                      hue=section_col,
                      data=data,
                      ax=curr_ax,
                      dodge=dodge,
                      size=marker_size)
        
        # remove the legend
        ax_leg = curr_ax.legend()
        ax_leg.remove()
        
        # tidy axis
        curr_ax.set(xlabel="")
        if col_no == 0 and row_no == 4:
            ylim = [40, 120]
            curr_ax.set(ylim=ylim)
        # ylim=ylim)
        curr_ax.tick_params(axis='both', which='major', labelsize=label_size)
        if col_no == 0:
            curr_ax.set_ylabel(measurement_col, fontsize=label_size)
        else:
            curr_ax.set_ylabel("")

        # add in statistical sig bars

        # get y value for sig line
        ycoord_disrupt = aplot.sig_line_coord_get(
            curr_ax=curr_ax,
            sig_line_ylevel=sig_line_ylevel_disrupt
        )
        ycoord_recovery = aplot.sig_line_coord_get(
            curr_ax=curr_ax,
            sig_line_ylevel=sig_line_ylevel_recovery
        )
        
        # get the locations for xvals to look up
        locs = curr_ax.get_xticks()
        labels = curr_ax.get_xticklabels()
        label_text = [x.get_text() for x in labels]
        label_loc_dict = dict(zip(label_text, locs))
        
        # get x vals for sig line
        # get the right df to lookup
        marker_types = ph_df_bothtypes.index.get_level_values(1).unique()
        data_type_label = data_types[col_no]
        ph_type = ph_df_bothtypes.loc[data_type_label]
        ph_marker = ph_type.loc[measurement_col]

        # get x values where sig
        sig_mask = ph_marker[sig_col] < sig_val
        sig_vals = ph_marker[sig_mask]
        sig_disrupt = sig_vals.loc[idx[:, 0], :].index.get_level_values(0)
        sig_recovery = sig_vals.loc[idx[:, 1], :].index.get_level_values(0)

        # put xvals into coordinates for drawing on axis
        for xval_label in sig_disrupt:
            curr_xval = label_loc_dict[xval_label]
            hxvals = [curr_xval - data_sep_value, curr_xval]
            hxvals_axes = curr_ax.transLimits.transform([(hxvals[0], 0),
                                                         (hxvals[1], 0)])
            hxvals_axes_xval = hxvals_axes[:, 0]

            curr_ax.axhline(
                ycoord_disrupt,
                xmin=hxvals_axes_xval[0],
                xmax=hxvals_axes_xval[1],
                color='C1'
            )
            
        for xval_label in sig_recovery:
            curr_xval = label_loc_dict[xval_label]
            hxvals = [curr_xval- data_sep_value, curr_xval + data_sep_value]
            hxvals_axes = curr_ax.transLimits.transform([(hxvals[0], 0),
                                                         (hxvals[1], 0)])
            hxvals_axes_xval = hxvals_axes[:, 0]

            curr_ax.axhline(
                ycoord_recovery,
                xmin=hxvals_axes_xval[0],
                xmax=hxvals_axes_xval[1],
                color='C2'
            )

# set the legend
handles, legends = curr_ax.get_legend_handles_labels()
fig.legend(handles=handles[:3], loc=(0.85, 0.9), fontsize=label_size,
           markerscale=0.5)

# set sizing parameters
plt.subplots_adjust(hspace=0.3, wspace=0.2)

# set titles
fig.suptitle("Effect on different light conditions on circadian "
             "disruption markers. Mean +/- SEM")
type_label_xval = 0.5
type_label_yval = 1.1
fig.text(
    type_label_xval,
    type_label_yval,
    "Activity",
    transform=ax[0, 0].transAxes,
    fontsize=col_title_size,
    ha='center'
)
fig.text(
    type_label_xval,
    type_label_yval,
    "Sleep",
    transform=ax[0, 1].transAxes,
    fontsize=col_title_size,
    ha='center'
)

fig.set_size_inches(8.27, 11.69)
plt.savefig(save_fig, dpi=600)

plt.close('all')
