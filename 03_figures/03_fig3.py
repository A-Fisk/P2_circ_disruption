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
activity_names = [x.name for x in activity_dfs]
sleep_dfs = [prep.read_file_to_df(x, index_col=index_cols)
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

######## Step 3 Calculate IV
# first step need IV equation
activity_iv_raw = als.intradayvar(activity_df, level=[0, 1])
sleep_iv_raw = als.intradayvar(sleep_df, level=[0, 1])

iv_cols = col_names.copy()
iv_cols[-1] = "Intraday Variability"

# normalise to mean
activity_iv_norm = activity_iv_raw.groupby(level=0).apply(norm_base_mean)
sleep_iv_norm = sleep_iv_raw.groupby(level=0).apply(norm_base_mean)

# fix cols for stats
ac_iv_label = activity_iv_norm.groupby(level=0).apply(prep.label_anim_cols)
sl_iv_label = sleep_iv_norm.groupby(level=0).apply(prep.label_anim_cols)

# tidy for plotting
activity_iv = longform(ac_iv_label, col_names=iv_cols)
sleep_iv = longform(sl_iv_label, col_names=iv_cols)

####### Step 4 Calculate Periodogram power

# calculate the enright periodogram
activity_qp = activity_df.groupby(level=[0, 1]).apply(per._enright_periodogram,
                                                   level=[0,1])
sleep_qp = sleep_df.groupby(level=[0, 1]).apply(per._enright_periodogram,
                                                level=[0, 1])

# calculate LS periodogram
activity_power = activity_df.groupby(level=[0, 1]).apply(per.get_period,
                                                         return_power=True,
                                                         drop_lastcol=False)
sleep_power = sleep_df.groupby(level=[0, 1]).apply(per.get_period,
                                                   return_power=True,
                                                   drop_lastcol=False)
for df in activity_power, sleep_power:
    df.columns = df.columns.droplevel(1)

# calculate max periodogram power baseline-disrupted
activity_max_power_raw = activity_power.groupby(level=[0, 1]).max()
sleep_max_power_raw = sleep_power.groupby(level=[0, 1]).max()

# normalise
activity_max_norm = activity_max_power_raw.groupby(level=0).apply(
    norm_base_mean)
sleep_max_norm = sleep_max_power_raw.groupby(level=0).apply(norm_base_mean)

# turn into longform
power_cols = col_names.copy()
power_cols[-1] = "Periodogram Power"
ac_max_power_label = activity_max_norm.groupby(level=0
                                               ).apply(prep.label_anim_cols)
sl_max_power_label = sleep_max_norm.groupby(level=0
                                            ).apply(prep.label_anim_cols)
activity_max_power = longform(ac_max_power_label, col_names=power_cols)
sleep_max_power = longform(sl_max_power_label, col_names=power_cols)

####### Step 5 Calculate IS
# Calculate IS baseline-disrupted
# going to calculate IS based on LS periodogram - can argue about
# qp vs ls IS later

# get max values from qp
ac_qp_max = activity_qp.groupby(level=[0, 1]).max()
sl_qp_max = sleep_qp.groupby(level=[0, 1]).max()

# calculate number of data points to normalise
activity_samples = activity_df.groupby(level=[0, 1]).count()
sleep_samples = sleep_df.groupby(level=[0, 1]).count()
# divide max by number of samples to get IS
activity_is_raw = ac_qp_max / activity_samples
sleep_is_raw = sl_qp_max / sleep_samples
# normalise
activity_is_norm = activity_is_raw.groupby(level=0).apply(norm_base_mean)
sleep_is_norm = sleep_is_raw.groupby(level=0).apply(norm_base_mean)

# turn into longform data
is_cols = col_names.copy()
is_cols[-1] = "Interday Stability"
ac_is_label = activity_is_norm.groupby(level=0).apply(prep.label_anim_cols)
sl_is_label = sleep_is_norm.groupby(level=0).apply(prep.label_anim_cols)
activity_is = longform(ac_is_label, col_names=is_cols)
sleep_is = longform(sl_is_label, col_names=is_cols)

####### Stats #######
subject = col_names[2]
within = col_names[1]
between = col_names[0]
protocols = ac_iv_label.index.get_level_values(0).unique()
ac_marks = [activity_iv, activity_is, activity_max_power]
sl_marks = [sleep_iv, sleep_is, sleep_max_power]
data_types = ["01_activity", "02_sleep"]
mark_cols = [iv_cols, is_cols, power_cols]
save_test_dir = save_csv_dir / "03_fig3"
save_dir_names = ['01_iv', '02_is', '03_per_power']

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
activity_list = [activity_iv, activity_max_power, activity_is]
sleep_list = [sleep_iv, sleep_max_power, sleep_is]
data_type_list = [activity_list, sleep_list]
sections = activity_iv[section_col].unique()
capsize = 0.2
errwidth = 1
dodge = 0.5
marker_size = 3
col_title_size = 10
label_size = 8
data_sep_value = 0.3
sig_val = 0.05
sig_col = "p-tukey"
sig_line_ylevel = 0.9

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
        curr_marker = marker_types
        
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
        ylim = [0, 150]
        if row_no == 0:
            ylim = [0, 200]
        curr_ax.set(xlabel="")
        # ylim=ylim)
        curr_ax.tick_params(axis='both', which='major', labelsize=label_size)
        if col_no == 0:
            curr_ax.set_ylabel(measurement_col, fontsize=label_size)
        else:
            curr_ax.set_ylabel("")

        # add in statistical sig bars

        # get y value for sig line
        axes_to_data = curr_ax.transLimits.inverted()
        ycoords = (0.5, sig_line_ylevel)
        ycoords_data = axes_to_data.transform(ycoords)
        ycoords_data_val = ycoords_data[1]

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

            curr_ax.axhline(ycoords_data_val,
                            xmin=hxvals_axes_xval[0],
                            xmax=hxvals_axes_xval[1])
            
        for xval_label in sig_recovery:
            curr_xval = label_loc_dict[xval_label]
            hxvals = [curr_xval- data_sep_value, curr_xval + data_sep_value]
            hxvals_axes = curr_ax.transLimits.transform([(hxvals[0], 0),
                                                         (hxvals[1], 0)])
            hxvals_axes_xval = hxvals_axes[:, 0]

            curr_ax.axhline(ycoords_data_val,
                            xmin=hxvals_axes_xval[0],
                            xmax=hxvals_axes_xval[1])

# set the legend
handles, legends = curr_ax.get_legend_handles_labels()
fig.legend(handles=handles[:3], loc=(0.85, 0.9), fontsize=label_size,
           markerscale=0.5)

# set sizing parameters
plt.subplots_adjust(hspace=0.3, wspace=0.2)
fig.set_size_inches(8.27, 11.69)

# set titles
fig.suptitle("Effect on different light conditions on circadian "
             "disruption markers. Mean +/- SEM")
fig.text(0.25, 0.85, "Activity")
fig.text(0.75, 0.85, "Sleep")

plt.savefig(save_fig, dpi=600)

plt.close('all')
