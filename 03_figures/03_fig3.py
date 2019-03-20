# Script to create figure 3
# percentage change from baseline for IV, IS, Qp, others?

# start with standard imports
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

###### Step 1
#  define constants
index_cols = [0, 1]
idx = pd.IndexSlice
save_fig = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/"
                        "01_projects/01_thesisdata/02_circdis/"
                        "03_analysis_outputs/03_figures/03_fig3.png")
save_csv = save_fig.parent / "00_csvs/03_fig3.csv"
LDR_COL = -1
col_names = ["condition", "section", "animal", "measurement"]

def longform(df,
             col_names: list):
    new_df = df.stack().reset_index()
    new_df.columns = col_names
    
    return new_df

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
activity_iv = als.intradayvar(activity_df, level=[0, 1])
sleep_iv = als.intradayvar(sleep_df, level=[0, 1])

iv_cols = col_names.copy()
iv_cols[-1] = "Intraday Variability"

activity_iv = longform(activity_iv, col_names=iv_cols)
sleep_iv = longform(sleep_iv, col_names=iv_cols)

####### Step 4 Calculate Periodogram power
# calculate LS periodogram
activity_power = activity_df.groupby(level=[0, 1]).apply(per.get_period,
                                               return_power=True,
                                                drop_lastcol=False)
sleep_power = sleep_df.groupby(level=[0, 1]).apply(per.get_period,
                                                   return_power=True,
                                                   drop_lastcol=False)
for df in activity_power, sleep_power:
    df.columns = df.columns.droplevel(1)
    
# calculate max periodogram power baseline-dsirupted
activity_max_power_raw = activity_power.groupby(level=[0, 1]).max()
sleep_max_power_raw = sleep_power.groupby(level=[0, 1]).max()

# turn into longform
power_cols = col_names.copy()
power_cols[-1] = "Periodogram Power"

activity_max_power = longform(activity_max_power_raw, col_names=power_cols)
sleep_max_power = longform(sleep_max_power_raw, col_names=power_cols)

####### Step 5 Calculate IS
# Calculate IS baseline-disrupted
# going to calculate IS based on LS periodogram - can argue about
# qp vs ls IS later

# calculate number of data points to normalise
activity_samples = activity_df.groupby(level=[0, 1]).count()
sleep_samples = sleep_df.groupby(level=[0, 1]).count()

# divide max by number of samples to get IS
activity_is = activity_max_power_raw / activity_samples
sleep_is = sleep_max_power_raw / sleep_samples

# turn into longform data
is_cols = col_names.copy()
is_cols[-1] = "Interday Stability"

activity_is = longform(activity_is, col_names=is_cols)
sleep_is = longform(sleep_is, col_names=is_cols)

# get all together for export for SNP
processed_data_list = [activity_max_power, activity_iv, activity_is,
                       sleep_max_power, sleep_iv, sleep_is]
processed_data = pd.concat(processed_data_list, sort=False)
processed_data.to_csv(save_csv)

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

# create figure with right number of rows and cols
fig, ax = plt.subplots(nrows=len(activity_list), ncols=nocols)

# set default font size
plt.rcParams.update({"font.size": 10})
label_size = 8

# loop throuh sleep/activity in different columns
for col_no, data_list in enumerate(data_type_list):
    axis_column = ax[:, col_no]
    
    # loop through each measurement type for each row
    for row_no, data in enumerate(data_list):
        
        measurement_col = data.columns[-1]
        curr_ax = axis_column[row_no]
        conditions = data[condition_col].unique()

        # use gridspec from subplot spec to put in the right number of subplots
        inner_grid =  gs.GridSpecFromSubplotSpec(nrows=1,
                                                 ncols=4,
                                                 subplot_spec=curr_ax,
                                                 wspace=0,
                                                 hspace=0)

        # select correct sub plot and slice the data
        for condition, grid in zip(conditions, inner_grid):
            
            # slice the data
            mask = data[condition_col] == condition
            condition_data = data.where(mask).dropna()
            
            # create the new sub sub plot
            ax1 = plt.Subplot(fig, grid)
            fig.add_subplot(ax1)

            # plot the connecting lines for each animal
            sns.pointplot(x=section_col, y=measurement_col, hue=animal_col,
                          data=condition_data, ax=ax1, color="b", markers="None")
            # plot the summary values
            sns.pointplot(x=section_col, y=measurement_col, data=condition_data,
                          ax=ax1, ci=sem, join=False, color='k', capsize=0.2,
                          errwidth=1)
            
            # set the alpha for the connecting lines
            mean_dots = ax1.get_children()[6]
            connecting_lines = ax1.get_children()[7:28:4]
            plt.setp(connecting_lines, alpha=0.5, color='b')
            # all_lines = ax1.get_children()
            # plt.setp(all_lines, alpha=0)
            # test_child = ax1.get_children()[11]
            # plt.setp(test_child, alpha=1)
            
            # remove the legend
            ax_leg = ax1.legend()
            ax_leg.remove()
            
            # remove the axis label
            ax1.yaxis.label.set_visible(False)
            ax1.set_xticklabels(ax1.get_xticklabels(), visible=False)
            
            # set the conditions on the bottom row
            if row_no == 2:
                ax1.set_xticklabels(ax1.get_xticklabels(), visible=True,
                                    rotation=45, size=label_size)
            ax1.set_xlabel("")
            
            # set the yaxis
            if condition != conditions[0]:
                ax1.set_yticklabels(ax1.get_yticklabels(), visible=False)
            ax1.tick_params(axis="y", which="major", labelsize=label_size)
            min = data.min()[measurement_col]
            max = data.max()[measurement_col]
            ax1.set(ylim=[min, max])

            # label the columns with the conditions
            if row_no == 0:
                ax1.set_title(condition, rotation=45, va='bottom',
                              size=label_size)
                
        # remove the subplot level markers
        curr_ax.set_xticklabels(curr_ax.get_xticklabels(), visible=False)
        curr_ax.set_yticks([])
        
        # label the measurement type on LHS
        curr_ax.yaxis.set_label_coords(-0.1, 0.5)
        if col_no == 0:
            curr_ax.set_ylabel(measurement_col, size=label_size)
        
# set sizing parameters
plt.subplots_adjust(hspace=0, wspace=0.1)
fig.set_size_inches(8.27, 11.69)

# set titles
fig.suptitle("Effect on different light conditions on circadian "
             "disruption markers. Mean +/- SEM")

plt.savefig(save_fig, dpi=600)

plt.close('all')
