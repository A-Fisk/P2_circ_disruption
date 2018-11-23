# script to create plots of count_per_day
# in input dir

import pathlib
import seaborn as sns
sns.set()
import sys
sys.path.insert(0, "/Users/angusfisk/Documents/01_PhD_files/07_python_package/"
                   "actiPy")
import actiPy.preprocessing as prep
import actiPy.analysis as als

input_directory = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/"
                                "01_projects/P2_Circ_Disruption_paper_chapt2/"
                                "01_data_files/02_sleep/00_clean")
save_directory = input_directory.parents[2] / "03_analysis_outputs/02_sleep"
subdir_name = '03_count_per_day'

save_csv_dir = input_directory.parent

init_kwargs = {
    "input_directory": input_directory,
    "save_directory": save_csv_dir,
    "subdir_name": subdir_name,
    "func": (prep, "read_file_to_df"),
    "index_col": [0, 1],
    "header": [0]
}
count_object = prep.SaveObjectPipeline(**init_kwargs)

process_kwargs = {
    "function": (als, "count_per_day"),
    "create_df": True,
    "savedfcsv": True,
}
count_object.process_file(**process_kwargs)

# convert to hours for plotting
count_object.processed_df = prep._convert_to_units(
    count_object.processed_df,
    base_freq="10s",
    target_freq="1H",
)

save_plot_path = prep.create_subdir(
    save_directory,
    subdir_name
)

plot_kwargs = {
    "function": (als, "pointplot_from_df"),
    "showfig": False,
    "savefig": True,
    "data_list": [count_object.processed_df],
    "file_list": [count_object.subdir_path.stem],
    "xlevel": 'light_period',
    "groups": 'group',
    "dodge": True,
    "capsize": 0.1,
    "remove_col": False,
    "figsize": (10, 10),
    "subdir_path": save_plot_path,
    "ylabel": "sleep_count_per_day, Hours"
}
count_object.create_plot(**plot_kwargs)


