# script to create actograms and save for all data files
# in input dir

import pathlib
import sys
sys.path.insert(0, "/Users/angusfisk/Documents/01_PhD_files/07_python_package/"
                   "actiPy")
import actiPy.preprocessing as prep
import actiPy.actogram_plot as act

input_directory = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/"
                                "01_projects/P2_Circ_Disruption_paper_chapt2/"
                                "01_data_files/01_activity")
save_directory = input_directory.parents[1] / "03_analysis_outputs/01_activity"
subdir_name = '02_short_actograms'

init_kwargs = {
    "input_directory": input_directory,
    "save_directory": save_directory,
    "subdir_name": subdir_name,
    "func": (prep, "read_file_to_df"),
    "index_col": [1, 0],
    "header": [0]
}
fig_object = prep.SaveObjectPipeline(**init_kwargs)

process_kwargs = {
    'function': (prep, "slice_by_label_col"),
    "drop_level": True
}
fig_object.process_file(**process_kwargs)

plot_kwargs = {
    "function": (act, "actogram_plot_all_cols"),
    "data_list": fig_object.processed_list,
    "LDR": -1,
    "set_file_title": True,
    "showfig": False,
    "period": "24H",
    "savefig": True,
    "figsize": (10, 10)
}
fig_object.create_plot(**plot_kwargs)

