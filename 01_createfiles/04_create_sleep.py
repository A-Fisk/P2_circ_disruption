# script to create actograms and save for all data files
# in input dir

import pathlib
import sys
sys.path.insert(0, "/Users/angusfisk/Documents/01_PhD_files/07_python_package/actiPy")
import actiPy.preprocessing as prep
import actiPy.sleep_process as sleep

input_directory = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/"
                                "01_projects/01_thesisdata/02_circdis/"
                                "01_data_files/01_activity")
save_directory = input_directory.parent
subdir_name = "02_sleep"

init_kwargs = {
    "input_directory": input_directory,
    "save_directory": save_directory,
    "subdir_name": subdir_name,
    "func": (prep, "read_file_to_df"),
    "index_col": [0, 1],
    "header": [0]
}
sleep_object = prep.SaveObjectPipeline(**init_kwargs)

process_kwargs = {
    "test_col": 0,
    "threshold": 1,
    "drop_level": True,
    "ldr_col": -1,
    "function": (sleep, "create_scored_df"),
    "save_suffix": "_sleep.csv",
    "savecsv": True
}
sleep_object.process_file(**process_kwargs)



