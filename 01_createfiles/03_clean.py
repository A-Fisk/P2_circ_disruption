# script to create actograms and save for all data files
# in input dir

import pathlib
import sys
sys.path.insert(0, "/Users/angusfisk/Documents/01_PhD_files/"
                    "07_python_package/actiPy")
import actiPy.preprocessing as prep

input_directory = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/"
                                "01_projects/P2_Circ_Disruption_paper_chapt2/"
                                "01_data_files/01_activity")
save_directory = input_directory
subdir_name = '00_clean'

init_kwargs = {
    "input_directory": input_directory,
    "save_directory": save_directory,
    "subdir_name": subdir_name,
    "func": (prep, "read_file_to_df"),
    "index_col": [1, 0],
    "header": [0]
}
clean_object = prep.SaveObjectPipeline(**init_kwargs)

process_kwargs = {
    "function": (prep, "clean_data"),
    "savecsv": True,
    "append_post": False,
}
clean_object.process_file(**process_kwargs)



