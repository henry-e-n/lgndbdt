import json
import os

def get_dsp_files(detector_name, source_location):
    f = open(f"{os.getcwd()}/paths.json")
    data = json.load(f)
    data = data[detector_name][source_location]

    dsp_data_dir = f"{data['path_to_dsp']}{data['detector_name']}/"
    raw_data_dir = f"{data['path_to_raw']}{data['detector_name']}/"
    run_list = data["run_list"]

    raw_files = []
    dsp_files = []

    for run in run_list:
        dsp_file = dsp_data_dir +  run + '.lh5'
        raw_file = raw_data_dir +  run + '.lh5'
        dsp_files.append(dsp_file)
        raw_files.append(raw_file)
    
    file_save_path = data["file_save_path"]
    return dsp_files, raw_files, file_save_path

def get_save_paths(detector_name, source_location):
    f = open(f"{os.getcwd()}/paths.json")
    data = json.load(f)
    data = data[detector_name][source_location]
    
    file_save_path = data["file_save_path"]
    plot_save_path = data["plot_save_path"]

    return file_save_path, plot_save_path