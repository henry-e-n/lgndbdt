import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from tqdm import tqdm
import os
import sys
import json


def get_config():
    # Configuration file

    terminalCMAP = ['#4B9CD3', '#13294B', '#EF426F', '#00A5AD', '#FFD100', '#C4D600']
    cmapNormal   = LinearSegmentedColormap.from_list("Custom", ["#151515", '#13294B', '#4B9CD3', "#F4E8DD"], N=50)#, '#C8A2C8'
    cmapNormal_r = cmapNormal.reversed("cmapNormal_r")
    cmapDiv      = LinearSegmentedColormap.from_list("Custom", ['#13294B', "#F4E8DD", '#4B9CD3'], N=50) #["#EF426F", '#F4E8DD', '#00A5AD'], N=50)
    jsonIndex    = 0

    f            = open(f"{os.getcwd()}/paths.json")
    data         = json.load(f)
    data         = data["runs"][jsonIndex]
    detName      = data["detector_name"]#"V05612B"
    targetPeak   = data["target_peak"]
    source       = data["source"]
    savePath     = data["save_path"]#os.getcwd()+f"/{data['detector_name']}"
    if data["path_to_PSDs"] == "":
        psdPath  = savePath
    else:
        psdPath  = data["path_to_PSDs"]

    plotPath     = f"{savePath}/Plots"#os.getcwd()

    fname = np.array(data["feature_names"]) # features to use in BDT
    fname = np.array(list(map(str.upper, fname)))

    distMatch = np.array(data["distribution_names"])
    distStep  = np.array(data["distribution_step"])

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


    filenames_config = [raw_files, dsp_files]
    BDT_config       = [fname, distMatch, distStep]
    path_config      = [detName, targetPeak, source, savePath, psdPath, plotPath, fname]
    visual_config    = [terminalCMAP, cmapNormal, cmapNormal_r, cmapDiv]
    return filenames_config, BDT_config, path_config, visual_config
