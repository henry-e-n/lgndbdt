# Configuration file
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from tqdm import tqdm
import os
import sys
import json


terminalCMAP = ['#4B9CD3', '#13294B', '#EF426F', '#00A5AD', '#FFD100', '#C4D600']
cmapNormal   = LinearSegmentedColormap.from_list("Custom", ["#151515", '#13294B', '#4B9CD3', "#F4E8DD"], N=50)#, '#C8A2C8'
cmapNormal_r = cmapNormal.reversed("cmapNormal_r")
cmapDiv      = LinearSegmentedColormap.from_list("Custom", ['#13294B', "#F4E8DD", '#4B9CD3'], N=50) #["#EF426F", '#F4E8DD', '#00A5AD'], N=50)


indf = open(f"{os.getcwd()}/ind.json")

jsonData = json.load(indf)
jsonIndex = jsonData["index"]
# print(jsonIndex)

f = open(f"{os.getcwd()}/paths.json")
data = json.load(f)
data = data["runs"][jsonIndex]
detName = data["detector_name"]#"V05612B"
targetPeak = data["target_peak"]
source = data["source"]
savePath = os.getcwd()+f"/{data['detector_name']}"
#f"{data['save_path']}{data['detector_name']}"
plotPath = f"{savePath}/Plots"#os.getcwd()
# modPath = data["mod_path"]

# fname = np.array(data["feature_names"]) # features to use in BDT
# distMatch = np.array(data["distribution_names"])
# distStep = np.array(data["distribution_step"])

dspFile = f"{data['path_to_dsp']}{data['detector_name']}/{data['run_list']}.lh5"
rawFile = f"{data['path_to_raw']}{data['detector_name']}/{data['run_list']}.lh5"

# def __main__():
#     global jsonIndex
    

#     return detName, savePath, plotPath, dspFile, rawFile
"""
Package Dependencies
numpy       =
matplotlib  =  conda install -c conda-forge matplotlib
scipy       =  conda install -c anaconda scipy
lightgbm    =  conda install -c conda-forge lightgbm
shap        =  conda install -c conda-forge shap
tqdm        =  conda install -c conda-forge tqdm
PyQt5       =  pip install PyQt5
imblearn    =  conda install -c conda-forge imbalanced-learn
h5py        =  conda install -c anaconda h5py
eli5        =  conda install -c conda-forge eli5
ipython     =  pip install ipython

Folder and File Dependencies:
lpData : Local Path to Data - within folder must be:
    - DataFiles
        - AnalysisRaw
            - <detName>
                - <peak>
                    - <detName>.lh5
                    - <detName>_Clean.lh5
        - AnalysisOutput
            - <detName>
                - <peak>
modPath : Local Path to MLutils modules '.../MLutils/'
"""
