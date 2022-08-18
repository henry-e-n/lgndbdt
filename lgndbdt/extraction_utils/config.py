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

# cmapNormal = LinearSegmentedColormap.from_list("Custom", ["#151515", '#13294B', '#2F638F', '#4B9CD3', 'A0C2D8', "#F4E8DD", "#DEC5D3", "#C8A2C8", "#B459B4","#A00FA0"], N=50)#, '#C8A2C8'
# cmapNormal = LinearSegmentedColormap.from_list("Custom", ["#151515","#13294b","#4b9cd3","#f4e8dd","#E4B1CF","#F859D0", "#B30BA2"], 50)
# ["13294b","2f638f","4b9cd3","a0c2d8","f4e8dd","f6cbdf","f7aee0","f991e1","fa74e2","ff00e6"]

f = open(f"{os.getcwd()}/paths.json")
data = json.load(f)

detName = data["detName"]#"V05612B"
lpData = data["lpData"]
modPath = data["modPath"]
# analysisPath = data["analysisPath"]
savePath = f"{lpData}DataFiles/AnalysisOutput/{detName}/"
plotPath = f"{lpData}/Plots"#os.getcwd()

fname = np.array(data["featureNames"]) # features to use in BDT
distMatch = np.array(data["distNames"])
distStep = np.array(data["distStep"])

CorrectionParameters = [2896.5810301207716, 89.33188128281084, 0.01]

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
