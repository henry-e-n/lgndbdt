import h5py as h5
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
from extraction_utils.config import detName

def searchFile(name, path):
    try: 
        for root, dirs, files in os.walk(path):
                if name in files:
   #                 print(os.path.join(root, name))
                    return os.path.join(root, name)
    except FileNotFoundError:
        return
        
def checkPath(name, rOTF):
    try:
        fp = searchFile(name, rOTF)
        return fp
    except FileNotFoundError:
        try:
            path = os.path.split(rOTF)[0]
            fp = searchFile(name, rOTF)
            return fp
        except FileNotFoundError:
            print("Sorry, file not found.")
            exit()
    return
        
def dsArray(filename, relativePathToFolder = ""):
    filepath = checkPath(filename, relativePathToFolder)
    wfd = h5.File(f"{filepath}", "r+")
    
    headKeys = list(wfd.keys()) # extracts group names within 'head' - only 'raw' exists
    raw = wfd[f"{headKeys[0]}"] # defines 'raw' variable from raw group

    rawKeys = list(raw.keys()) # extracts sub structures within 'raw' group
    datParam = len(rawKeys) # number of sub structures
    rawDat = np.empty(datParam, dtype = object) # Preallocates data array - size of datParam
    numGroups = 0
    for i in range(datParam): # Arranges substructures within rawDat array 
        rawDat[i] = raw[f"{rawKeys[i]}"]
        try:
            subGroups = list(rawDat[i].keys())
            numGroups += len(subGroups)
        except AttributeError:
            continue
    # By this point - every first level structure is placed in a separate index of rawDat
    #
    # The next chunk of code extracts each of the second level structures to a separate array appRawArr
    appRawArr = np.empty(numGroups, dtype = object)
    newArrIndex = 0
    for i in range(datParam):
        try:
            subGroups = list(rawDat[i].keys())
            print(f"Key: {rawKeys[i]}, has {len(subGroups)} additional subgroups, {subGroups}")
            for n in range(len(subGroups)):
                print(f"Extracting... {rawKeys[i]}/{subGroups[n]}")
                appRawArr[newArrIndex] = raw[f"{rawKeys[i]}/{subGroups[n]}"]
                #rawDat = np.delete(rawDat, i)
                newArrIndex += 1
        except AttributeError:
            continue
    print(f"\nSubdatasets: \n{appRawArr}")
    
    #
    # The next block removes the groups from the first level array 
    rawDat1 = rawDat
    m = 0
    while m <= len(rawDat1):
        try:
            subGroups = list(rawDat1[m].keys())
            rawDat1 = np.delete(rawDat1, m)
        except AttributeError:
            pass
        m += 1
    
    # This adds the second level datasets to the first level array (with groups removed)
    data = np.concatenate([rawDat1, appRawArr], axis=-1)
    
    return wfd, raw, data
"""
def paramExtract(filename, relativePathToFolder = ""):
    filepath = checkPath(filename, relativePathToFolder)
    wfd = h5.File(f"{filepath}", "r+")
    
    headKeys = list(wfd.keys()) # extracts group names within 'head' - only 'raw' exists
    raw = wfd[f"{headKeys[0]}"]
    
    paramArr = [raw["A/E"], raw["dt"], raw["index"], raw["tp_0"], raw["waveform/dt"], raw["waveform/t0"], raw["waveform/values"]]
    return wfd, raw, paramArr
"""

def PEG(filename, relativePathToFolder = "", og = "raw"):
    """
    Function: Extracts parameter arrays from an lh5 file

    Parameters:
        - filename: Name of file to be extracted, without the root
        - relativePathToFolder: Root of path to folder
        - *og = True: Indicates if it is a raw LEGEND file

    Returns:
        - wfd: open file
        - headKeys/raw
        - paramArr
    """
    filepath = checkPath(filename, relativePathToFolder)
    # print(filepath)
    wfd = h5.File(f"{filepath}", "r+")
    
    headKeys = list(wfd.keys()) # extracts group names within 'head' - only 'raw' exists
        
    if og == "raw":
        raw = wfd[f"{headKeys[0]}"]
        paramArr = [raw["A/E"], raw["dt"],
                    raw["index"], raw["tp_0"],
                    raw["waveform/dt"], 
                    raw["waveform/t0"],
                    raw["waveform/values"]]
        return wfd, raw, paramArr
    elif og == "clean":
        raw = wfd[f"{headKeys[0]}"]
        paramArr = [raw["A/E"], raw["dt"],
                    raw["index"], raw["tp_0"],
                    raw["waveform/dt"], 
                    raw["waveform/t0"],
                    raw["waveform/values"],
                    raw["dc_labels"]]
        return wfd, raw, paramArr
    else:
        paramArr = []
        for n in range(len(headKeys)):
            paramArr.append(wfd[f"{headKeys[n]}"])
        return wfd, headKeys, paramArr
    
def appNewh5(ogH5, choosePeak, appArr, appArrN, ts, wfdCorr, module_path):
    file, raw, paramArr = paramExtract(f"{ogH5}", f"{module_path}/DataFiles/AnalysisRaw/V05612B/{choosePeak}/", og = "clean")
    numWave = appArr.shape[1]
    newFile = h5.File(f"{module_path}/DataFiles/AnalysisOutput/{detName}/{choosePeak}/{ogH5[:-4]}_StandardAnalysis.lh5", "w")
    for i in range(len(paramArr)):
        try:
            if paramArr[i].ndim == 2:
                dset = newFile.create_dataset(f"{str(os.path.split(paramArr[i].name)[1])}", [numWave, paramArr[i].shape[1]], dtype = f"{paramArr[i].dtype}", maxshape=(None,None), chunks = True)
                dset[:] = paramArr[i][:numWave, :]
            else:
                dset = newFile.create_dataset(f"{str(os.path.split(paramArr[i].name)[1])}", [numWave], dtype = f"{paramArr[i].dtype}", maxshape=(None), chunks = True)
                dset[:] = paramArr[i][:numWave]
        except ValueError:
            dset = newFile.create_dataset(f"{str(os.path.split(paramArr[i].name)[1])}_1", [numWave], dtype = f"{paramArr[i].dtype}", maxshape=(None), chunks = True)
            dset[:] = paramArr[i][:numWave]
    for n in range(len(appArr)):
        try:
            if appArr[n].ndim == 2:
                dset = newFile.create_dataset(f"{str(appArrN[n])}", appArr[n].shape, dtype = f"{appArr[n].dtype}", maxshape=(None, appArr[n].shape[1]), chunks = True)
                dset[:] = appArr[n]
            else:
                dset = newFile.create_dataset(f"{str(appArrN[n])}", appArr[n].shape, dtype = f"{appArr[n].dtype}", maxshape=(None), chunks = True)
                dset[:] = appArr[n]
        except ValueError:
            dset = newFile.create_dataset(f"{str(appArrN[n])}_1", appArr[n].shape, dtype = f"{appArr[n].dtype}", maxshape=(None))
            dset[:] = appArr[n]
    tdset = newFile.create_dataset("times", ts.shape, dtype = ts.dtype, maxshape = (None, ts.shape[1]), chunks = True)
    tdset[:] = ts[:numWave, :]
    wfddset = newFile.create_dataset("wfdCorr", wfdCorr.shape, dtype = wfdCorr.dtype, maxshape = (None, wfdCorr.shape[1]), chunks = True)
    wfddset[:] = wfdCorr[:numWave, :]

    print(f"Saved as {ogH5[:-4]}_{choosePeak}_StandardAnalysis.lh5")
    return

def addParam(h5File, parameter, paramName):
    modFile = h5.File(f"{h5File}", "r+")
    # if parameter.ndim == 2:
    
    # else:
        # dset = modFile.create_dataset(paramName, len(parameter), dtype = parameter.dtype)
        # dset[:] = parameter[:]
    dset = modFile.create_dataset(paramName, data=parameter, maxshape=(None), chunks = True)
    print(f"Added parameter {paramName}")
    return

def modParam(h5File, parameterArray, paramNames):
    modFile = h5.File(f"{h5File}", "r+")
    for i in range(len(paramNames)):
        set = modFile[paramNames[i]]
        set.resize(parameterArray[i].shape)
        set[...] = parameterArray[i]
        # print(f"Modified {paramNames[i]}, shape: {parameterArray[i].shape}, ")
    return

def pullFiles(detName, datapath):
    filePaths = {"DEP":" ", "FEP": " "}
    try: 
        for root, dirs, files in os.walk(datapath):
                if detName in files:
                    dataSet = os.path.split(root)[1]
                    if dataSet == 'DEP':
                        filePaths["DEP"] = root#os.path.join(root, detName)
                    elif dataSet == "FEP":
                        filePaths["FEP"] = root#os.path.join(root, detName)
                    else:
                        filePaths["DEP"] = "DEP-NOTFOUND"
                        filePaths["FEP"] = "FEP-NOTFOUND"
        return filePaths
    except FileNotFoundError:
        print("FNF")
        return




#################################
# General lh5 code
#################################
import h5py

def openGroup(group, kList):
    for key in group.keys():
        # print(f"{group.name}/{key}")
        kList.append(f"{group.name}/{key}")
        if type(group[key])==h5py._hl.group.Group:
            if len(group[key].keys())>0:
                kList = openGroup(group[key], kList)
    return kList                

def paramExtract(filename, relativePathToFolder, og="raw"):
    filepath = checkPath(filename, relativePathToFolder)
    wfd = h5.File(f"{filepath}", "r")
    keys = []
    keys = openGroup(wfd, keys)

    targetKeys = ["E", "index", "tp_0", "values", "t0", "m/dt", "dc"]
    paramArr = [] # np.empty(len(targetKeys), dtype = object)
    for target in targetKeys:
        cut = np.where(np.char.find(np.array(keys, dtype=str), target)>0)
        # print(wfd[keys[cut[0][0]]][:])
        paramArr.append(wfd[keys[cut[0][0]]])
    
    return wfd, targetKeys, paramArr
    # paramArr = [raw["A/E"], raw["dt"],
    #                 raw["index"], raw["tp_0"],
    #                 raw["waveform/dt"], 
    #                 raw["waveform/t0"],
    #                 raw["waveform/values"]]

