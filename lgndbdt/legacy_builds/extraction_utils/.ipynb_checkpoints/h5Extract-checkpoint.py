import h5py as h5
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm

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

def paramExtract(filename, relativePathToFolder = ""):
    filepath = checkPath(filename, relativePathToFolder)
    wfd = h5.File(f"{filepath}", "r+")
    
    headKeys = list(wfd.keys()) # extracts group names within 'head' - only 'raw' exists
    raw = wfd[f"{headKeys[0]}"]
    
    paramArr = [raw["A/E"], raw["dt"], raw["index"], raw["tp_0"], raw["waveform/dt"], raw["waveform/t0"], raw["waveform/values"]]
    return wfd, raw, paramArr

def appNewh5(ogH5, choosePeak, appArr, appArrN, module_path):
    file, raw, paramArr = paramExtract(f"{ogH5}", module_path)
    #[E, dt4, index, tp0, dt8, t0, vals] = paramArr
    #paramArr = np.array([paramArr], dtype=object)
    #print(paramArr)
    #print(" ")
    newFile = h5.File(f"DataFiles/AnalysisOutput/{ogH5[:-4]}_{choosePeak}_StandardAnalysis.lh5", "w")
    #paramArr = [paramArr, appArr]
    #print(paramArr.shape)
    for i in range(len(paramArr)):
        #print(f"{str(os.path.split(paramArr[i].name)[1])} -- {paramArr[i].shape} -- {paramArr[i].dtype}")
        try:
            dset = newFile.create_dataset(f"{str(os.path.split(paramArr[i].name)[1])}", paramArr[i].shape, dtype = f"{paramArr[i].dtype}")
        except ValueError:
            dset = newFile.create_dataset(f"{str(os.path.split(paramArr[i].name)[1])}_1", paramArr[i].shape, dtype = f"{paramArr[i].dtype}")
            
    for n in range(len(appArr)):
        #print(f"{str(appArrN[n])} -- {appArr[n].shape} -- {appArr[n].dtype}")
        try:
            dset = newFile.create_dataset(f"{str(appArrN[n])}", appArr[n].shape, dtype = f"{appArr[n].dtype}")
        except ValueError:
            dset = newFile.create_dataset(f"{str(appArrN[n])}_1", appArr[n].shape, dtype = f"{appArr[n].dtype}")
    print(f"Saved as {ogH5[:-4]}_{choosePeak}_StandardAnalysis.lh5")
    return