import h5py
import os
import numpy as np

def openGroup(group, kList):
    for key in group.keys():
        # print(f"{group.name}/{key}")
        kList.append(f"{group.name}/{key}")
        if type(group[key])==h5py._hl.group.Group:
            if len(group[key].keys())>0:
                kList = openGroup(group[key], kList)
    return kList

def search_file(name, path):
    try: 
        for root, dirs, files in os.walk(path):
                if name in files:
                    return os.path.join(root, name)
    except FileNotFoundError:
        return
    
def save_new_lh5(appArr, appArrN, ts, wfdCorr, file_save_path, detector_name, source_location, target_peak):
    numWave = appArr.shape[1]
    newFile = h5py.File(f"{file_save_path}/{detector_name}_PSDs_{target_peak}_{source_location}.lh5", "w")
    
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

    # print(f"Saved as {file_save_path}/{detector_name}_PSDs.lh5")
    return

###################################################################

def extract_h5(filepath, targetKeys):
    wfd = h5py.File(f"{filepath}", "r")
    keys = []
    keys = openGroup(wfd, keys)

    paramArr = [] # np.empty(len(targetKeys), dtype = object)
    for target in targetKeys:
        cut = np.where(np.char.find(np.array(keys, dtype=str), target)>0)
        # print(wfd[keys[cut[0][0]]][:])
        paramArr.append(wfd[keys[cut[0][0]]])
    
    return wfd, targetKeys, paramArr

    

def checkPath(name, rOTF):
    try:
        fp = search_file(name, rOTF)
        return fp
    except FileNotFoundError:
        try:
            path = os.path.split(rOTF)[0]
            fp = search_file(name, rOTF)
            return fp
        except FileNotFoundError:
            print("Sorry, file not found.")
            exit()
    return

def paramExtract(filename, relativePathToFolder = "", og = "raw"):
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
    wfd = h5py.File(f"{filepath}", "r+")
    
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
    