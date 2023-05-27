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
    
def save_new_lh5(appArr, appArrN, ts, wfdCorr, file_save_path, detector_name, target_peak):
    numWave = appArr.shape[1]
    newFile = h5py.File(f"{file_save_path}/{detector_name}_PSDs_{target_peak}.lh5", "w")
    
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

    print(f"Saved as {file_save_path}/{detector_name}_PSDs.lh5")
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
