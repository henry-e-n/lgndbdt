import numpy as np
import h5py as h5
from extraction_utils.h5Extract import *
from extraction_utils.config import *

def cleanAndSave(paramArr, choosePeak, fpaths):
    [E, dt4, index, tp0, dt8, t0, vals] = paramArr
    nans = []
    for i in range(E.shape[0]):
        if np.isnan(tp0[i]):
            nans.append(1)
        else:
            nans.append(0)   
    # for n in range(len(paramArr)):
    #     paramArr[n] = np.delete(paramArr[n], nans, 0)
    [E, dt4, index, tp0, dt8, t0, vals] = paramArr

    cleanFile = h5.File(f"{fpaths[choosePeak]}\\{detName}_Clean.lh5", "w")

    Eset   = cleanFile.create_dataset("raw/A/E", data=E)
    dt4set = cleanFile.create_dataset("raw/dt", data=dt4)
    indexset = cleanFile.create_dataset("raw/index", data=index)
    tp0set = cleanFile.create_dataset("raw/tp_0", data=tp0)
    dt8set = cleanFile.create_dataset("raw/waveform/dt", data=dt8)
    t0set = cleanFile.create_dataset("raw/waveform/t0", data=t0)
    valsset = cleanFile.create_dataset("raw/waveform/values", data=vals)
    dc_labels = cleanFile.create_dataset("raw/dc_labels", data=nans)


    print(f"Saved Clean Data File as {detName}_{choosePeak}_Clean.lh5, new shape is {E.shape}")
    return paramArr

def runClean(detName, choosePeak):
    filePaths            = pullFiles(f"{detName}.lh5", lpData)
    if len(pullFiles(f"{detName}_Clean.lh5", lpData)[f"{choosePeak}"]) == 1:
        file, raw, paramArr  = paramExtract(f"{detName}.lh5", filePaths[f"{choosePeak}"])
        paramArr             = cleanAndSave(paramArr, choosePeak, filePaths)
    else:
        print(f"Clean Data Already exists at {pullFiles(f'{detName}_Clean.lh5', lpData)[f'{choosePeak}']}")
    return

def postCleanSelect(detName, choosePeak):
    filePaths            = pullFiles(f"{detName}_Clean.lh5", lpData)
    file, raw, paramArr  = paramExtract(f"{detName}_Clean.lh5", filePaths[f"{choosePeak}"], og = "clean")
    dc_labels = paramArr[-1]
    remove = np.where(dc_labels[:] != 0) # 0 indicates physical waveform in dc labels

    for n in range(len(paramArr)):
        paramArr[n] = np.delete(paramArr[n], remove, 0)
    
    [E, dt4, index, tp0, dt8, t0, vals, dcLabels] = paramArr
    nans = []
    for i in range(E.shape[0]):
            if np.isnan(tp0[i]):
                nans.append(i)
    for n in range(len(paramArr)):
        paramArr[n] = np.delete(paramArr[n], nans, 0)
    

    return paramArr