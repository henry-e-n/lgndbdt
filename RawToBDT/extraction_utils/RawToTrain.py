#
# Imports

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# module_path = [os.path.abspath(os.path.join("C:/Users/henac/GitHub/ENAP_Personal/LegendMachineLearning/AnalysisExtraction"))]
# for path in module_path:
#     if path not in sys.path:
#         sys.path.append(path)

from extraction_utils.Extraction import param_Extract
from extraction_utils.h5Extract import *
from extraction_utils.DCR import *

from extraction_utils.config import *
# print(savePath)
# print(modPath)


# detName = "V05612B"
# numWave = 10
# argumentList = [detName, 1, numWave, 1]

def rawToTrain(argumentList, peak):
    param_Extract(argumentList, peak)
    filename                = f"{detName}_Clean_StandardAnalysis.lh5"
    wfd, fname, paramArr    = paramExtract(filename, f"{savePath}{peak}/", False)
    selectArr       = []
    for i in range(len(fname)):
        selectArr.append([fname[i], i])
    selectDict      = dict(selectArr)

    paramArr = DCRquantileCut(paramArr) # Performs a DCR cut and removes NANs
    modParam(f"{savePath}{peak}/{detName}_Clean_StandardAnalysis.lh5", paramArr, fname) # Saves to file
    wfd, fname, paramArr    = paramExtract(filename, f"{savePath}{peak}/", False)
    normDCR  = normalizeDCR(paramArr[selectDict["deltasCorrected"]])
    printDCRHist(normDCR, peak)
    addParam(f"{savePath}{peak}/{detName}_Clean_StandardAnalysis.lh5", normDCR, "DCR")
    return

def getRaw(filename, fpath):
    file, names, paramArr = paramExtract(filename, fpath, False)
    dataDict = []
    dataArr = np.zeros((len(fname), paramArr[0].shape[0]))

    select = []
    counter = 0
    for i in range(len(paramArr)):
        if np.any(np.isin(fname, paramArr[i].name)):
            dataDict.append([paramArr[i].name, paramArr[i][:]])
            dataArr[counter, :] = paramArr[i]
            select.append([paramArr[i].name, counter])
            counter += 1
    dataDictionary = dict(dataDict)
    selectDictionary = dict(select)
    # dataArr  = [] 
    # for key in dataDict.keys():
    #     if np.any(np.isin(fname, key)):
    #         dataArr.append(dataDict[key])
    dataArr = np.stack(dataArr, 1)
    print(f"Returned {fpath}{filename}")#, shape {dataArr.shape}")
    return dataArr, dataDictionary, selectDictionary