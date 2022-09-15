# 1. Open dsp and Raw File
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import json
import h5py

from extraction_utils.h5utils import paramExtract


def getWFD(fitResults, peakIndex):
    # Fit Results [ [uncal energies, cal energies], [[1st peak fit param], 2nd peak fit...]] 
    evE = fitResults[0][0]
    adcE = fitResults[0][1]
    peakFits = fitResults[1]
    
    f = open(f"{sys.path[0]}/paths.json")
    configData = json.load(f)

    detector_name = configData["detector_name"]
    raw_data_dir = configData["path_to_raw"] + detector_name + "/"
    dsp_data_dir = configData["path_to_dsp"] + detector_name + "/"
    source = configData["source"]

    run = configData["run_list"]

    raw_files = []
    dsp_files = []

    dsp_file = dsp_data_dir +  run + '.lh5'
    raw_file = raw_data_dir +  run + '.lh5'
    dsp_files.append(dsp_file)
    raw_files.append(raw_file)

    dsptargetKeys = ["trapEmax", "tp_0"]
    wfd, keys, DSPparamArr = paramExtract(dsp_files[0], dsptargetKeys)

    rawtargetKeys = ["t0", "dt", "values"]
    wfd, keys, RAWparamArr = paramExtract(raw_files[0], rawtargetKeys)


    energies = DSPparamArr[0][:]
    
    peakEnergy = adcE[peakIndex]
    sigma = peakFits[peakIndex][2]
    # sigmas = np.array(sigmas)
    print(peakEnergy, sigma)
    
    selection_crit =  (energies>(peakEnergy-sigma))*(energies<(peakEnergy+sigma))
    
    # RAWparamArr = np.array(RAWparamArr, dtype=object)
    paramArr = np.empty(len(DSPparamArr)+len(RAWparamArr), dtype = object)
    paramArrKeys = []
    for i in range(len(DSPparamArr)):
        paramArr[i] = np.array(DSPparamArr[i][selection_crit])
        paramArrKeys.append(os.path.split(DSPparamArr[i].name)[1])
    for i in range(len(RAWparamArr)):
        paramArr[i+len(DSPparamArr)] = np.array(RAWparamArr[i][selection_crit])
        paramArrKeys.append(os.path.split(RAWparamArr[i].name)[1])
        
    print(paramArr.shape, paramArr[0].shape, paramArr[-1].shape)

    return paramArr, paramArrKeys

if __name__ == "__main__":
    getWFD([1143.53239803, 2200.70625742, 2499.94000479, 4905.99035165])