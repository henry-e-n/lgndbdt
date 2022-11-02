# 1. Open dsp and Raw File
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import json
import h5py

import importlib
import extraction_utils.config
importlib.reload(extraction_utils.config)
from extraction_utils.config import *

from extraction_utils.h5utils import paramExtract
from pygama import lh5

def getWFD(fitResults, peakIndex, verbose=False):
    # Fit Results [ [uncal energies, cal energies], [[1st peak fit param], 2nd peak fit...]] 
    evE = fitResults[0][0]
    adcE = fitResults[0][1]
    peakFits = fitResults[1]
    
    # f = open(f"{sys.path[0]}/paths.json")
    # configData = json.load(f)

    # detector_name = configData["detector_name"]
    # raw_data_dir = configData["path_to_raw"] + detector_name + "/"
    # dsp_data_dir = configData["path_to_dsp"] + detector_name + "/"
    # source = configData["source"]

    # run = configData["run_list"]

    # raw_files = []
    # dsp_files = []

    # dsp_file = dsp_data_dir +  run + '.lh5'
    # raw_file = raw_data_dir +  run + '.lh5'
    # dsp_files.append(dsp_file)
    # raw_files.append(raw_file)

    dsptargetKeys = ["trapEmax", "tp_0"]
    # wfd, keys, DSPparamArr = paramExtract(dspFile, dsptargetKeys)
    dsp_stack = lh5.load_nda(dsp_files, dsptargetKeys, "icpc1/dsp")
    DSPparamArr = [dsp_stack["trapEmax"], dsp_stack["tp_0"]]

    rawtargetKeys = ["t0", "dt", "values"]
    # wfd, keys, RAWparamArr = paramExtract(rawFile, rawtargetKeys)
    raw_stack = lh5.load_nda(raw_files, rawtargetKeys, "icpc1/raw/waveform")
    RAWparamArr = [raw_stack["t0"], raw_stack["dt"], raw_stack["values"]]

    # Need to translate dictionary to array from load_nda and name DSPparamArr and RAWparamArr
    # Specify location of rawtargetKeys in raw files
    #
    #


    energies = DSPparamArr[0][:]
    
    peakEnergy = adcE[peakIndex]
    sigma = peakFits[peakIndex][2]
    # sigmas = np.array(sigmas)
    # print(peakEnergy, sigma)
    
    selection_crit =  (energies>(peakEnergy-sigma))*(energies<(peakEnergy+sigma))
    
    # RAWparamArr = np.array(RAWparamArr, dtype=object)
    paramArr = np.empty(len(DSPparamArr)+len(RAWparamArr), dtype = object)
    paramArrKeys = []
    for i in range(len(DSPparamArr)):
        paramArr[i] = np.array(DSPparamArr[i][selection_crit])
        paramArrKeys.append(dsptargetKeys[i])# os.path.split(DSPparamArr[i].name)[1])
    for i in range(len(RAWparamArr)):
        paramArr[i+len(DSPparamArr)] = np.array(RAWparamArr[i][selection_crit])
        paramArrKeys.append(rawtargetKeys[i])#os.path.split(RAWparamArr[i].name)[1])
        
    if verbose:
        print(f"Number of features: {paramArr[0].shape}")
        print(f"Number of Extracted Waveforms (pre-clean): {paramArr[1].shape}")
    return paramArr, paramArrKeys

# if __name__ == "__main__":
#     getWFD([1143.53239803, 2200.70625742, 2499.94000479, 4905.99035165])