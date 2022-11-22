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

def getWFDOld(fitResults, peakIndex, verbose=False):
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
    rawtargetKeys = ["t0", "dt", "values"]

    paramArr = np.empty(len(dsptargetKeys)+len(rawtargetKeys), dtype = object)
    paramArrKeys = dsptargetKeys + rawtargetKeys
    
    for file in range(len(dsp_files)):
        dspFile = dsp_files[file]
        rawFile = raw_files[file]

        # wfd, keys, DSPparamArr = paramExtract(dspFile, dsptargetKeys)
        dsp_stack = lh5.load_nda(dspFile, dsptargetKeys, "icpc1/dsp")
        DSPparamArr = [dsp_stack["trapEmax"], dsp_stack["tp_0"]]

        # wfd, keys, RAWparamArr = paramExtract(rawFile, rawtargetKeys)
        raw_stack = lh5.load_nda(rawFile, rawtargetKeys, "icpc1/raw/waveform")
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
        if file == 0:
            for i in range(len(DSPparamArr)):
                paramArr[i] = DSPparamArr[i][selection_crit]
                # paramArrKeys.append(dsptargetKeys[i])# os.path.split(DSPparamArr[i].name)[1])
            for i in range(len(RAWparamArr)):
                paramArr[i+len(DSPparamArr)] = RAWparamArr[i][selection_crit]
                # paramArrKeys.append(rawtargetKeys[i])#os.path.split(RAWparamArr[i].name)[1])
        if file >= 1:
            for i in range(len(DSPparamArr)):
                paramArr[i] = np.append(paramArr[i], DSPparamArr[i][selection_crit], axis = 0)
                # paramArrKeys.append(dsptargetKeys[i])# os.path.split(DSPparamArr[i].name)[1])
            for i in range(len(RAWparamArr)):
                paramArr[i+len(DSPparamArr)] = np.append(paramArr[i+len(DSPparamArr)], RAWparamArr[i][selection_crit], axis = 0)
                # paramArrKeys.append(rawtargetKeys[i])#os.path.split(RAWparamArr[i].name)[1])
        
    if verbose:
        print(f"Number of features: {len(paramArrKeys)}")
        print(f"Number of Extracted Waveforms (pre-clean): {paramArr.shape} {paramArr[1].shape}")
    return paramArr, paramArrKeys

def getWFD(fitResults, peakIndex, verbose=False):
    # Fit Results [ [uncal energies, cal energies], [[1st peak fit param], 2nd peak fit...]] 
    evE = fitResults[0][0]
    adcE = fitResults[0][1]
    peakFits = fitResults[1]
    
    dsptargetKeys = ["trapEmax", "tp_0"]
    rawtargetKeys = ["t0", "dt", "values"]

    paramArr = np.empty(len(dsptargetKeys)+len(rawtargetKeys), dtype = object)
    paramArrKeys = dsptargetKeys + rawtargetKeys
    
    if verbose:
        print(f"Number of files: {len(dsp_files)}")
        
    for file in range(len(dsp_files)):
        dspFile = dsp_files[file]
        rawFile = raw_files[file]
        try: 
            try:
                dsp_stack = lh5.load_nda(dspFile, dsptargetKeys, "icpc1/dsp")
                DSPparamArr = [dsp_stack["trapEmax"], dsp_stack["tp_0"]]
            except TypeError:
                dsp_stack = lh5.load_nda(dspFile, dsptargetKeys, "icpcs/icpc1/dsp")
                DSPparamArr = [dsp_stack["trapEmax"], dsp_stack["tp_0"]]

            try:
                raw_stack = lh5.load_nda(rawFile, rawtargetKeys, "icpc1/raw/waveform")
                RAWparamArr = [raw_stack["t0"], raw_stack["dt"], raw_stack["values"]]
            except TypeError:
                raw_stack = lh5.load_nda(rawFile, rawtargetKeys, "icpcs/icpc1/raw/waveform")
                RAWparamArr = [raw_stack["t0"], raw_stack["dt"], raw_stack["values"]]
                
            if DSPparamArr[0].shape != RAWparamArr[0].shape:
                print("Error in File - DSP and RAW size don't match.")
            else:
                energies = DSPparamArr[0][:]

                peakEnergy = adcE[peakIndex]
                sigma = peakFits[peakIndex][2]

                selection_crit =  (energies>(peakEnergy-sigma))*(energies<(peakEnergy+sigma))
                if file == 0:
                    for i in range(len(DSPparamArr)):
                        paramArr[i] = DSPparamArr[i][selection_crit]
                    for i in range(len(RAWparamArr)):
                        paramArr[i+len(DSPparamArr)] = RAWparamArr[i][selection_crit]
                if file >= 1:
                    for i in range(len(DSPparamArr)):
                        paramArr[i] = np.append(paramArr[i], DSPparamArr[i][selection_crit], axis = 0)
                    for i in range(len(RAWparamArr)):
                        paramArr[i+len(DSPparamArr)] = np.append(paramArr[i+len(DSPparamArr)], RAWparamArr[i][selection_crit], axis = 0)
        except ValueError:
            print(f"Value Error {dspFile}")
    if verbose:
        print(f"Number of features: {len(paramArrKeys)}")
        print(f"Number of Extracted Waveforms (pre-clean): {paramArr[1].shape[0]}")
    return paramArr, paramArrKeys

# if __name__ == "__main__":
#     getWFD([1143.53239803, 2200.70625742, 2499.94000479, 4905.99035165])