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

def extract_waveforms(fitResults, peakIndex, verbose=False):
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
