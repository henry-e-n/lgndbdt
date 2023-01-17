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
from pygama import lh5

def getEnergies(verbose=False):
    # Fit Results [ [uncal energies, cal energies], [[1st peak fit param], 2nd peak fit...]] 
    
    dsptargetKeys = ["trapEmax", "tp_0"]
    rawtargetKeys = ["values"]

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
                RAWparamArr = [raw_stack["values"]]
            except TypeError:
                raw_stack = lh5.load_nda(rawFile, rawtargetKeys, "icpcs/icpc1/raw/waveform")
                RAWparamArr = [raw_stack["values"]]
            
            print(DSPparamArr[0].shape[0], RAWparamArr[0].shape[0])
            if DSPparamArr[0].shape[0] != RAWparamArr[0].shape[0]:
                print("Error in File - DSP and RAW size don't match.")
            else:
                energies = DSPparamArr[0][:]

                # peakEnergy = adcE[peakIndex]
                # sigma = peakFits[peakIndex][2]

                # selection_crit =  (energies>(peakEnergy-sigma))*(energies<(peakEnergy+sigma))
                if file == 0:
                    for i in range(len(DSPparamArr)):
                        paramArr[i] = DSPparamArr[i]#[selection_crit]
                    for i in range(len(RAWparamArr)):
                        paramArr[i+len(DSPparamArr)] = RAWparamArr[i]#[selection_crit]
                if file >= 1:
                    for i in range(len(DSPparamArr)):
                        paramArr[i] = np.append(paramArr[i], DSPparamArr[i], axis = 0)
                    for i in range(len(RAWparamArr)):
                        paramArr[i+len(DSPparamArr)] = np.append(paramArr[i+len(DSPparamArr)], RAWparamArr[i], axis = 0)
        except ValueError:
            print(f"Value Error {dspFile}")
    if verbose:
        print(f"Number of features: {len(paramArrKeys)}")
        print(f"Number of Extracted Waveforms (pre-clean): {paramArr[1].shape[0]}")
    return paramArr, paramArrKeys