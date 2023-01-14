import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.markers import MarkerStyle
import matplotlib

from tqdm import tqdm
import os
import sys
import time
import h5py as h5

import importlib
import extraction_utils.config
importlib.reload(extraction_utils.config)
from extraction_utils.config import *

from extraction_utils.waveform import *
from extraction_utils.AvsE import *
from extraction_utils.DCR import *
from extraction_utils.BaselineNoise import *
from extraction_utils.DriftTime import *
from extraction_utils.LQ import *
from extraction_utils.h5utils import paramExtract



from extraction_utils.h5utils import *


def extraction(paramArr, paramKeys, plotBool=False):
    # Make Dictionary for easy referencing of Parameters
    pa = dict(zip(paramKeys, paramArr))
    # print(detName)
    # print(pa)
    # Make times array
    # if searchFile(f"timesArr_{detName}.npy", savePath) == None:
    #     print(f"Creating times array, please wait...")
    cTimes([pa["dt"], pa["t0"], pa["values"]], detName, -1) # Can remove detname and number in lgndbdt update
    ts = np.load(searchFile(f'timesArr_{detName}.npy', savePath))
    
    numWave = paramArr[0].shape[0]
    #####################################################################
    ### AvsE
    #####################################################################

    maxA, Eest = AvsE(ts, pa["values"], pa["dt"], plots = [], numWF = numWave)

    
    #####################################################################
    ### DCR - P0 corrected
    #####################################################################
    wfIn, wfCorr     = getP0(pa["values"], 0, numWave)
    deltasCorr           = np.zeros(numWave)
    for i in tqdm(range(numWave), 
                  desc   ="Running DCR-P0................", 
                  colour = terminalCMAP[1]):
        deltasCorr[i]    = findSlopeCorr(wfIn[i, :], wfCorr[i, :], pa["dt"][i])
    
    np.save(f"{savePath}/wfIn_{targetPeak}.npy", wfIn)
    np.save(f"{savePath}/wfCorr_{targetPeak}.npy", wfCorr)
    #####################################################################
    ### LQ80
    #####################################################################
    lqVal                = getLQ80(ts, wfCorr)
    
    #####################################################################
    ### Energy - Redundent from Eest Line 131
    #####################################################################

    energyArr            = trapENS(ts[:,:], wfCorr[:,:], pa["dt"][:])
    energyArr            = np.amax(energyArr, 1)


    np.save(f"{savePath}/energyArr_dsp_{targetPeak}.npy", pa["trapEmax"])
    np.save(f"{savePath}/energyArr_extracted_{targetPeak}.npy", energyArr)
    np.save(f"{savePath}/energyArr_Eest_{targetPeak}.npy", Eest)

    energyArr            = pa["trapEmax"] # Currently uncalibrated

    #####################################################################
    ### Baseline and Noise
    #####################################################################
    noise                = np.zeros(numWave)
    noiseTail            = np.zeros(numWave)

    if np.any(np.isin(fname, "/noise")) or np.any(np.isin(fname, "/noiseTail")): 

        for i in tqdm(range(0,numWave), 
                    desc   = "Calculating Noise.............", 
                    colour = terminalCMAP[0]):
            window           = blWindow(pa["tp_0"][i], pa["dt"][i])
            popt             = blLinFit(window, ts[i], wfIn[i])
            if type(window) == None:
                noise[i] = 666
            else:
                noise[i]         = findNoise(linFit, popt, window, ts[i], wfIn[i])

            windowTail       = tailWindow(pa["tp_0"][i], pa["dt"][i])
            if windowTail[0] + 250 >= len(wfCorr[i])-100:
                # print(windowTail)
                windowTail[0]    = windowTail[0] + int(np.floor((len(wfCorr[i])-100-windowTail[0])/5))
                # print(windowTail)
            else:
                windowTail[0]    = windowTail[0] + 250
            try:
                poptTail         = blLinFit(windowTail, ts[i], wfCorr[i])
            except ValueError:
                print(f"Error {i}")
                print(f"WindowTail {windowTail}")
                # print(np.where(wfCorr[i]<0))
                # print(np.where(np.isnan(wfCorr[i])))

            noiseTail[i]     = findNoise(linFit, poptTail, windowTail, ts[i], wfCorr[i])
        
    #####################################################################
    ### TDrift
    #####################################################################
    tdrift, tdrift50, tdrift10 = getTDriftInterpolate(ts[:numWave, :], pa["values"][:numWave, :], pa["tp_0"][:numWave], pa["dt"][:numWave])
    
    ### Save Parameters to LH5
    standardAnalysisArray = np.array([pa["dt"], pa["t0"], pa["tp_0"], maxA, deltasCorr, lqVal, noise, noiseTail, tdrift, tdrift50, tdrift10, energyArr, maxA/energyArr]) # replace energy Arr with Eest
    standardAnalysisNames = np.array(["dt", "t0", "tp_0", "maxA", "DCR", "LQ80", "noise", "noiseTail", "tdrift", "tdrift50", "tdrift10", "TrapEnergy", "AvsE_c"])
    appNewh5(standardAnalysisArray, standardAnalysisNames, ts, wfCorr)
    
    if plotBool:
        # maxA hist
        plt.hist(maxA, bins = 25)
        plt.xlabel("Current Amplitude")
        plt.ylabel("Number")
        plt.title(f"Current spread")
        plt.savefig(f'{savePath}/_AHist.jpg')
        plt.close()    
        plt.figure()
    return