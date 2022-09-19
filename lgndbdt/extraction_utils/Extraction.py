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


def extraction(paramArr, paramKeys):
    # Make Dictionary for easy referencing of Parameters
    pa = dict(zip(paramKeys, paramArr))
    print(detName)
    # print(pa)
    # Make times array
    if searchFile(f"timesArr_{detName}.npy", savePath) == None:
        print(f"Creating times array, please wait...")
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
    
    #####################################################################
    ### LQ80
    #####################################################################
    lqVal                = getLQ80(ts, wfCorr)
    
    #####################################################################
    ### Energy - Redundent from Eest Line 131
    #####################################################################

    energyArr            = trapENS(ts[:,:], wfCorr[:,:], pa["dt"][:])
    energyArr            = np.amax(energyArr, 1)
    
    #####################################################################
    ### Baseline and Noise
    #####################################################################

    noise                = np.zeros(numWave)
    noiseTail            = np.zeros(numWave)

    for i in tqdm(range(0,numWave), 
                  desc   = "Calculating Noise.............", 
                  colour = terminalCMAP[0]):
        window           = blWindow(pa["tp_0"][i], pa["dt"][i])
        popt             = blLinFit(window, ts[i], wfIn[i])
        noise[i]         = findNoise(linFit, popt, window, ts[i], wfIn[i])

        windowTail       = tailWindow(pa["tp_0"][i], pa["dt"][i])
        windowTail[0]    = windowTail[0] + 250
        poptTail         = blLinFit(windowTail, ts[i], wfCorr[i])
        noiseTail[i]     = findNoise(linFit, poptTail, windowTail, ts[i], wfCorr[i])
    
    #####################################################################
    ### TDrift
    #####################################################################
    tdrift, tdrift50, tdrift10 = getTDriftInterpolate(ts[:numWave, :], pa["values"][:numWave, :], pa["tp_0"][:numWave], pa["dt"][:numWave])
    
    ### Save Parameters to LH5
    standardAnalysisArray = np.array([pa["dt"], pa["t0"], pa["tp_0"], maxA, deltasCorr, lqVal, noise, noiseTail, tdrift, tdrift50, tdrift10, energyArr, maxA/energyArr]) # replace energy Arr with Eest
    standardAnalysisNames = np.array(["dt", "t0", "tp_0", "maxA", "deltasCorrected", "LQ80", "noise", "noiseTail", "tdrift", "tdrift50", "tdrift10", "TrapEnergy", "AvsE_c"])
    appNewh5(standardAnalysisArray, standardAnalysisNames, ts, wfCorr)
    
    return