import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.markers import MarkerStyle

from tqdm import tqdm
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

def clean_data(paramArr, verbose=False):
    # print(f"Initial Shape: {paramArr[0].shape}")
    nans = []
    for i in range(len(paramArr)):
        for w in range(len(paramArr[i])):
            whereNan = np.where(np.isnan(paramArr[i][w]))
            if len(whereNan[0])>0:
                nans.append(w)
    # print(nans)
    for n in range(len(paramArr)):
        paramArr[n] = np.delete(paramArr[n], nans, 0)
    # print(f"Final Shape: {paramArr[0].shape}")
    if verbose:
        print(f"Number of Waveforms (post-clean) : {paramArr[0].shape}")
    return paramArr

def psd_extraction(paramArr, paramKeys):
    # Make Dictionary for easy referencing of Parameters
    pa = dict(zip(paramKeys, paramArr))
    
    cTimes([pa["dt"], pa["t0"], pa["values"]], number=-1) # Can remove detname and number in lgndbdt update
    ts = np.load(searchFile(f'timesArr_{detName}.npy', savePath))
    numWave = paramArr[0].shape[0]
    #####################################################################
    ### AvsE
    #####################################################################
    maxA = AvsE(pa["values"], pa["dt"], plots = [], numWF = numWave)
    #####################################################################
    ### DCR - P0 corrected
    #####################################################################
    wfIn, wfCorr, trashPZ     = getP0(pa["values"], 0, numWave)
    deltasCorr           = np.zeros(numWave)
    for i in tqdm(range(numWave), 
                  desc   ="Running DCR-P0................", 
                  colour = terminalCMAP[0]):
        deltasCorr[i]    = findSlopeCorr(wfIn[i, :], wfCorr[i, :], pa["dt"][i])
    
    np.save(f"{savePath}/wfIn_{targetPeak}.npy", wfIn)
    np.save(f"{savePath}/wfCorr_{targetPeak}.npy", wfCorr)
    #####################################################################
    ### LQ80
    #####################################################################
    lqVal, trash                = getLQ80(ts, wfCorr, trashPZ)
    #####################################################################
    ### Energy - Redundent from Eest Line 131
    #####################################################################
    TRAP_RES            = trapENS(ts[:,:], wfCorr[:,:], pa["dt"][:])
    TRAP_E              = np.amax(TRAP_RES, 1) 
    DAQ_E               = pa["trapEmax"] # Currently uncalibrated
    
    Norm_Vals           = Normalize_Waveforms(pa["values"]) # Currently Using non-PZ corrected
    Norm_A              = AvsE(Norm_Vals, pa["dt"], plots = [], numWF = numWave)

    # np.save(f"{savePath}/energyArr_dsp_{targetPeak}.npy", pa["trapEmax"])
    # np.save(f"{savePath}/energyArr_extracted_{targetPeak}.npy", energyArr)
    # np.save(f"{savePath}/energyArr_Eest_{targetPeak}.npy", Eest)

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
            popt             = blLinFit(window, ts[0], wfIn[i])
            if type(window) == None:
                noise[i] = 666
            else:
                noise[i]         = findNoise(linFit, popt, window, ts[0], wfIn[i])

            windowTail       = tailWindow(pa["tp_0"][i], pa["dt"][i])
            if windowTail[0] + 250 >= len(wfCorr[i])-100:
                # print(windowTail)
                windowTail[0]    = windowTail[0] + int(np.floor((len(wfCorr[i])-100-windowTail[0])/5))
                # print(windowTail)
            else:
                windowTail[0]    = windowTail[0] + 250
            try:
                poptTail         = blLinFit(windowTail, ts[0], wfCorr[i])
            except:
                print(f"Error {i}")
                print(f"WindowTail {windowTail}")
            noiseTail[i]     = findNoise(linFit, poptTail, windowTail, ts[0], wfCorr[i])
        
    #####################################################################
    ### TDrift
    #####################################################################
    tdrift, tdrift50, tdrift10 = getTDriftInterpolate(ts[0, :], pa["values"][:numWave, :], pa["tp_0"][:numWave], pa["dt"][:numWave])
    
    ### Save Parameters to LH5
    standardAnalysisArray = np.array([pa["dt"], pa["t0"], pa["tp_0"], maxA, deltasCorr, lqVal, noise, noiseTail, tdrift, tdrift50, tdrift10, TRAP_E, DAQ_E, Norm_A, maxA/TRAP_E, maxA/DAQ_E]) # replace energy Arr with Eest
    standardAnalysisArray = np.delete(standardAnalysisArray, trash, axis=1)
    wfCorr = np.delete(wfCorr, trash, axis=0)
    standardAnalysisNames = np.array(["dt", "t0", "tp_0", "MAXA", "DCR", "LQ80", "NOISE", "NOISETAIL", "TDRIFT", "TDRIFT50", "TDRIFT10", "TRAP_E", "DAQ_E", "A_NORM", "A_TRAPE", "A_DAQE"])
    appNewh5(standardAnalysisArray, standardAnalysisNames, ts, wfCorr)
    print(f"Final Shape of PSD array, after removing late rise waveforms: {standardAnalysisArray.shape}")
    return