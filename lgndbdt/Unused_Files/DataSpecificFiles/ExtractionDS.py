from matplotlib.markers import MarkerStyle
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
import os
import sys
import time
import h5py as h5

# cwd = os.getcwd()
# module_path = os.path.abspath(os.path.split(cwd)[0])
# sys.path.append(module_path)

from extraction_utils.h5Extract import *
from extraction_utils.waveform import *
from extraction_utils.AvsE import *
from extraction_utils.DCR import *
from extraction_utils.BaselineNoise import *
from extraction_utils.DriftTime import *
from extraction_utils.LQ import *
from extraction_utils.CleanData import *
from extraction_utils.config import *

import argparse                  # allows us to deal with arguments to main()
from argparse import RawTextHelpFormatter

print("Finished Import")

# >>>>>>>>>>>>>>>>>>>>>
# > module load python
# > activate MLenv
# > cd LegendMachineLearning/
# > python RunAnalysis.py
# >>>>>>>>>>>>>>>>>>>>>
def cleanData(keys, paramArr):
    pa = dict(zip(keys, paramArr))
    nans = []
    for i in range(pa["m/dt"].shape[0]):
        if np.isnan(pa["tp_0"][i]):
            nans.append(i)   
    for n in range(len(paramArr)):
        paramArr[n] = np.delete(paramArr[n], nans, 0)
    print(f"Removed NANs, new shape is {paramArr[n].shape}")
    return paramArr

def pullFiles(detName, datapath):
    filePaths = {"DEP":" ", "FEP": " "}
    try: 
        for root, dirs, files in os.walk(datapath):
                if detName in files:
                    dataSet = os.path.split(root)[1]
                    if dataSet == 'DEP':
                        filePaths["DEP"] = root#os.path.join(root, detName)
                    elif dataSet == "FEP":
                        filePaths["FEP"] = root#os.path.join(root, detName)
                    else:
                        filePaths["DEP"] = "DEP-NOTFOUND"
                        filePaths["FEP"] = "FEP-NOTFOUND"
        return filePaths
    except FileNotFoundError:
        print("FNF")
        return

def getArguments():
    #####################################################################
    ### ArgParse
    #####################################################################

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter, 
                                     description = "If the data is already clean, set clean to false and fname to the clean data name")
    parser.add_argument("detName",type=str,
                        help="Name of Raw Data File",
                        default = "V05612B", nargs='?')
    parser.add_argument("sf",type=int,
                        help="Save Intermediate Files (True ( 1 ) or False ( 0 ))",
                        default=0, nargs='?')
    
    # parser.add_argument("clean",type=int,
    #                     help="Clean Data - remove nans(True ( 1 ) or False ( 0 ))",
    #                     default=0, nargs='?')
    parser.add_argument("Waves",type=int,
                        help="Number of Waves to print",
                        default=100, nargs='?')
    parser.add_argument("P0",type=int,
                        help="Reuse P0 parameters (0 for Redo P0 fitting) (1 for using default fit)",
                        default=1, nargs='?')
    
    args                 = parser.parse_args()
    
    return args

def param_Extract(argumentList, choosePeak = "DEP"):
    print(f"\n                                    |-------------------------------- Beginning Main {choosePeak} --------------------------------|")
    
    [detName, saveFiles, numWave, p0Param] = argumentList

    #####################################################################
    ### Data Import and Extraction
    #####################################################################
    # try:
    #     filePaths            = pullFiles(f"{detName}_Clean.lh5", lpData)
    #     print(filePaths)
    #     # file, raw, paramArr  = paramExtract(f"{detName}_Clean.lh5", filePaths[f"{choosePeak}"], "clean")
    #     paramArr             = postCleanSelect(detName, choosePeak)
    # except FileNotFoundError:

    #     runClean(detName, choosePeak)
    #     filePaths            = pullFiles(f"{detName}_Clean.lh5", lpData)
    #     file, raw, paramArr  = paramExtract(f"{detName}_Clean.lh5", filePaths[f"{choosePeak}"])
    filePaths            = pullFiles(f"{detName}_Clean.lh5", lpData)
    file, keys, paramArr  = paramExtract(f"{detName}_Clean.lh5", filePaths[f"{choosePeak}"])

    #####################################################################
    ### Clean Data if Specified
    #####################################################################

    # if cData:
    paramArr         = cleanData(keys, paramArr)
    if numWave == -1:
        print(paramArr[0].shape)
        numWave = paramArr[0].shape[0]
    
    # [E, dt4, index, pa["tp_0"], pa["m/dt"], pa["t0"], pa["values"], dc_labels] = paramArr
    pa = dict(zip(keys, paramArr))

    #####################################################################
    ### Times Array
    #####################################################################
    if searchFile(f"timesArr_{choosePeak}.npy", lpData) == None:
        print(f"Creating times array, please wait...")
        cTimes([pa["m/dt"], pa["t0"], pa["values"]], choosePeak, numWave)
    ts = np.load(searchFile(f'timesArr_{choosePeak}.npy', lpData))
    ts = ts[:numWave, :]
    print(f"{choosePeak} file extracted - Post Clean Shape: {ts.shape}")
    
    #####################################################################
    ### AvsE
    #####################################################################

    maxA, Eest = AvsE(ts, pa["values"], pa["m/dt"], plots = [], numWF = numWave)

    # if saveFiles == True:
        # np.save(f"DataFiles/AnalysisIntermediate/{choosePeak}/maxA.npy", maxA)

    #####################################################################
    ### DCR - no P0 correction
    #####################################################################
    ### DCR - no P0 correction
    # deltas = np.zeros(numWave)
    # for i in tqdm(range(numWave), 
    #               desc   = "Running DCR...................",
    #               ascii  = '░▒█', 
    #               colour = terminalCMAP[1]):
    #     deltas[i]        = findSlope(pa["values"][i, :], pa["m/dt"][i])
    # if saveFiles == True:
        # np.save(f"{lpData}/DataFiles/AnalysisIntermediate/{choosePeak}/DCRSlopesUncorrected.npy", deltas)    
    


    #####################################################################
    ### DCR - P0 corrected
    #####################################################################

    # if numWave >= 10:
    #     dp0Num           = 10
    # else:
    #     dp0Num           = numWave
    
    if p0Param == 0:
        wfIn, wfCorr     = getP0(pa["values"], 0, numWave)
    elif p0Param == 1:
        popt             = tuple([2896.5810301207716, 89.33188128281084, 0.01])
        wfIn, wfCorr     = getP0(pa["values"], popt, numWave)
    # print(f"P0 fitted Parameters..........:     {popt}")

    deltasCorr           = np.zeros(numWave)
    for i in tqdm(range(numWave), 
                  desc   ="Running DCR-P0................", 
                  colour = terminalCMAP[1]):
        deltasCorr[i]    = findSlopeCorr(wfIn[i, :], wfCorr[i, :], pa["m/dt"][i])
    # if saveFiles == True:
        # np.save(f"{lpData}/DataFiles/AnalysisIntermediate/{choosePeak}/DCRSlopesCorrected.npy", deltasCorr)

    #####################################################################
    ### LQ80
    #####################################################################
    # wfCorr is good here
    lqVal                = getLQ80(ts, wfCorr)

    #####################################################################
    ### Energy - Redundent from Eest Line 131
    #####################################################################

    energyArr            = trapENS(ts[:,:], wfCorr[:,:], pa["m/dt"][:])
    energyArr            = np.amax(energyArr, 1)
    if saveFiles == True:
        np.save(f"{lpData}/DataFiles/AnalysisIntermediate/{choosePeak}/EnergyEstimation{energyArr.shape[0]}.npy", energyArr)
    #####################################################################
    ### Baseline and Noise
    #####################################################################

    noise                = np.zeros(numWave)
    noiseTail            = np.zeros(numWave)

    for i in tqdm(range(0,numWave), 
                  desc   = "Calculating Noise.............", 
                  colour = terminalCMAP[0]):
        window           = blWindow(pa["tp_0"][i], pa["m/dt"][i])
        popt             = blLinFit(window, ts[i], wfIn[i])
        noise[i]         = findNoise(linFit, popt, window, ts[i], wfIn[i])

        windowTail       = tailWindow(pa["tp_0"][i], pa["m/dt"][i])
        windowTail[0]    = windowTail[0] + 250
        poptTail         = blLinFit(windowTail, ts[i], wfCorr[i])
        noiseTail[i]     = findNoise(linFit, poptTail, windowTail, ts[i], wfCorr[i])
        # visBL2(windowTail, pa["tp_0"][i], linFit, poptTail, pa["m/dt"][i], ts[i], wfCorr[i], i/numWave)

    # if saveFiles == True:
        # np.save(f"{lpData}/DataFiles/AnalysisIntermediate/{choosePeak}/NoiseEst.npy", noise)
 
    #####################################################################
    ### TDrift
    #####################################################################

    # tdrift, tdrift50, tdrift10 = getTDrift(pa["values"][:numWave, :], pa["tp_0"][:numWave], pa["m/dt"][:numWave])
    tdrift, tdrift50, tdrift10 = getTDriftInterpolate(ts[:numWave, :], pa["values"][:numWave, :], pa["tp_0"][:numWave], pa["m/dt"][:numWave])

    # meanTDrift           = np.mean(tdrift)
    # meanTDrift50         = np.mean(tdrift50)
    # meanTDrift10         = np.mean(tdrift10)

    # print(f"Tdrift:            Mean tdrift {round(meanTDrift,2)}, tdrift50 {round(meanTDrift50, 2)}, tdrift10 {round(meanTDrift10,2)}")
    # deltas and deltasCorr removed
    standardAnalysisArray = np.array([maxA, deltasCorr, lqVal, noise, noiseTail, tdrift, tdrift50, tdrift10, energyArr, maxA/energyArr]) # replace energy Arr with Eest
    standardAnalysisNames = np.array(["maxA", "deltasCorrected", "LQ80", "noise", "noiseTail", "tdrift", "tdrift50", "tdrift10", "TrapEnergy", "AvsE_c"])
    if saveFiles == True:
        appNewh5(f"{detName}_Clean.lh5", choosePeak, standardAnalysisArray, standardAnalysisNames, ts, wfCorr, lpData)
        printAllPlots(choosePeak, pa, ts, standardAnalysisArray, wfIn, wfCorr)
        
    return standardAnalysisArray

def printAllPlots(peakName, pa, times, analysisArr, wfIn, wfCorr):
    print("Exporting all Figures to AnalysisImages directory...")

    [maxA, deltasCorr, lqVal, noise, noiseTail, tdrift, tdrift50, tdrift10, energyArr, AvsE] = analysisArr
    # Waveform
    plt.figure()
    # plt.close()

    plt.plot(times[0], pa["values"][0,:], color=terminalCMAP[1])
    plt.xlabel("Time")
    plt.title(f"Sample Waveform")
    # plt.show()
    plt.savefig(f'{plotPath}/{peakName}_Waveform.jpg')
    
    plt.close()
    plt.figure()

    # P0
    plt.plot(times[0], wfIn[0], color=terminalCMAP[0])
    plt.plot(times[0], wfCorr[0], color=terminalCMAP[1])
    plt.xlabel("Time")
    plt.title(f"Waveform With Pole Zero Correction")
    # plt.show()
    plt.savefig(f'{plotPath}/{peakName}_P0.jpg')

    plt.close()    
    plt.figure()

    # DCR
    visualizeDCR(times[1], wfCorr[1, :], pa["m/dt"][1], find97(wfIn[1]))
    plt.xlabel("Time")
    plt.title(f"Waveform with DCR visualization")
    # plt.show()
    plt.savefig(f'{plotPath}/{peakName}_DCRCorr.jpg')
    plt.close()    
    plt.figure()

    plt.hist(deltasCorr, bins = 25)
    plt.xlabel("DCR Slope")
    plt.ylabel("Number")
    plt.title(f"DCR spread")
    plt.savefig(f'{plotPath}/{peakName}_DCRHist.jpg')
    plt.close()    
    plt.figure()

    # AvsE
    plt.plot(pa["E"][:len(maxA)], maxA,'.', color = terminalCMAP[1])
    plt.xlabel("Energy (MeV)")
    plt.ylabel("Max Current (A)")
    plt.title(f"A vs E")
    # plt.show()
    plt.savefig(f'{plotPath}/{peakName}_AvsE.jpg')
    plt.close()    
    plt.figure()

    plt.hist(maxA, bins = 25)
    plt.xlabel("Current Amplitude")
    plt.ylabel("Number")
    plt.title(f"Current spread")
    plt.savefig(f'{plotPath}/{peakName}_AHist.jpg')
    plt.close()    
    plt.figure()

    # LQ
    LQvis(times[0], wfCorr[0])
    plt.savefig(f'{plotPath}/{peakName}_LQ80.jpg')
    plt.close()    
    plt.figure()

    plt.figure(figsize=(4,8))
    LQvisZoom(times[0], wfCorr[0])
    plt.xlabel("Time (ns)")
    plt.ylabel("ADC")
    plt.axis("off")
    plt.savefig(f'{plotPath}/{peakName}_LQ80Zoom.jpg')
    plt.close()    
    plt.figure()

    plt.plot(energyArr, lqVal,'.')
    plt.xlabel("Peak Energy")
    plt.ylabel("LQ80 Value")
    plt.savefig(f"{plotPath}/{peakName}_LQvsE.jpg")
    plt.close()    
    plt.figure()

    # Tdrift
    plt.plot(times[0], pa["values"][0], color = terminalCMAP[0])
    tdriftInd   = int(pa["tp_0"][0]/pa["m/dt"][0] + tdrift[0]/pa["m/dt"][0])
    tdriftInd50 = int(pa["tp_0"][0]/pa["m/dt"][0] + tdrift50[0]/pa["m/dt"][0])
    tdriftInd10 = int(pa["tp_0"][0]/pa["m/dt"][0] + tdrift10[0]/pa["m/dt"][0])
    plt.plot(times[0, tdriftInd], pa["values"][0, tdriftInd], '.', color = terminalCMAP[2])
    plt.plot(times[0, tdriftInd50], pa["values"][0, tdriftInd50], '.', color = terminalCMAP[2])
    plt.plot(times[0, tdriftInd10], pa["values"][0, tdriftInd10], '.', color = terminalCMAP[2])
    plt.close()    
    plt.figure()


    # DCR
    visualizeDCR(times[0], wfCorr[0, :], pa["m/dt"][0], find97(wfIn[0]))

    plt.scatter(times[0, tdriftInd], wfCorr[0, tdriftInd], marker="s", color = terminalCMAP[0], s=15, linewidths = 10, label="Drift Time")
    plt.scatter(times[0, tdriftInd50], wfCorr[0, tdriftInd50], marker="o", color = terminalCMAP[0], s=15,  linewidths = 10, label="50% Drift Time")
    plt.scatter(times[0, tdriftInd10], wfCorr[0, tdriftInd10], marker=".", color = terminalCMAP[0], s=15,  linewidths = 12, label="10% Drift Time")
    plt.legend()
    plt.xlabel("Time (ns)")
    plt.title(f"Drift Time and DCR visualization")
    # plt.show()
    plt.savefig(f"{plotPath}/{peakName}_TDrift_DCR.jpg")

    plt.figure(figsize=(4,8))
    plt.scatter(times[0, tdriftInd], wfCorr[0, tdriftInd], marker="s", color = terminalCMAP[0], s=1300, linewidths=5, label="Drift Time")
    LQvisZoom(times[0], wfCorr[0])
    plt.axis("off")
    plt.savefig(f'{plotPath}/{peakName}_LQ80Zoom_TDDCR.jpg')
    plt.close()    
    plt.figure()

    plt.close()    
    plt.figure()

    # Energy Hist
    plt.hist(energyArr, bins = 25)
    plt.xlabel("Energy")
    plt.ylabel("Counts")
    plt.title("Energy Histogram")
    plt.savefig(f"{plotPath}/{peakName}_EnergyHist.jpg")
    plt.close()
    plt.figure()

    return

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def runMain():
    pyStart = time.time()

    args = getArguments()
    detName              = args.detName
    saveFiles            = args.sf
    numWave              = args.Waves
    p0Param              = args.P0

    argumentList = [detName, saveFiles, numWave, p0Param]

    print(f"RUNNING RunAnalysisM2.py -- Detector Name: {detName}, Save Files? {bool(saveFiles)}, Number of Waves: {numWave}, Reuse P0 fit? {bool(p0Param)}") 


    DEPSAA = param_Extract(argumentList, "DEP")
    FEPSAA = param_Extract(argumentList, "FEP")

    plt.plot(DEPSAA[-1], DEPSAA[3],'.', color=terminalCMAP[0])
    plt.plot(FEPSAA[-1], DEPSAA[3],'.', color=terminalCMAP[1])
    plt.xlabel("Peak Energy")
    plt.ylabel("LQ80 Value")
    plt.savefig(f"AnalysisImages/Joint_LQvsE.jpg")
    plt.figure()

    totalTime = time.time() - pyStart
    print(f"\n Total run time:  {totalTime/60} minutes ")
    return
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>