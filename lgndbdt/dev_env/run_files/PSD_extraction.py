import numpy as np
from tqdm import tqdm

from extraction_utilities.waveform import cTimes
from utilities.get_files import get_save_paths
from utilities.h5_utils import search_file, save_new_lh5
from extraction_utilities.AvsE import AvsE 
from extraction_utilities.PZ import get_PZ
from extraction_utilities.DCR import find_slope_corr, Normalize_Waveforms
from extraction_utilities.LQ80 import get_LQ80
from extraction_utilities.TDRIFT import get_TDRIFT_interpolate
from utilities.global_config import terminalCMAP

def psd_extraction(paramArr, paramKeys, detector_name, source_location, target_peak):
    file_save_path, plot_save_path = get_save_paths(detector_name, source_location)
    # Make Dictionary for easy referencing of Parameters
    pa = dict(zip(paramKeys, paramArr))
    print(paramKeys)

    cTimes([pa["dt"], pa["t0"], pa["values"]], detector_name, file_save_path, number=-1) # Can remove detname and number in lgndbdt update
    ts = np.load(search_file(f'timesArr_{detector_name}.npy', file_save_path))
    numWave = paramArr[0].shape[0]


    #####################################################################
    ### AvsE
    #####################################################################
    maxA = AvsE(pa["values"], pa["dt"], plots = [], numWF = numWave)


    #####################################################################
    ### DCR - PZ corrected
    #####################################################################
    wfIn, wfCorr, trashPZ     = get_PZ(pa["values"], 0, numWave)
    deltasCorr           = np.zeros(numWave)
    for i in tqdm(range(numWave), 
                  desc   ="Running DCR-P0................", 
                  colour = terminalCMAP[0]):
        deltasCorr[i]    = find_slope_corr(wfIn[i, :], wfCorr[i, :], pa["dt"][i])
    
    np.save(f"{file_save_path}/wfIn_{target_peak}.npy", wfIn)
    np.save(f"{file_save_path}/wfCorr_{target_peak}.npy", wfCorr)


    #####################################################################
    ### LQ80
    #####################################################################
    lqVal, trash                = get_LQ80(ts, wfCorr, trashPZ)
    #####################################################################
    ### Energy - Redundent from Eest Line 131
    #####################################################################

    DAQ_E               = pa["trapEmax"] # Currently uncalibrated
    
    Norm_Vals           = Normalize_Waveforms(pa["values"]) # Currently Using non-PZ corrected
    Norm_A              = AvsE(Norm_Vals, pa["dt"], plots = [], numWF = numWave)

    #####################################################################
    ### Baseline and Noise
    #####################################################################
    noise                = np.zeros(numWave)
    noiseTail            = np.zeros(numWave)

    # if np.any(np.isin(fname, "/NOISE")) or np.any(np.isin(fname, "/NOISETAIL")): 

    #     for i in tqdm(range(0,numWave), 
    #                 desc   = "Calculating Noise.............", 
    #                 colour = terminalCMAP[0]):
    #         window           = blWindow(pa["tp_0"][i], pa["dt"][i])
    #         popt             = blLinFit(window, ts[0], wfIn[i])
    #         if type(window) == None:
    #             noise[i] = 666
    #         else:
    #             noise[i]         = findNoise(linFit, popt, window, ts[0], wfIn[i])

    #         windowTail       = tailWindow(pa["tp_0"][i], pa["dt"][i])
    #         if windowTail[0] + 250 >= len(wfCorr[i])-100:
    #             # print(windowTail)
    #             windowTail[0]    = windowTail[0] + int(np.floor((len(wfCorr[i])-100-windowTail[0])/5))
    #             # print(windowTail)
    #         else:
    #             windowTail[0]    = windowTail[0] + 250
    #         try:
    #             poptTail         = blLinFit(windowTail, ts[0], wfCorr[i])
    #         except:
    #             print(f"Error {i}")
    #             print(f"WindowTail {windowTail}")
    #         noiseTail[i]     = findNoise(linFit, poptTail, windowTail, ts[0], wfCorr[i])
        
    #####################################################################
    ### TDrift
    #####################################################################
    tdrift, tdrift50, tdrift10 = get_TDRIFT_interpolate(ts[0, :], pa["values"][:numWave, :], pa["tp_0"][:numWave], pa["dt"][:numWave])
    
    ### Save Parameters to LH5
    standardAnalysisArray = np.array([pa["dt"], pa["t0"], pa["tp_0"], maxA, deltasCorr, lqVal, noise, noiseTail, tdrift, tdrift50, tdrift10, DAQ_E, Norm_A, maxA/DAQ_E]) # replace energy Arr with Eest
    standardAnalysisArray = np.delete(standardAnalysisArray, trash, axis=1)
    wfCorr = np.delete(wfCorr, trash, axis=0)
    print(f"Trash {len(trash)}, Shape {wfCorr.shape}")
    standardAnalysisNames = np.array(["dt", "t0", "tp_0", "MAXA", "DCR", "LQ80", "NOISE", "NOISETAIL", "TDRIFT", "TDRIFT50", "TDRIFT10", "DAQ_E", "A_NORM", "A_DAQE"])
    save_new_lh5(standardAnalysisArray, standardAnalysisNames, ts, wfCorr, file_save_path, detector_name, target_peak)
    print(f"Final Shape of PSD array, after removing late rise waveforms: {standardAnalysisArray.shape}")
    return