import numpy as np
import os

from raw_to_calibration      import calibrate_spectrum
from calibration_to_peakdata import extract_waveforms
from PSD_extraction          import psd_extraction
from utilities.get_files     import get_save_paths
from utilities.h5_utils      import search_file
detector_list = ["V01415A"]
source_loc    = ["top"]


for detector in detector_list: # Loops over all specified detectors
    for source in source_loc:  # Loops over source locations (top and side)

        # Defines the save paths and creates appropriate directories as needed
        file_save_path, plot_save_path = get_save_paths(detector, source)
        path_exists = os.path.exists(file_save_path)
        if not path_exists:
            os.makedirs(file_save_path)
            os.makedirs(plot_save_path)
            print(f"{detector} directory was created!")


        # Calibration Pass
        fit_exists = os.path.exists(f"{file_save_path}/fitResults.npy")
        if fit_exists:
            print(f"Using Pre-calibrated Fit found in {file_save_path}/fitResults.npy")
        else:
            search_file(os.getcwd(), "fitResults.npy")
            cal_pars, [fitData, fit_pars]  = calibrate_spectrum(detector_name = detector, source_location = source)
            fitResults = np.array([fitData, fit_pars], dtype=object)
            np.save(f"{file_save_path}/fit_results_{source}.npy", fitResults)
            np.save(f"{file_save_path}/cal_param_{source}.npy", cal_pars)

        # WFD Extraction
        fit_params  = np.load(f"{file_save_path}/fit_results_{source}.npy", allow_pickle=True)
        cal_params  = np.load(f"{file_save_path}/cal_param_{source}.npy", allow_pickle=True)
        
        peak_list = ["DEP", "SEP", "FEP"]
        for target_peak in peak_list:
            wfd, param_keys  = extract_waveforms(detector_name = detector, source_location = source, calibration_parameters = cal_params, fit_parameters = fit_params, target_peak = target_peak)
            np.save(f"{file_save_path}/paramArr_{target_peak}.npy", wfd)
            np.save(f"{file_save_path}/paramKeys_{target_peak}.npy", param_keys)

        for target_peak in peak_list:
            param_arr  = np.load(f"{file_save_path}/paramArr_{target_peak}.npy")
            param_keys = np.load(f"{file_save_path}/paramKeys_{target_peak}.npy")
            psd_extraction(param_arr, param_keys, detector, source, target_peak)
        # END for targ_peak
    # END for source in source_loc
# END for detector in detector_list