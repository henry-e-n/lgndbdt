detector_list = ["V01415A"]

for detector in detector_list:
    top_calibration  = calibrate_spectrum(detector_name = detector, source_location = "top")
    side_calibration = calibrate_spectrum(detector_name = detector, source_location = "side")

    top_fit_params  = np.load()
    side_fit_params = np.load()
    top_WFD_DEP  = extract_WFD(detector_name = detector, source_location = "top", fit_parameters = top_fit_params, target_peak = "DEP")
    top_WFD_SEP  = extract_WFD(detector_name = detector, source_location = "top", fit_parameters = top_fit_params, target_peak = "SEP")
    top_WFD_FEP  = extract_WFD(detector_name = detector, source_location = "top", fit_parameters = top_fit_params, target_peak = "FEP")
    side_WFD_DEP = extract_WFD(detector_name = detector, source_location = "side", fit_parameters = top_fit_params, target_peak = "DEP")
    side_WFD_SEP = extract_WFD(detector_name = detector, source_location = "side", fit_parameters = top_fit_params, target_peak = "SEP")
    side_WFD_FEP = extract_WFD(detector_name = detector, source_location = "side", fit_parameters = top_fit_params, target_peak = "FEP")

    top_PSD_DEP  = extract_PSD(detector_name = detector, waveform_array = top_WFD_DEP)
    top_PSD_SEP  = extract_PSD(detector_name = detector, waveform_array = top_WFD_SEP)
    top_PSD_FEP  = extract_PSD(detector_name = detector, waveform_array = top_WFD_FEP)
    side_PSD_DEP = extract_PSD(detector_name = detector, waveform_array = side_WFD_DEP)
    side_PSD_SEP = extract_PSD(detector_name = detector, waveform_array = side_WFD_SEP)
    side_PSD_FEP = extract_PSD(detector_name = detector, waveform_array = side_WFD_FEP)
    