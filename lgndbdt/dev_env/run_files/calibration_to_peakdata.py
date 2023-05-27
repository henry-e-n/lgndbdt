import numpy as np

from utilities.get_files import get_dsp_files

from pygama import lh5

def get_peak_index(target_peak):
    if "DEP" in target_peak:
        # print(f"Calibrating on 228Th DEP")
        peakIndex = 2
    elif "SEP" in target_peak:
        # print(f"Calibrating on 228Th SEP")
        peakIndex = 3
    elif "FEP" in target_peak:
        # print(f"Calibrating on 228Th FEP")
        peakIndex = 4
    return peakIndex

def extract_waveforms(detector_name, source_location, calibration_parameters, fit_parameters, target_peak, verbose=False):
    # print(cal_pars)
    # Fit Results [ [uncal energies, cal energies], [[1st peak fit param], 2nd peak fit...]] 
    evE = fit_parameters[0][0]
    adcE = fit_parameters[0][1]
    peakFits = fit_parameters[1]
    
    dsptargetKeys = ["trapEmax", "tp_0"]
    rawtargetKeys = ["t0", "dt", "values"]


    paramArr = np.empty(len(dsptargetKeys)+len(rawtargetKeys), dtype = object)
    paramArrKeys = dsptargetKeys + rawtargetKeys# + ["sidebandNum"]
    
    dsp_files, raw_files, file_save_path = get_dsp_files(detector_name, source_location)

    if verbose:
        print(f"Number of files: {len(dsp_files)}")
        
    for file in range(len(dsp_files)):
        
        dspFile = dsp_files[file]
        rawFile = raw_files[file]
        try: 
            try:
                dsp_stack = lh5.load_nda(dspFile, dsptargetKeys, "ORGretina4MWaveformDecoder/dsp")
                DSPparamArr = [dsp_stack["trapEmax"], dsp_stack["tp_0"]]
            except TypeError:
                dsp_stack = lh5.load_nda(dspFile, dsptargetKeys, "icpcs/icpc1/dsp")
                DSPparamArr = [dsp_stack["trapEmax"], dsp_stack["tp_0"]]

            try:
                raw_stack = lh5.load_nda(rawFile, rawtargetKeys, "ORGretina4MWaveformDecoder/raw/waveform")
                RAWparamArr = [raw_stack["t0"], raw_stack["dt"], raw_stack["values"]]
            except TypeError:
                raw_stack = lh5.load_nda(rawFile, rawtargetKeys, "icpcs/icpc1/raw/waveform")
                RAWparamArr = [raw_stack["t0"], raw_stack["dt"], raw_stack["values"]]
                
            if DSPparamArr[0].shape != RAWparamArr[0].shape:
                print("Error in File - DSP and RAW size don't match.")
            else:
                energies = DSPparamArr[0][:]

                peakIndex = get_peak_index(target_peak)
                peakEnergy = adcE[peakIndex]
                sigmaKEV = peakFits[peakIndex][2]
                sigmaADC = sigmaKEV*calibration_parameters[0]
                sigma = sigmaADC
                # print(f"Selection Window : {(peakEnergy-sigma)} - {(peakEnergy+sigma)}, mean ADC: {peakEnergy}, sigma {sigma}")

                if "_sideband" in target_peak:
                    sideband_width_ratio = 2 # Make sure if you change this to also change it in Visualization ROC, and in rtc plot
                    peakEnergy_left = peakEnergy-(2.5 + sideband_width_ratio)*sigma
                    peakEnergy_right = peakEnergy+(2.5 + sideband_width_ratio)*sigma

                    sigma = sideband_width_ratio*sigma
                    # print("SIDEBAND TIME!!")
                    selection_crit = ((energies>(peakEnergy_left-sigma))*(energies<(peakEnergy_left+sigma)))|((energies>(peakEnergy_right-sigma))*(energies<(peakEnergy_right+sigma)))
                    # print(f"Selection Window : {(peakEnergy_left-sigma)} - {(peakEnergy_left+sigma)}, {(peakEnergy_left-sigma)} - {(peakEnergy_left+sigma)}, mean ADC: {peakEnergy}, sigma {sigma}")

                else:
                    selection_crit =  (energies>(peakEnergy-sigma))*(energies<(peakEnergy+sigma))
                    # print(f"Selection Window : {(peakEnergy-sigma)} - {(peakEnergy+sigma)}, mean ADC: {peakEnergy}, sigma {sigma}")
                # sideband_crit = (energies>(peakEnergy+1.5*sigma))*(energies<(peakEnergy+3.5*sigma))
                if file == 0:
                    for i in range(len(DSPparamArr)):
                        paramArr[i] = DSPparamArr[i][selection_crit]
                    for i in range(len(RAWparamArr)):
                        paramArr[i+len(DSPparamArr)] = RAWparamArr[i][selection_crit]
                    # paramArr[len(RAWparamArr)+len(DSPparamArr)] = int(np.sum(sideband_crit))
                if file >= 1:
                    for i in range(len(DSPparamArr)):
                        paramArr[i] = np.append(paramArr[i], DSPparamArr[i][selection_crit], axis = 0)
                    for i in range(len(RAWparamArr)):
                        paramArr[i+len(DSPparamArr)] = np.append(paramArr[i+len(DSPparamArr)], RAWparamArr[i][selection_crit], axis = 0)
                    # paramArr[len(RAWparamArr)+len(DSPparamArr)] += int(np.sum(sideband_crit))
        except ValueError:
            print(f"Value Error {dspFile}")
    if verbose:
        print(f"Number of features: {len(paramArrKeys)}")
        print(f"Number of Extracted Waveforms (pre-clean): {paramArr[1].shape[0]}")
    return paramArr, paramArrKeys

