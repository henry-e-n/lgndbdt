import numpy as np
import matplotlib.pyplot as plt
import h5py
import json, os

from utilities.get_files import get_files, get_save_paths
from utilities.h5_utils import openGroup

from pygama import lh5
import pygama.analysis.calibration as pgc
import pygama.analysis.histograms as pgh
import pygama.analysis.peak_fitting as pgp

def clean_dsp(dsp_files):
    delList = []
    dsp_files_icpcs = []
    for i in range(len(dsp_files)):
        file = dsp_files[i]
        try:
            checkFile = h5py.File(file)
            group  = openGroup(checkFile, [])
            if group == ["/icpcs/icpc1/dsp/trapEmax"]:
                dsp_files_icpcs.append(dsp_files[i])
                delList.append(i)
                print("Found one")
        except FileNotFoundError:
            print(f"FNF: {i}, {file}")
            delList.append(i)
            
    dsp_files = np.delete(dsp_files, delList)

    return dsp_files, dsp_files_icpcs

##############################
def calibrate_spectrum(detector_name, source_location, files_for_calibration=6, verbose=False, plots_bool=False):
    f = open(f"{os.getcwd()}/paths.json")
    data = json.load(f)
    data = data[detector_name][source_location]

    dsp_files, raw_files, file_save_path = get_files(data)
    file_save_path, plot_save_path = get_save_paths(detector_name, source_location)
    dsp_files_clean, dsp_files_icpcs = clean_dsp(dsp_files)

    try:
        calibration_files = dsp_files_clean[:files_for_calibration]
        energy_stack = lh5.load_nda(calibration_files, ["trapEmax"], "ORGretina4MWaveformDecoder/dsp")
        energies = energy_stack["trapEmax"]
    except TypeError:
        print("rtc 52: ICPCS")
        calibration_files = dsp_files_icpcs[:files_for_calibration]
        energy_stack = lh5.load_nda(calibration_files, ["trapEmax"], "icpcs/icpc1/dsp")
        energies = energy_stack["trapEmax"]
    
    if verbose:
        print(f"Number of events for Calibration: {len(energies)}")

    #####################################
    # First Calibration Pass
    #####################################
    peaks = np.array([1173.24, # 60Co
                        1332.5, # 60Co,
                        1592.5, # 228Th DEP
                        2103.5, # 228Th SEP
                        2614.53]) # 228Th -> 208Tl (99.8%) 
    # if "DEP" in targetPeak:
    #     print(f"Calibrating on 228Th DEP")
    #     peakIndex = 2
    # elif "SEP" in targetPeak:
    #     print(f"Calibrating on 228Th SEP")
    #     peakIndex = 3
    # elif "FEP" in targetPeak:
    #     print(f"Calibrating on 228Th FEP")
    #     peakIndex = 4
       
    hist, bins, var = pgh.get_hist(energies, bins=1000)
    uncal_peaks, cal_peaks, cal_pars = pgc.hpge_find_E_peaks(hist, bins, var, peaks)

    def match_peaks(data_pks, cal_pks, plotBool=plots_bool):
        """
        Match uncalibrated peaks with literature energy values.
        """
        from itertools import combinations
        from scipy.stats import linregress

        n_pks = len(cal_pks) if len(cal_pks) < len(data_pks) else len(data_pks)

        cal_sets = combinations(range(len(cal_pks)), n_pks)
        data_sets = combinations(range(len(data_pks)), n_pks)

        best_err, best_m, best_b = np.inf, None, None
        for i,cal_set in enumerate(cal_sets):

            cal = cal_pks[list(cal_set)] # lit energies for this set

            for data_set in data_sets:

                data = data_pks[list(data_set)] # uncal energies for this set

                m, b, _, _, _ = linregress(data, y=cal)
                err = np.sum((cal - (m * data + b))**2)

                if err < best_err:
                    best_err, best_m, best_b = err, m, b

        # if plots_bool:
        # # Plots Scatter of Calibration Pass
        #     plt.figure(figsize=(8,5))
        #     plt.scatter(data, cal, label='min.err:{:.2e}'.format(err))
        #     xs = np.linspace(data[0], data[-1], 10)
        #     plt.plot(xs, best_m * xs + best_b , c="r",
        #             label="y = {:.2f} x + {:.2f}".format(best_m,best_b) )
        #     plt.xlabel("Energy (ADC)", fontsize=16)
        #     plt.ylabel("Energy (keV)", fontsize=16)
        #     plt.title("Calibration Fit", fontsize=16)
        #     plt.legend(loc='best', fontsize=14)
        #     plt.savefig(f"{savePath}/CalibrationFit.pdf", dpi=300)
        #     plt.clf()
        #     plt.cla()
        # # plt.show()

        return [best_m, best_b], [cal, data]

    linear_cal, fitDat = match_peaks(uncal_peaks, cal_peaks)

    def linearFit(en, linCal):
        calibratedEnergy = en*linCal[0] + linCal[1]
        return calibratedEnergy
    cal_energies_first = linearFit(energies, linear_cal)

    # Plots Energy Histogram
    if plots_bool:
        plt.figure(figsize=(8,5))
        plt.hist(cal_energies_first, bins=1000, color='#13294B', histtype='step')
        for peak in peaks:
            plt.axvline(peak, 0, 5e5, color='#EF426F', lw=1, alpha=0.75)

        plt.xlabel('Energy (keV)', fontsize=18)
        plt.ylabel('Counts', fontsize=18)
        plt.title("Calibrated Energy Spectrum", fontsize=18)
        plt.yscale('log')
        plt.xlim(0,3000)
        plt.savefig(f"{plot_save_path}/EnergyHist.pdf", dpi=300)
        plt.clf()
        plt.cla()



    #####################################
    # Second Calibration Pass
    #####################################
    pars_list, _ = pgc.hpge_fit_E_peak_tops(hist, bins, var, uncal_peaks, n_to_fit=5)
    modes = np.asarray([pars_list[i][0] for i in range(len(pars_list))])
    sigmas = np.asarray([pars_list[i][1] for i in range(len(pars_list))]) 

    #Plot histograms and fit initial functions
    print(f"Check Sigmas {sigmas}")
    # if "DEP" in targetPeak or "SEP" in targetPeak:
    widths = np.array([sigmas[0]*4, # 515 peak sigma*3
                sigmas[1]*8, #  
                sigmas[2]*4,
                sigmas[3]*15,
                sigmas[4]*20])
    # else:
    #     widths = sigmas * 3
    print(f"Check widths {widths}")

    n_peaks = uncal_peaks.shape[0]
    for i in range(n_peaks):
        #Get histogram for peak within bounds of 5 sigma
        hi, lo = modes[i] + widths[i], modes[i] - widths[i]
        hist, bins, var = pgh.get_hist(energies, bins=100, range=(lo, hi))
        bin_centers = pgh.get_bin_centers(bins)

    #Perform fits for a gaussian plus a step function

    fit_pars, fit_errs = [], []
    func = pgp.gauss_step_pdf #define function for fitting

    def get_parameters(i, modified = False):
        hi, lo = modes[i] + widths[i], modes[i] - widths[i]
        if modified:
            print("Modified Fit")
            widths[i] = 1.5*widths[i]
            hi, lo = modes[i] + widths[i], modes[i] - widths[i]
        hist, bins, var = pgh.get_hist(energies, bins=100, range=(lo, hi))
        par_guesses = pgc.get_hpge_E_peak_par_guess(hist, bins, var, func)
        bounds = pgc.get_hpge_E_bounds(func)
        fixed, mask = pgc.get_hpge_E_fixed(func)
        pars_i, errs_i, cov_i = pgp.fit_binned(func, hist,  bins, var=var, 
                                guess=par_guesses, cost_func='Least Squares', Extended=True, 
                                            fixed=fixed, simplex=True, bounds = bounds)
        pars_i = np.array(pars_i)[mask]
        errs_i = np.array(errs_i)[mask]

        return pars_i, errs_i
    
    for i in range (n_peaks):
        pars_i, errs_i = get_parameters(i)
        fit_pars.append(pars_i)
        fit_errs.append(errs_i)


    sigmas = []
    fig, axs = plt.subplots(n_peaks, 1, figsize=(12,24))
    labels = [r'$^{60}$Co (1173.2 keV)', r'$^{60}$Co (1332.5 keV)', r'$^{228}$Th DEP (1592.5 keV)', r'$^{228}$Th SEP (2103.5 keV)', r'$^{228}$Th FEP (2614.5 keV)']

    for i in range(n_peaks):

        #Get histogram for peak within bounds of 5 sigma
        hi, lo = modes[i] + widths[i], modes[i] - widths[i]
        hist, bins, var = pgh.get_hist(energies, bins=100, range=(lo, hi))
        bin_centers = pgh.get_bin_centers(bins)

        #Compute components of fit function
        fit = func(bin_centers, *fit_pars[i], components=False)
        gaussian, step = func(bin_centers, *fit_pars[i], components=True)
        while np.count_nonzero(gaussian) <= 50:
            print("NOT ENOUGH GAUSSIAN PARAMETERS")
            pars_i, errs_i = get_parameters(i, modified=True)
            gaussian, step = func(bin_centers, *pars_i, components=True)
            fit_pars[i] = pars_i
            fit_errs[i] = errs_i
            # cal_pars, [fitData, fit_pars], peakIndex = energy_calibration(FilesForCalibration+2, verbose, plotBool)
            # return cal_pars, [fitData, fit_pars], peakIndex
        
        sigmas.append(fit_pars[i][2])

        #Plot data and fit components
        if plots_bool:
            axs[i].semilogy(bin_centers, hist, ds="steps-mid", color="k", label=labels[i])
            axs[i].semilogy(bin_centers, fit, color="r", label='fit')
            axs[i].semilogy(bin_centers, gaussian, color="orange", label="gaussian")
            axs[i].semilogy(bin_centers, step, color="cyan", label="step")
            axs[i].set_ylim(hist[-1:]+1, np.amax(hist)+100)
            axs[i].set_ylabel("Counts")
            axs[i].set_xlabel("Uncalibrated Energy (ADC)")
            axs[i].legend(fontsize=18, loc='best')
    axs[0].set_title("Compound Fit to Peaks", fontsize=22)
    plt.savefig(f"{plot_save_path}/fitPeaks.pdf", dpi=300)
    plt.cla()
    plt.clf()

    mus = np.asarray([fit_pars[i][1] for i in range(len(fit_pars))])
    mu_errs = np.asarray([fit_errs[i][1] for i in range(len(fit_pars))]) 

    cal_pars, fitData = match_peaks(mus, cal_peaks)
    print(f"Cal pars {cal_pars}")
    print(f"Fit Data {fitData}")

    if plots_bool:
        for i in range(2, 4):
            sideband_width_ratio = 2
            sideband_LE_left = peaks[i]-(2.5+sideband_width_ratio)*sigmas[i] - sideband_width_ratio*sigmas[i]
            sideband_LE_right = peaks[i]-(2.5+sideband_width_ratio)*sigmas[i] + sideband_width_ratio*sigmas[i]
            sideband_HE_left = peaks[i]+(2.5+sideband_width_ratio)*sigmas[i] - sideband_width_ratio*sigmas[i]
            sideband_HE_right = peaks[i]+(2.5+sideband_width_ratio)*sigmas[i] + sideband_width_ratio*sigmas[i]

            plt.figure(figsize=(6,8))
            plt.hist(cal_energies_first[(cal_energies_first>peaks[i]-200)*(cal_energies_first<peaks[i]+200)], bins=1000, color='k', ec='k')
            plt.axvline(peaks[i], 0, 5e5, color='r', lw=1, alpha=0.75)
            plt.axvline(peaks[i] + sigmas[i], 0, 5e5, color='r', lw=2, alpha=0.75)
            plt.axvline(peaks[i] - sigmas[i], 0, 5e5, color='r', lw=2, alpha=0.75)
            plt.axvline(sideband_LE_left, 0, 5e5, color='b', lw=2, alpha=1)
            plt.axvline(sideband_LE_right, 0, 5e5, color='b', lw=2, alpha=1)
            plt.axvline(sideband_HE_left, 0, 5e5, color='b', lw=2, alpha=1)
            plt.axvline(sideband_HE_right, 0, 5e5, color='b', lw=2, alpha=1)
            
            plt.xlabel('Energy (keV)', fontsize=18)
            plt.ylabel('Counts', fontsize=18)
            plt.yscale('log')
            plt.xlim(peaks[i] - 10*sigmas[i], peaks[i] + 10*sigmas[i])
            if i == 2:
                plt.title("DEP Peak", fontsize=18)
                plt.savefig(f"{plot_save_path}/EnergyHist_DEP.pdf", dpi=300)
            if i == 3:
                plt.title("SEP Peak", fontsize=18)
                plt.savefig(f"{plot_save_path}/EnergyHist_SEP.pdf", dpi=300)
            plt.clf()
            plt.cla()


        # plt.hist(cal_energies_first[(cal_energies_first>peaks[3]-200)*(cal_energies_first<peaks[3]+200)], bins=200, color='k', ec='k')
        # plt.axvline(peaks[3], 0, 5e5, color='r', lw=1, alpha=0.75)
        # plt.axvline(peaks[3] + sigmas[3], 0, 5e5, color='r', lw=2, alpha=0.75)
        # plt.axvline(peaks[3] - sigmas[3], 0, 5e5, color='r', lw=2, alpha=0.75)
        # plt.axvline(peaks[3]-(1.5 + 2)*sigmas[3], 0, 5e5, color='b', lw=2, alpha=0.75)
        # plt.axvline(peaks[3]-(1.5 + 2)*sigmas[3]-2*sigmas[3], 0, 5e5, color='b', lw=2, alpha=0.75)
        # plt.axvline(peaks[3]-(1.5 + 2)*sigmas[3]+2*sigmas[3], 0, 5e5, color='b', lw=2, alpha=0.75)
        
        # plt.xlabel('Energy (keV)', fontsize=24)
        # plt.ylabel('Counts', fontsize=24)
        # plt.yscale('log')
        # plt.xlim(peaks[3] - 7*sigmas[3], peaks[3] + 7*sigmas[3])
        # plt.savefig(f"{savePath}/EnergyHist_SEP.pdf", dpi=300)
        # plt.clf()
        # plt.cla()


    # if verbose:
    #     print(f"Known Energy: {fitData[0][peakIndex]}, Calibrated ADC {fitData[1][peakIndex]}")

    return cal_pars, [fitData, fit_pars] #, peakIndex
