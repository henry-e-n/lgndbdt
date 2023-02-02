import json, os, sys, h5py
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import pygama.analysis.calibration as pgc
import pygama.analysis.histograms as pgh
import pygama.analysis.peak_fitting as pgp
from pygama import lh5
from pygama.dsp.dsp_optimize import *
from pygama.dsp.WaveformBrowser import WaveformBrowser as wfb

import importlib
import extraction_utils.config
importlib.reload(extraction_utils.config)
from extraction_utils.config import *

from extraction_utils.h5utils import paramExtract
from extraction_utils.h5Extract import openGroup

def clean_dsp(dsp_files):
    delList = []
    dsp_files_icpcs = []
    for i in range(len(dsp_files)):
        file = dsp_files[i]
        try:
            checkFile = h5py.File(file)
            group  = openGroup(checkFile, [])
            # print(group)
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
def energy_calibration(verbose=False, plotBool=False):
    dsp_files_clean, dsp_files_icpcs = clean_dsp(dsp_files)

    try:
        calibration_files = dsp_files_clean[:2]
        energy_stack = lh5.load_nda(calibration_files, ["trapEmax"], "icpc1/dsp")
        energies = energy_stack["trapEmax"]
    except TypeError:
        print("rtc 52: ICPCS")
        calibration_files = dsp_files_icpcs[:2]
        energy_stack = lh5.load_nda(calibration_files, ["trapEmax"], "icpcs/icpc1/dsp")
        energies = energy_stack["trapEmax"]
    
    if verbose:
        print(f"Number of events for Calibration: {len(energies)}")

    #####################################
    # First Calibration Pass
    #####################################
    if source == "60Co": # 583.187, # 228Th -> 208Tl (85%)
        peaks = np.array([1173.24, # 60Co
                          1332.5, # 60Co,
                          1592.5, # 228Th DEP
                          2103.5, # 228Th SEP
                          2614.53]) # 228Th -> 208Tl (99.8%) 
        if "228ThDEP" in targetPeak:
            print(f"Calibrating on 228Th DEP")
            peakIndex = 2
        elif "228ThSEP" in targetPeak:
            print(f"Calibrating on 228Th SEP")
            peakIndex = 3
        elif "228ThFEP" in targetPeak:
            print(f"Calibrating on 228Th FEP")
            peakIndex = 4
        
    else:
        peaks = np.array([583.187, # 228Th -> 208Tl (85%)
                          2614.53]) # 228Th -> 208Tl (99.8%) 

    hist, bins, var = pgh.get_hist(energies, bins=1000)
    uncal_peaks, cal_peaks, cal_pars = pgc.hpge_find_E_peaks(hist, bins, var, peaks)

    def match_peaks(data_pks, cal_pks, plotBool=plotBool):
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

        if plotBool:
        # Plots Scatter of Calibration Pass
            plt.scatter(data, cal, label='min.err:{:.2e}'.format(err))
            xs = np.linspace(data[0], data[-1], 10)
            plt.plot(xs, best_m * xs + best_b , c="r",
                    label="y = {:.2f} x + {:.2f}".format(best_m,best_b) )
            plt.xlabel("Energy (ADC)", fontsize=24)
            plt.ylabel("Energy (keV)", fontsize=24)
            plt.legend(loc='best', fontsize=20)
            plt.savefig(f"{savePath}/CalibrationFit.jpg")
            plt.clf()
            plt.cla()
        # plt.show()

        return [best_m, best_b], [cal, data]

    linear_cal, fitDat = match_peaks(uncal_peaks, cal_peaks)

    def linearFit(en, linCal):
        calibratedEnergy = en*linCal[0] + linCal[1]
        return calibratedEnergy
    cal_energies_first = linearFit(energies, linear_cal)

    # Plots Energy Histogram
    if plotBool:
        plt.hist(cal_energies_first, bins=1000, color='k', ec='k')
        for peak in peaks:
            plt.axvline(peak, 0, 5e5, color='r', lw=1, alpha=0.75)

        plt.xlabel('Energy (keV)', fontsize=24)
        plt.ylabel('Counts', fontsize=24)
        plt.yscale('log')
        plt.xlim(0,3000)
        plt.savefig(f"{savePath}/EnergyHist.jpg")
        plt.clf()
        plt.cla()



    #####################################
    # Second Calibration Pass
    #####################################
    pars_list, _ = pgc.hpge_fit_E_peak_tops(hist, bins, var, uncal_peaks, n_to_fit=5)
    modes = np.asarray([pars_list[i][0] for i in range(len(pars_list))])
    sigmas = np.asarray([pars_list[i][1] for i in range(len(pars_list))]) 

    #Plot histograms and fit initial functions

    if np.isin("228ThDEP", targetPeak) or np.isin("228ThSEP", targetPeak):
        widths = np.array([sigmas[0]*4, # 515 peak sigma*3
                   sigmas[1]*6, #  
                   sigmas[2]*5,
                   sigmas[3]*4,
                   sigmas[4]*8])
    else:
        widths = sigmas * 3

    n_peaks = uncal_peaks.shape[0]
    for i in range(n_peaks):
        #Get histogram for peak within bounds of 5 sigma
        hi, lo = modes[i] + widths[i], modes[i] - widths[i]
        hist, bins, var = pgh.get_hist(energies, bins=100, range=(lo, hi))
        bin_centers = pgh.get_bin_centers(bins)

    #Perform fits for a gaussian plus a step function

    fit_pars, fit_errs = [], []
    func = pgp.gauss_step_pdf #define function for fitting

    for i in range (n_peaks):

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

        fit_pars.append(pars_i)
        fit_errs.append(errs_i)

    sigmas = []
    fig, axs = plt.subplots(n_peaks, 1, figsize=(12,24))
    labels = [r'$^{228}$Th -> Tl', r'$^{60}$Co', r'$^{60}$Co', r'$^{228}$Th DEP', r'$^{228}$Th SEP', r'$^{228}$Th -> Tl']

    for i in range(n_peaks):

        #Get histogram for peak within bounds of 5 sigma
        hi, lo = modes[i] + widths[i], modes[i] - widths[i]
        hist, bins, var = pgh.get_hist(energies, bins=100, range=(lo, hi))
        bin_centers = pgh.get_bin_centers(bins)

        #Compute components of fit function
        fit = func(bin_centers, *fit_pars[i], components=False)
        gaussian, step = func(bin_centers, *fit_pars[i], components=True)
        sigmas.append(fit_pars[i][2])

        #Plot data and fit components
        if plotBool:
            axs[i].semilogy(bin_centers, hist, ds="steps-mid", color="k")#, label=labels[i])
            axs[i].semilogy(bin_centers, fit, color="r", label='fit')
            axs[i].semilogy(bin_centers, gaussian, color="orange", label="gaussian")
            axs[i].semilogy(bin_centers, step, color="cyan", label="step")
            axs[i].set_ylim(hist[-1:]+1, np.amax(hist)+100)
            axs[i].legend(fontsize=18, loc='best')
    
    plt.savefig(f"{savePath}/fitPeaks.jpg")
    plt.cla()
    plt.clf()

    mus = np.asarray([fit_pars[i][1] for i in range(len(fit_pars))])
    mu_errs = np.asarray([fit_errs[i][1] for i in range(len(fit_pars))]) 

    cal_pars, fitData = match_peaks(mus, cal_peaks)
    print(f"Cal pars {cal_pars}")
    print(f"Fit Data {fitData}")

    if plotBool:
        for i in range(2, 4):
            sideband_LE_left = peaks[i]-(2.5*sigmas[i]+4*sigmas[i])
            sideband_LE_right = peaks[i]-(2.5*sigmas[i])
            sideband_HE_left = peaks[i]+(2.5*sigmas[i]+4*sigmas[i])
            sideband_HE_right = peaks[i]+(2.5*sigmas[i])

            plt.hist(cal_energies_first[(cal_energies_first>peaks[i]-200)*(cal_energies_first<peaks[i]+200)], bins=1000, color='k', ec='k')
            plt.axvline(peaks[i], 0, 5e5, color='r', lw=1, alpha=0.75)
            plt.axvline(peaks[i] + sigmas[i], 0, 5e5, color='r', lw=2, alpha=0.75)
            plt.axvline(peaks[i] - sigmas[i], 0, 5e5, color='r', lw=2, alpha=0.75)
            plt.axvline(sideband_LE_left, 0, 5e5, color='b', lw=2, alpha=1)
            plt.axvline(sideband_LE_right, 0, 5e5, color='b', lw=2, alpha=1)
            plt.axvline(sideband_HE_left, 0, 5e5, color='b', lw=2, alpha=1)
            plt.axvline(sideband_HE_right, 0, 5e5, color='b', lw=2, alpha=1)
            
            plt.xlabel('Energy (keV)', fontsize=24)
            plt.ylabel('Counts', fontsize=24)
            plt.yscale('log')
            plt.xlim(peaks[i] - 10*sigmas[i], peaks[i] + 10*sigmas[i])
            if i == 2:
                plt.tile("DEP Peak")
                plt.savefig(f"{savePath}/EnergyHist_DEP.jpg")
            if i == 3:
                plt.title("SEP Peak")
                plt.savefig(f"{savePath}/EnergyHist_SEP.jpg")
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
        # plt.savefig(f"{savePath}/EnergyHist_SEP.jpg")
        # plt.clf()
        # plt.cla()




    if verbose:
        print(f"Known Energy: {fitData[0][peakIndex]}, Calibrated ADC {fitData[1][peakIndex]}")

    return cal_pars, [fitData, fit_pars], peakIndex

if __name__ == "__main__":
    calibration()