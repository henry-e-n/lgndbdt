import json, os, sys, h5py
import numpy as np
from scipy.optimize import curve_fit

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

##############################
def calibration():
    

    energy_stack = lh5.load_nda(dsp_files, ["trapEmax"], "icpc1/dsp")
    energies = energy_stack["trapEmax"]
    
    # dsp, keys, energies = paramExtract(dspFile, ["trapEmax"])
    # energies = energies[0]

    #####################################
    # First Calibration Pass
    #####################################
    if source == "60Co":
        peaks = np.array([583.187, # 228Th -> 208Tl (85%)
                          1173.24, # 60Co
                          1332.5, # 60Co,
                          1592.5, # 228Th DEP
                          2103.5, # 228Th SEP
                          2614.53]) # 228Th -> 208Tl (99.8%) 
        if targetPeak == "228ThDEP":
            print("DEP")
            peakIndex = 3
        elif targetPeak == "228ThSEP":
            print("SEP")
            peakIndex = 4
        
    else:
        peaks = np.array([583.187, # 228Th -> 208Tl (85%)
                          2614.53]) # 228Th -> 208Tl (99.8%) 

    hist, bins, var = pgh.get_hist(energies, bins=1000)
    uncal_peaks, cal_peaks, cal_pars = pgc.hpge_find_E_peaks(hist, bins, var, peaks)

    def match_peaks(data_pks, cal_pks):
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

        # print(i, best_err)
        # print("cal:",cal)
        # print("data:",data)
        # Plots Scatter of Calibration Pass
        # plt.scatter(data, cal, label='min.err:{:.2e}'.format(err))
        # xs = np.linspace(data[0], data[-1], 10)
        # plt.plot(xs, best_m * xs + best_b , c="r",
        #          label="y = {:.2f} x + {:.2f}".format(best_m,best_b) )
        # plt.xlabel("Energy (ADC)", fontsize=24)
        # plt.ylabel("Energy (keV)", fontsize=24)
        # plt.legend(loc='best', fontsize=20)
        # plt.show()

        return [best_m, best_b], [cal, data]

    linear_cal, fitDat = match_peaks(uncal_peaks, cal_peaks)
    print(linear_cal)

    def linearFit(en, linCal):
        calibratedEnergy = en*linCal[0] + linCal[1]
        return calibratedEnergy
    cal_energies_first = linearFit(energies, linear_cal)

    # Plots Energy Histogram
    # plt.hist(cal_energies_first, bins=1000, color='k', ec='k')
    # for peak in peaks:
    #     plt.axvline(peak, 0, 5e5, color='r', lw=1, alpha=0.75)

    # plt.xlabel('Energy (keV)', fontsize=24)
    # plt.ylabel('Counts', fontsize=24)
    # plt.yscale('log')
    # plt.xlim(0,3000)
    # plt.show()


    #####################################
    # Second Calibration Pass
    #####################################
    pars_list, _ = pgc.hpge_fit_E_peak_tops(hist, bins, var, uncal_peaks, n_to_fit=5)
    modes = np.asarray([pars_list[i][0] for i in range(len(pars_list))])
    sigmas = np.asarray([pars_list[i][1] for i in range(len(pars_list))]) 

    #Plot histograms and fit initial functions

    if targetPeak == "228ThDEP" or targetPeak=="228ThSEP":
        widths = np.array([sigmas[0]*3,
                   sigmas[1]*4, 
                   sigmas[2]*6,
                   sigmas[3]*5,
                   sigmas[4]*4, 
                   sigmas[5]*8])
    else:
        widths = sigmas * 3

    n_peaks = uncal_peaks.shape[0]
    # fig, axs = plt.subplots(n_peaks, 1, figsize=(12,24)) 
    # labels = [r'$^{228}$Th', r'$^{60}$Co', r'$^{60}$Co', r'$^{228}$Th', r'$^{228}$Th SEP'] #If other peaks are chosen, make sure to modify this

    for i in range(n_peaks):
        #Get histogram for peak within bounds of 5 sigma
        hi, lo = modes[i] + widths[i], modes[i] - widths[i]
        hist, bins, var = pgh.get_hist(energies, bins=100, range=(lo, hi))
        bin_centers = pgh.get_bin_centers(bins)

        #Plot data 
        # axs[i].semilogy(bin_centers, hist, ds="steps-mid", color="k", label=labels[i])
        # axs[i].legend(fontsize=30, loc='best')

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

    #print the fit errors for each parameter, useful for debugging
    # print(fit_errs)


    #Plot fit results and compare to data
    # fig, axs = plt.subplots(n_peaks, 1, figsize=(12,24)) 

    sigmas = []
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
        # axs[i].semilogy(bin_centers, hist, ds="steps-mid", color="k", label=labels[i])
        # axs[i].semilogy(bin_centers, fit, color="r", label='fit')
        # axs[i].semilogy(bin_centers, gaussian, color="orange", label="gaussian")
        # axs[i].semilogy(bin_centers, step, color="cyan", label="step")
        # axs[i].set_ylim(hist[-1:]+1, np.amax(hist)+100)
        # axs[i].legend(fontsize=18, loc='best')

    mus = np.asarray([fit_pars[i][1] for i in range(len(fit_pars))])
    mu_errs = np.asarray([fit_errs[i][1] for i in range(len(fit_pars))]) 

    cal_pars, fitData = match_peaks(mus, cal_peaks)

    return cal_pars, [fitData, fit_pars], peakIndex

if __name__ == "__main__":
    calibration()