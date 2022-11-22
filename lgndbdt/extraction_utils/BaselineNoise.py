import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from extraction_utils.config import *
from tqdm import tqdm
import os
import sys
import scipy.optimize as spy
cwd = os.getcwd()
module_path = os.path.abspath(os.path.split(cwd)[0])
sys.path.append(module_path)


from extraction_utils.waveform import *
from extraction_utils.AvsE import *
from extraction_utils.DCR import *

###########################################################################################
###########################################################################################

def blWindow(startTime, dtime, buffer = 1):
    """
    
    Arguments:
        - startTime: start time (not cell) of waveform rise
        - dtime: Time delta between cells
        - *buffer = 1: Buffer time at end of waveform in us
    """
    startCell = startTime/dtime
    if buffer == 0:
        window = [0, int(startCell)]
    elif buffer >= 0:
        try:
            left = int(buffer*us2cell(dtime))
            right = int(startCell - buffer*us2cell(dtime))
            if left >= right:
                print("Invalid buffer Size, please choose a smaller buffer")
                window = [0, int(startCell)]
                return
            window = [left, right]
        except ValueError:
            print("Value Error Exception")
            window = [0, int(300)]
    else:
        print("Sorry invalid buffer, please enter an integer or float >= 0")
    return window

def tailWindow(startTime, dtime, buffer = 1):
    """
    
    Arguments:
        - startTime: start time (not cell) of waveform rise
        - dtime: Time delta between cells
        - *buffer = 1: Buffer time at end of waveform in us
    """
    startCell = startTime/dtime
    window = [int(startCell), -(buffer*us2cell(dtime))]
    return window

###########################################################################################

def blAverage(indexWindow, values):
    """
    Function: Average ADC values within window given by indexWindow

    Arguments:
        - indexWindow: [left index, right index] of window to be averaged
        - values: Waveform ADC values
    """
    [leftInd, rightInd] = indexWindow
    averageValue = np.mean(values[leftInd:rightInd])
    return averageValue

###########################################################################################

def linFit(x, m, b):
    """
    Function: Linear Fit Function

    Arguments:
        - x: independent variable
        - m: slope
        - b: y-intercept
    """
    return m*x + b

###########################################################################################

def blLinFit(indexWindow, times, values):
    """
    Function: Fits the baseline bounded by indexWindow to a linear regression

    Arguments:
        - indexWindow: [left index, right index] of window to be fit
        - times: Waveform times (independent) array
        - values: Waveform ADC values
    
    """
    try:
        print(indexWindow[1])
    except TypeError:
        print(f"TYPE ERROR {indexWindow, values[0:100]}")
        indexWindow = [0, int(300)]

    xs = times[indexWindow[0]: indexWindow[1]]
    [leftInd, rightInd] = indexWindow
    ys = values[leftInd:rightInd]
    
    popt, pcov = spy.curve_fit(linFit, xs, ys)
    return popt

###########################################################################################

def avgFit(x, avg):
    return x*0 + avg

###########################################################################################

def findNoise(func, popt, window, times, values):
    """
    Function: Estimates the noise within a window of the waveform

    Arguments:
        - func: function to use to estimate noise
        - popt: Necessary parameters for func
        - window: window in which to estimate noise
        - times: waveform times array
        - values: waveform values array

    Returns:
        - rms: Average Value of Noise within window (float64)
    
    """
    exp = func(times[window[0]:window[1]], *popt)
    act = values[window[0]:window[1]]
    diff = np.absolute(exp-act)
    squareDiff = np.dot(diff, diff)
    sumSquare = np.sum(squareDiff)
    avgSquare = sumSquare/len(act)
    rms = np.sqrt(avgSquare)
    return rms

###########################################################################################
###########################################################################################
"""
def visBL(window, startTime, blAvg, dtime, times, values, colorIter = 0.3):
    startCell = int(startTime/dtime)
    plt.plot(times[:startCell], values[:startCell], color = cmapNormal(colorIter), alpha = 0.5)
    plt.plot(times[window[0]:window[1]],times[window[0]:window[1]]*0 + blAvg, color = cmapNormal(colorIter))
"""
def visBL2(window, startTime, func, popt, dtime, times, values, colorIter = 0.3):
    startCell = int(startTime/dtime)
    plt.plot(times[window[0]:window[1]], values[window[0]:window[1]], color = cmapNormal(colorIter), alpha = 0.5)
    plt.plot(times[window[0]:window[1]],func(times[window[0]:window[1]], *popt), color = cmapNormal(colorIter))
    plt.xlabel("time")
    plt.ylabel("ADC, I think")
    plt.show()
    return
###########################################################################################
###########################################################################################
