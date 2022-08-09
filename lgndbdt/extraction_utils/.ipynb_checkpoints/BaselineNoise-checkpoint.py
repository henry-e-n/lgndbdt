import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from MLutils.config import *
from tqdm import tqdm
import os
import sys
import scipy.optimize as spy
cwd = os.getcwd()
module_path = os.path.abspath(os.path.split(cwd)[0])
sys.path.append(module_path)


from MLutils.h5Extract import *
from MLutils.waveform import *
from MLutils.AvsE import *
from MLutils.DCR import *

def blWindow(startTime, dtime, buffer = 1):
    #print(f"Buffer is {buffer} \u03BCs")
    startCell = startTime/dtime
    if buffer == 0:
        window = [0, int(startCell)]
    elif buffer >= 0:
        left = int(buffer*us2cell(dtime))
        right = int(startCell - buffer*us2cell(dtime))
        if left >= right:
            print("Invalid buffer Size, please choose a smaller buffer")
            return
        window = [left, right]
    else:
        print("Sorry invalid buffer, please enter an integer or float >= 0")
    return window

def blAverage(indexWindow, values):
    [leftInd, rightInd] = indexWindow
    averageValue = np.mean(values[leftInd:rightInd])
    return averageValue

def linFit(x, m, b):
    return m*x + b

def blLinFit(indexWindow, times, values):
    xs = times[indexWindow[0]: indexWindow[1]]
    [leftInd, rightInd] = indexWindow
    ys = values[leftInd:rightInd]
    
    popt, pcov = spy.curve_fit(linFit, xs, ys)
    return popt

def avgFit(x, avg):
    return x*0 + avg
"""
def visBL(window, startTime, blAvg, dtime, times, values, colorIter = 0.3):
    startCell = int(startTime/dtime)
    plt.plot(times[:startCell], values[:startCell], color = cmapNormal(colorIter), alpha = 0.5)
    plt.plot(times[window[0]:window[1]],times[window[0]:window[1]]*0 + blAvg, color = cmapNormal(colorIter))
"""
def visBL2(window, startTime, func, popt, dtime, times, values, colorIter = 0.3):
    startCell = int(startTime/dtime)
    plt.plot(times[:startCell], values[:startCell], color = cmapNormal(colorIter), alpha = 0.5)
    plt.plot(times[window[0]:window[1]],func(times[window[0]:window[1]], *popt), color = cmapNormal(colorIter))
    plt.xlabel("time")
    plt.ylabel("ADC, I think")

    
def findNoise(func, popt, window, times, values):
    exp = func(times[window[0]:window[1]], *popt)
    act = values[window[0]:window[1]]
    diff = np.absolute(exp-act)
    squareDiff = np.dot(diff, diff)
    sumSquare = np.sum(squareDiff)
    avgSquare = sumSquare/len(act)
    rms = np.sqrt(avgSquare)
    return rms