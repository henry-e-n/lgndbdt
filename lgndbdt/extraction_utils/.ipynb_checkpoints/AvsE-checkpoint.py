import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from tqdm import tqdm

from matplotlib import cm
from MLutils.config import *

def getA(times, values, dtimes, currentEstimator = 100):
    # Arguments:
    # times array - [#Waveforms, #TimeCells]
    # values - [#TimeCells] - choose one waveform
    # dtimes - [#Waveforms, #TimeCells]
    dWindow = int(currentEstimator/dtimes[0])
    slopes = np.zeros(times.shape[1]-dWindow)
    for i in range(times.shape[1]-dWindow):
        leftVal = float(values[i])
        rightVal = float(values[i+dWindow])
        rise = rightVal-leftVal
        slopes[i] = rise
    return slopes

def AvsE(times, values, dtimes, plots = [], numWF = 2500, currentEstimator = 100):
    # Arguments:
    # times array - [#Waveforms, #TimeCells]
    # values - [#Waveforms, #TimeCells] - choose one waveform
    # dtimes - [#Waveforms, #TimeCells]
    
    # Plots should either be:
    # False - no plots to be printed
    # E - energy array same length as times
    maxA = np.zeros(numWF)

    for m in tqdm(range(numWF), desc = "Getting Current Amplitude...", colour = terminalCMAP[0]):
        aArr = getA(times, values[m, :], dtimes)
        maxA[m] = np.max(aArr)
    
    if not plots==[]:
        plt.plot(plots[:len(maxA)], maxA,'.', color = cmapNormal(0.3))
        plt.xlabel("Energy (MeV)")
        plt.ylabel("Max Current (A)")
        plt.show()
    return maxA