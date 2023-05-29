import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
from extraction_utils.DCR import EfromA
from extraction_utils.config import *
from time import time

def getA(times, values, dtimes, currentEstimator = 100):
    """
    Arguments:
        - times array - [#Waveforms, #TimeCells]
        - values - [#TimeCells] - choose one waveform
        - dtimes - [#Waveforms, #TimeCells]
    """
    
    dWindow = int(currentEstimator/dtimes[0])
    slopes = np.zeros(times.shape[1]-dWindow)
    for i in range(times.shape[1]-dWindow):
        leftVal = float(values[i])
        rightVal = float(values[i+dWindow])
        rise = rightVal-leftVal
        slopes[i] = rise
    return slopes

def AvsE(values, dtimes, plots = [], numWF = 2500, currentEstimator = 100):
    """
    Arguments:
        - ntimes array - [#Waveforms, #TimeCells]
        - values - [#Waveforms, #TimeCells] - choose one waveform
        - dtimes - [#Waveforms, #TimeCells]
    
    Plots should either be:
        - False - no plots to be printed
        - E - energy array same length as times
    """
    dWindow = int(currentEstimator/dtimes[0])
    wfdLen = values.shape[1]
    row = np.zeros(wfdLen)
    row[0] = -1
    row[dWindow] = 1
    tensor = np.tile(row, (wfdLen, 1))
    for i in range(wfdLen):
        tensor[i] = np.roll(tensor[i], i)
    
    tensor = np.triu(tensor, -1)
    maxA = np.zeros(numWF)
    # plt.imshow(tensor[:100, :100])
    # plt.show()
    for w in tqdm(range(numWF),
                  desc = "Getting Current Amplitude.....",
                  colour = terminalCMAP[1]):
        maxA[w] = np.max(np.dot(values[w, :], tensor.T))
    return maxA