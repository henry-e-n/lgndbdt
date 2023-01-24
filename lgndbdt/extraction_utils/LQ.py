import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
from matplotlib import cm
from extraction_utils.config import *


def find80(vals, buffer = 100):
    """ 
    Finds the index along waveform rise = 80% of max
    """
    totalPeakInd = np.argmax(vals[:-2])
    if totalPeakInd >= len(vals)-buffer:
        return -1
    peak = np.max(vals[:-buffer])
    peakInd = vals[:-buffer].argmax()
    val80 = 0.80*peak
    diffArr = np.absolute(vals - val80)
    closestFromPeak = diffArr[:peakInd].argmin()
    closest = closestFromPeak
    return closest

def getMid(ts, ind80, buffer = 100):
    """
    Finds the index of the middle of the flat tail 
        used to split tail and compare rise integral to tail integral
    """
    endOfInt = len(ts) - buffer
    remaining = endOfInt-ind80
    if remaining % 2 == 1:
        remaining = remaining - 1
    midInd = int(remaining/2) + ind80
    return midInd, endOfInt, buffer

def getLQ80(ts, vals, trashPZ):
    """
    Returns the LQ80 of a waveform - the area above the turning peak
    Parameters: 
        ts   - independent variable of waveform
        vals - Pole 0 Corrected dependent variable of waveform
    """
    LQ80 = np.zeros(vals.shape[0])
    trash_ind = []
    for i in range(vals.shape[0]):
        ind80 = find80(vals[i])
        if ind80 == -1 or i in trashPZ:
            trash_ind.append(i)
        else:
            midInd, endOfInt, buffer = getMid(ts[0], ind80)
            blue = (np.trapz(vals[i, ind80:midInd],ts[0, ind80:midInd]))
            red = (np.trapz(vals[i, midInd:endOfInt],ts[0, midInd:endOfInt]))

            LQ80[i] = red-blue
    return LQ80, trash_ind

def LQvis(ts, vals):
    """
    Visualizes the LQ algorithm
    """
    ind80 = find80(vals)
    midInd, endOfInt, buffer = getMid(ts, ind80)

    plt.plot(ts[ind80-buffer:], vals[ind80-buffer:])
    plt.fill_between(ts[ind80:midInd], 0, vals[ind80:midInd],  color = terminalCMAP[0], alpha=.3)
    plt.fill_between(ts[midInd:endOfInt], 0, vals[midInd:endOfInt], color = terminalCMAP[1], alpha=.3)

    return

def LQvisZoom(ts, vals):
    """
    Zoomed in visualization of LQ
    """
    ind80 = find80(vals)
    midInd, endOfInt, buffer = getMid(ts, ind80)
    peakInd = np.argmax(vals)
    meanTop = np.mean(vals[ind80:endOfInt])
    plt.plot(ts[ind80-2:ind80+16], vals[ind80-2:ind80+16], linewidth=6, color = terminalCMAP[1])
    # plt.fill_between(ts[ind80:ind80+25], np.min(vals[ind80-5:ind80+25]), vals[ind80:ind80+25],  color = terminalCMAP[0], alpha=.3)
    plt.fill_between(ts[ind80:ind80+16], meanTop, vals[ind80:ind80+16],  linewidth=8, color = terminalCMAP[0], alpha=.4, hatch='\\\\')
    return