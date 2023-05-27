import numpy as np
import tqdm

from utilities.global_config import terminalCMAP

def find_percent(vals, buffer = 100, percent = 0.8):
    """ 
    Finds the index along waveform rise = 80% of max
    """
    totalPeakInd = np.argmax(vals[:-2])
    if totalPeakInd >= len(vals)-buffer:
        return -1
    peak = np.max(vals[:-buffer])
    peakInd = vals[:-buffer].argmax()
    val80 = percent*peak
    diffArr = np.absolute(vals - val80)
    closestFromPeak = diffArr[:peakInd].argmin()
    closest = closestFromPeak
    return closest


def get_mid_index(ts, ind80, buffer = 100):
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


def get_LQ80(ts, vals, trashPZ):
    LQ80 = np.zeros(vals.shape[0])
    trash_ind = []
    for i in tqdm(range(vals.shape[0]), 
                  desc   ="Running LQ80..................", 
                  colour = terminalCMAP[1]):
        if i in trashPZ:
            trash_ind.append(i)
        else:
            ind80 = find_percent(vals[i])
            indPeak = find_percent(vals[i], percent = .99)
            if indPeak-ind80 <0:
                # print(i, indPeak, ind80)
                trash_ind.append(i)
            else:
                if ind80 == -1:
                    trash_ind.append(i)
                else:
                    midInd, endOfInt, buffer = get_mid_index(ts[0], ind80)
                    avgTailVal = np.mean(vals[i, midInd:endOfInt])
                    # print(avgTailVal)
                    auc = (np.trapz(vals[i, ind80:indPeak],ts[0, ind80:indPeak]))
                    # print(avgTailVal*np.ones(midInd-ind80))
                    auMean = (np.trapz(avgTailVal*np.ones(indPeak-ind80),ts[0, ind80:indPeak]))
                    LQ80[i] = auMean-auc
                    # plt.fill_between(ts[0, ind80:indPeak], avgTailVal, vals[i, ind80:indPeak],  linewidth=1, color = terminalCMAP[0], alpha=.4, hatch='\\\\')
                    # plt.plot(ts[0, ind80-10:indPeak+10], vals[i, ind80-10:indPeak+10])
                    # plt.xlim(9500,10600)
                    if LQ80[i] < 0:
                        # print(i, LQ80[i])
                        trash_ind.append(i)
    return LQ80, trash_ind
