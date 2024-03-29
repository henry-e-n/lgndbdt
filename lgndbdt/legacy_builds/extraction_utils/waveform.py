import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from extraction_utils.config import terminalCMAP, savePath, detName

from tqdm import tqdm

def cTimes(wfParams, number, save=True):
    [dt, t0, val] = wfParams
    if number == -1:
        rg = 1 # Save small file of one row
        t=np.zeros((1,val.shape[1]))
    else:
        rg = val.shape[0] # Save full file
        t=np.zeros(val.shape)
    for event in tqdm(range(rg), desc=f"Loading Raw Data.............:", ascii=False, ncols=75, colour = terminalCMAP[0]):#):
        t[event, 0] = t0[event]
        if event > 1 and t0[event] == t0[event-1] and dt[event] == dt[event-1]:
            t[event, :] = t[event-1, :]
        else:
            for points in range(1,val.shape[1]):
                t[event, points] = t[event, points-1] + dt[event]
            
    if save:
        np.save(f"{savePath}/timesArr_{detName}.npy", t)
    return t

def plotWF(times, values):
    # for single Waveform
    plt.plot(times, values, color = terminalCMAP[1])
    plt.xlabel("time")
    plt.ylabel("ADC")
    return