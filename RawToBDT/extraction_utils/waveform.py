import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from extraction_utils.config import terminalCMAP, lpData

from tqdm import tqdm

def cTimes(wfParams, choosePeak, number, save=True):
    [dt, t0, val] = wfParams
    t=np.zeros(val.shape)
    if number == -1:
        rg = val.shape[0]
    else:
        rg = val.shape[0] # Actually always save full times file
        
    for event in tqdm(range(rg), desc=f"Loading...", ascii=False, ncols=75, colour = terminalCMAP[1]):#):
        t[event, 0] = t0[event]
        if event > 1 and t0[event] == t0[event-1] and dt[event] == dt[event-1]:
            t[event, :] = t[event-1, :]
        else:
            for points in range(1,val.shape[1]):
                t[event, points] = t[event, points-1] + dt[event]
            
    if save:
        np.save(f"{lpData}DataFiles/timesArr_{choosePeak}.npy", t)
    
    return t

def plotWF(times, values):
    # for single Waveform
    plt.plot(times, values, color = terminalCMAP[1])
    plt.xlabel("time")
    plt.ylabel("ADC")
    return