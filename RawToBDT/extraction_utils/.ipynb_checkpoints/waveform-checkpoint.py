import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from MLutils.config import terminalCMAP

from tqdm import tqdm

def cTimes(wfParams, number = 1000, save=True):
    [dt, t0, val] = wfParams
    t=np.zeros(val.shape)
    if number:
        rg = val.shape[0]
    else:
        rg = number
        
    for event in tqdm(range(number), desc=f"Loading...", ascii=False, ncols=75):#):
        t[event, 0] = t0[event]
        for points in range(1,val.shape[1]):
            t[event, points] = t[event, points-1] + dt[event]
            
    if save:
        np.save("timesArr.npy", t)
    
    return t

def plotWF(times, values):
    # for single Waveform
    plt.plot(times, values, color = terminalCMAP[1])
    plt.xlabel("time")
    plt.ylabel("ADC")
    return