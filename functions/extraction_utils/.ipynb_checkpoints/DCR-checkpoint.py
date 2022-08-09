import numpy as np
from tqdm import tqdm
from matplotlib import cm
from MLutils.config import cmapNormal, terminalCMAP
from MLutils.waveform import *

from scipy.optimize import minimize


def find97(vals):
    peak = np.max(vals)
    peakInd = vals.argmax()
    val97 = 0.97*peak
    diffArr = np.absolute(vals - val97)
    closestFromPeak = diffArr[peakInd:].argmin()
    closest = peakInd+closestFromPeak

    return closest

def us2cell(dtimes, us = 1000):
    return int(us/dtimes)

def boxAverage(values, index, side, dtimes, boxWidth = 1):
    # boxWidth in us
    widthCells = us2cell(dtimes, 1000*boxWidth)
    if side == "left":
        boxSum = np.sum(values[index:index+widthCells])
        leftSide = index
    elif side == "right":
        boxSum = np.sum(values[index-widthCells:index])
        leftSide = index-widthCells
        if leftSide < 0:
            leftSide = len(values)+leftSide
    avgBox = boxSum/widthCells
    return avgBox, leftSide

def findSlope(values, dtimes):
    avg1, left1 = boxAverage(values, find97(values), 'left', dtimes)
    avg2, left2 = boxAverage(values, -1, 'right', dtimes)
    
    delta = (avg2 - avg1)/(left2-left1)
    return delta

def findSlopeCorr(values, valuesCorrected, dtimes):
    avg1, left1 = boxAverage(valuesCorrected, find97(values), 'left', dtimes)
    avg2, left2 = boxAverage(valuesCorrected, -1, 'right', dtimes)
    
    delta = (avg2 - avg1)/(left2-left1)
    return delta

def visualizeDCR(times, values, dtimes, ind97 = []):
    plotWF(times, values)
    if ind97 == []:
        ind97 = find97(values)
    avg1, left1 = boxAverage(values, ind97, 'left', dtimes)
    avg2, left2 = boxAverage(values, -1, 'right', dtimes)
    plt.plot(times[left1:left1+us2cell(dtimes)], times[left1:left1+us2cell(dtimes)]*0 + avg1, '*', color = terminalCMAP[2])
    plt.plot(times[left2:-1], times[left2:-1]*0 +avg2, '*', color=terminalCMAP[2])
    plt.xlabel("time")
    plt.ylabel("ADC")
    plt.show()
    return

"""
def poleCorrection(times, values, averageDelta, numWF = 1000):
    valsCorr = np.copy(values)
    for m in tqdm(range(numWF), desc="Running Pole Correction Algorithm..."):
        peakInd = np.argmax(values[m])
        #plotWF(times[m, :], vals[m, :])
        #plt.plot(times[m, peakInd:], np.arange(len(vals[m, peakInd:]))*(-avgDeltas) + vals[m, peakInd:], 'r')
        valsCorr[m, peakInd:] = np.arange(len(values[m, peakInd:]))*(-averageDelta) + values[m, peakInd:]
    return valsCorr

"""

### P0 Correction Code
"""
def updatePbar(x):
    with tqdm(total=200) as pbar:
        pbar.update(x)
    
    wfArray, neval = info[0]
    
    info[1] += 1
    if info[1] % 10 == 0:
        updatePbar(info[1])

"""
def dp0fx(popt, wfArray):
    [tau1, tau2, f] = popt
    y_out = np.zeros(wfArray.shape[0])
    max_amp = 10
    
    for wf in range(wfArray.shape[0]):
        wf_in = wfArray[wf,:]
        wf_in = (wf_in-wf_in.min())/(wf_in.max()-wf_in.min())*max_amp
        wf_out = np.zeros(len(wf_in))
        # Defines the constant terms
        const1 = 1 / tau1
        const2 = 1 / tau2
        # Defines the exponential terms
        exp1 = np.exp(-1 / tau1)
        exp2 = np.exp(-1 / tau2)
        
        frac = f
        # Sets initial of output to same as input
        wf_out[0] = wf_in[0]
        e1 = e2 = wf_in[0]
        e3 = 0
        
        for i in range(1, len(wf_in), 1): # Iterates over rest of wf
            e1 += wf_in[i] - e2 + e2*const1
            e3 += wf_in[i] - e2 - e3*const2
            e2  = wf_in[i]
            
            wf_out[i] = e1 - frac*e3
        
        maxInd = np.argmax(wf_in) # Index of max
        pre_wf = wf_in[:maxInd] # waveform before max
        
        peak = np.max(wf_in)
        peakInd = wf_in.argmax()
        val97 = 0.97*peak
        diffArr = np.absolute(wf_in - val97)
        closestFromPeak = diffArr[peakInd:].argmin()
        ind97 = peakInd+closestFromPeak
        y_out[wf] = (np.std(wf_out[(ind97):]))
    
    return np.mean(y_out)

def dp0Vis(popt, wfArray):
    [tau1, tau2, f] = popt
    wfInAdj = np.zeros(wfArray.shape)
    wfCorr = np.zeros(wfArray.shape)
    max_amp = 10
    
    for wf in tqdm(range(wfArray.shape[0]), desc="Applying P0 to waveforms...", colour = terminalCMAP[0]):
        wf_in = wfArray[wf,:]
        wf_in = (wf_in-wf_in.min())/(wf_in.max()-wf_in.min())*max_amp
        wfInAdj[wf, :] = wf_in
        wf_out = np.zeros(len(wf_in))
        # Defines the constant terms
        const1 = 1 / tau1
        const2 = 1 / tau2
        # Defines the exponential terms
        exp1 = np.exp(-1 / tau1)
        exp2 = np.exp(-1 / tau2)
        
        frac = f
        # Sets initial of output to same as input
        wf_out[0] = wf_in[0]
        e1 = e2 = wf_in[0]
        e3 = 0
        
        for i in range(1, len(wf_in), 1): # Iterates over rest of wf
            e1 += wf_in[i] - e2 + e2*const1
            e3 += wf_in[i] - e2 - e3*const2
            e2  = wf_in[i]
            
            wfCorr[wf, i] = e1 - frac*e3
    return wfInAdj, wfCorr


