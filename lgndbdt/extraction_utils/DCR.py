import numpy as np
from tqdm import tqdm
from matplotlib import cm
from extraction_utils.config import cmapNormal, terminalCMAP, plotPath
from extraction_utils.waveform import *

from scipy.optimize import minimize, curve_fit

###########################################################################################
###########################################################################################

def find97(vals):
    """
    Function: Finds index of 97% along trailing edge of waveform

    Parameters: 
        - vals: waveform ADC values
    
    Returns:
        - closest: index of 97% along trailing edge
    """
    peak = np.max(vals)
    peakInd = vals.argmax()
    val97 = 0.97*peak
    diffArr = np.absolute(vals - val97)
    closestFromPeak = diffArr[peakInd:].argmin()
    closest = peakInd+closestFromPeak

    return closest

###########################################################################################

def us2cell(dtimes , ns = 1000):
    """
    Function: Takes in time (in ns) to be changed to cell width (based on dt) 

    Parameters:
        - dtimes: time cell deltas in ns (i think)
    
    Returns:
        - number of cells corresponding to time
    """
    return int(ns/dtimes)

###########################################################################################

def boxAverage(values, index, side, dtimes, boxWidth = 1):
    """
    Function: Computes average ADC value of defined box

    Parameters:
        - values: Waveform ADC values
        - index: Index of evaluation box edge
        - side: Which edge as given by index
        - dtimes: time cell deltas
        - *boxWidth = 1 defines the box width in microseconds
    
    Returns
        - avgBox: Average ADC value of box
        - leftSide: Index of the left side of the box
    """
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

###########################################################################################

def findSlope(values, dtimes):
    """
    Function: Finds the slope of the trailing tail of the Waveform before P0 correction

    Parameters:
        - values: Waveform ADC values
        - dtimes: time cell deltas

    Returns:
        - delta: Estimated Slope of tail    
    """
    avg1, left1 = boxAverage(values, find97(values), 'left', dtimes)
    avg2, left2 = boxAverage(values, -1, 'right', dtimes)
    
    delta = (avg2 - avg1)/(left2-left1)
    return delta

###########################################################################################

def findSlopeCorr(values, valuesCorrected, dtimes):
    """
    Function: Finds the slope of the trailing tail of the Waveform after P0 correction

    Parameters:
        - values: Waveform ADC values
        - values: P0 corrected waveform ADC values
        - dtimes: time cell deltas

    Returns:
        - delta: Estimated Slope of tail    
    """
    avg1, left1 = boxAverage(valuesCorrected, find97(values), 'left', dtimes)
    avg2, left2 = boxAverage(valuesCorrected, -1, 'right', dtimes)
    
    delta = (avg2 - avg1)/(left2-left1)
    return delta

###########################################################################################

def visualizeDCR(times, values, dtimes, ind97 = []):
    """
    Function: Visualizes the DCR parameter extraction
    """
    plotWF(times, values)
    if ind97 == []:
        ind97 = find97(values)
    avg1, left1 = boxAverage(values, ind97, 'left', dtimes)
    avg2, left2 = boxAverage(values, -1, 'right', dtimes)
    plt.plot(times[left1:left1+us2cell(dtimes)], times[left1:left1+us2cell(dtimes)]*0 + avg1, '*', color = terminalCMAP[2], label="DCR Windows")
    plt.plot(times[left2:-1], times[left2:-1]*0 +avg2, '*', color=terminalCMAP[2])
    plt.xlabel("time")
    plt.ylabel("ADC")
    # plt.show()
    return

###########################################################################################
### P0 Correction Code
###########################################################################################

def dp0fx(popt, wfArray):
    """
    Function: Pole Zero function

    Parameters:
        - popt: [tau1, tau2, f] 
            - tau1, tau2 - 2 exponential decay terms
            - f - relation fraction
        - wfArray: [len(Waveforms), ADC values len(cells)]
    
    Returns:
        - y_out: mean of standard deviation of fit waveforms
    """
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

###########################################################################################

def dp0Vis(popt, wfArray):
    [tau1, tau2, f] = popt
    wfInAdj = np.zeros(wfArray.shape)
    wfCorr = np.zeros(wfArray.shape)
    max_amp = 10
    
    for wf in tqdm(range(wfArray.shape[0]),
                    desc="Applying P0 to waveforms......",
                    colour = terminalCMAP[0]):
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

###########################################################################################

def getP0(vals, popt, numWave = 100):
    """
    Function: Performs P0 correction on waveforms

    Parameters:
        - vals: Waveform ADC values [len(numWave), len(Cells)]
        - popt: Boolean 0, runs minimization on P0 function to determine slope correction, uses input popt
        - *numWave: number of waveforms
    
    Returns:
        - wfIn: input waveforms in corresponding data structure
        - wfCorr: Corrected waveforms
    """
    if popt == 0:
        if numWave >= 10:
            dp0Num = 10
        else:
            dp0Num = numWave
            
        # print(f"Running P0 minimization - Please Wait...")
        res = minimize(dp0fx,
                    [72*40, 2.1*40, 0.0105], 
                    args    = vals[:dp0Num,:], 
                    method  = 'Nelder-Mead', 
                    tol     = 1e-4, 
                    bounds  = ((60*40, 90*40), (1, 5*40),(0.01,0.012)))
        
        # print(f"\nP0 fit parameters - {res.x/40}")
        popt = tuple(res.x)
        # print(f"fitted value {dp0fx(res.x, vals[:1,:])}")
        # print(f"initial value {dp0fx([72*40, 2.1*40, 0.0105], vals[:1,:])}\n")
        # print(f"POPT {popt}")
    wfIn, wfCorr  = dp0Vis(popt, vals[:numWave,:])
    return wfIn, wfCorr

###########################################################################################

def trapENS3(times, values, dtimes, intTimes = (1000, 5000)):
    """
    Returns a Simple Trapezoidal Filter to Determine Energy from Waveform
    Parameters:
        - times array     -- (waveforms, cells)
        - ADC values      -- (waveforms, cells)
        - time resolution -- (waveforms)
        - * integration times in ns (riseTime, flat top Time)
    """
    riseTime   = intTimes[0] #us
    ftTime     = intTimes[1] #ns
    riseCell   = us2cell(dtimes[0], riseTime)
    ftCell   = us2cell(dtimes[0], ftTime)
    bufferCell = 2*riseCell + ftCell
    trapArr = np.zeros([np.shape(values)[0], np.shape(values)[1]-bufferCell])
    for m in tqdm(range(np.shape(values)[0]), 
                  desc   ="Running Trap Filter...........", 
                  colour = terminalCMAP[1]):
        for i in range(np.shape(values)[1]-bufferCell):
            leftInt  = np.mean(values[m, i:i+riseCell])
            rightInt = np.mean(values[m, i+riseCell+ftCell:i+riseCell+riseCell+ftCell])
            trapArr[m, i] = rightInt-leftInt
    return trapArr
###########################################################################################
def trapENS(times, values, dtimes, intTimes = (1000, 5000)):
    """
    ***MATRIX ACCELERATED***
    """
    riseTime   = intTimes[0] #us
    ftTime     = intTimes[1] #ns
    riseCell   = us2cell(dtimes[0], riseTime)
    ftCell   = us2cell(dtimes[0], ftTime)
    bufferCell = 2*riseCell + ftCell

    wfdLen = values.shape[1]
    row = np.zeros(wfdLen)
    row[:riseCell] = -1
    row[(riseCell+ftCell):(bufferCell)] = 1

    tensor = np.tile(row, (wfdLen, 1))
    for i in range(wfdLen):
        tensor[i] = np.roll(tensor[i], i)
    
    tensor = np.triu(tensor, -1)

    trapArr = np.zeros([np.shape(values)[0], np.shape(values)[1]-bufferCell])
    for w in tqdm(range(values.shape[0]), 
                  desc   ="Running Trap Filter...........", 
                  colour = terminalCMAP[1]):
        trapArr[w, :] = np.dot(values[w], tensor.T)[:-bufferCell]/riseCell
    return trapArr

###########################################################################################

def DCRquantileCut(paramArr):
    for i in range(len(paramArr)):
        if paramArr[i].name == "/deltasCorrected":
            deltasCorr = paramArr[i]
    qt         = np.quantile(deltasCorr, (0.05, 0.99))
    keep       = (deltasCorr > qt[0])*(deltasCorr < qt[1])
    for n in range(len(paramArr)):
        paramArr[n] = np.array(paramArr[n])
        paramArr[n] = paramArr[n][keep]
    print(f"Removed NANs, and Outlier DCR, new shape is {paramArr[n].shape}")
    return paramArr

###########################################################################################

def normalizeDCR(dcrArr):
    meanDelta  = np.mean(dcrArr[:])
    sigDelta   = np.std(dcrArr[:], ddof = 1)
    normDCR = (dcrArr[:] - meanDelta)/((sigDelta))

    return normDCR

###########################################################################################

def Gauss(x, A, B):
    y = A*np.exp(-1*B*x**2)
    return y

def printDCRHist(normDCR, peak):
    binNum = 50
    
    hist = np.histogram(normDCR, binNum)
    histVals = hist[0]
    binEdges = hist[1]
    binWidth = binEdges[1]-binEdges[0]
    
    midBins = np.linspace(binEdges[0] + 0.5*binWidth, binEdges[-1] - 0.5*binWidth, binNum)
    popt, pcov = curve_fit(Gauss, midBins, histVals)
    
    print(f"DCRFit popt {popt}")
    plt.hist(normDCR, bins = binNum)
    plt.plot(midBins, Gauss(midBins, popt[0], popt[1]))
    plt.savefig(f"{plotPath}/{peak}normDCRDistribution.jpg", dpi=100)
    plt.cla()
    plt.clf()
    plt.close()
###########################################################################################
###########################################################################################

def EfromA(CurrentAmplitudeArray):
    """
    Function: Energy estimation method from integration of current amplitude curve.
    """
    Eest = np.trapz(CurrentAmplitudeArray, np.arange(len(CurrentAmplitudeArray)))
    return Eest

###########################################################################################
###########################################################################################