import numpy as np


def us2cell(dtimes , ns = 1000):
    """
    Function: Takes in time (in ns) to be changed to cell width (based on dt) 

    Parameters:
        - dtimes: time cell deltas in ns (i think)
    
    Returns:
        - number of cells corresponding to time
    """
    return int(ns/dtimes)

def box_average(values, index, side, dtimes, boxWidth = 1):
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

def find_slope_corr(values, valuesCorrected, dtimes):
    """
    Function: Finds the slope of the trailing tail of the Waveform after P0 correction

    Parameters:
        - values: Waveform ADC values
        - values: P0 corrected waveform ADC values
        - dtimes: time cell deltas

    Returns:
        - delta: Estimated Slope of tail    
    """
    avg1, left1 = box_average(valuesCorrected, find97(values), 'left', dtimes)
    avg2, left2 = box_average(valuesCorrected, -1, 'right', dtimes)
    
    delta = (avg2 - avg1)/(left2-left1)
    return delta


def Normalize_Waveforms(values):
    newVals = np.zeros_like(values)
    for i in range(len(values)):
        maxVal = np.max(values[i, :])
        newVals[i, :] = values[i,:]/maxVal
    
    return newVals