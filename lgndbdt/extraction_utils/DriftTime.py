
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from extraction_utils.config import *
from tqdm import tqdm
import os
import sys
import scipy.optimize as spy
cwd = os.getcwd()
module_path = os.path.abspath(os.path.split(cwd)[0])
sys.path.append(module_path)


# from extraction_utils.h5Extract import *
from extraction_utils.waveform import *
from extraction_utils.AvsE import *
from extraction_utils.DCR import *

from scipy.interpolate import interp1d

def findHPoint(values, startCell, hPoint):
    """
    Function: Finds the index of the cell closest to the 'hPoint' along rising edge

    Parameters:
        - values: Waveform ADC values
        - startCell: Cell of start of the waveform, as determined by the front end electronics
        - hPoint: Fraction of max of waveform (0, 1)
    
    Returns:
        - closest: 
    """
    peak = np.max(values)
    peakInd = values.argmax()
    baseOfPeak = values[startCell]
    
    val97 = baseOfPeak +(hPoint*(peak-baseOfPeak))
    diffArr = np.absolute(values - val97)
    closestToPeak = diffArr[:peakInd].argmin()
    closest = closestToPeak
    return closest

def getTDrift(values, startTime, dtimes):
    """
    Function: Find the times of the 10%, 50%, and 99% points along the rising edge of each waveform

    Parameters:
        - values:
        - startTime:
        - dtimes

    Returns:
        - tdrift, tdrift50, tdrift 10
    """
    tdrift = np.zeros(values.shape[0])
    tdrift50 = np.zeros(values.shape[0])
    tdrift10 = np.zeros(values.shape[0])
    
    for i in range(values.shape[0]):
        startCell = int(startTime[i]/dtimes[i])
        cell99 = findHPoint(values[i,:], startCell, 0.999)
        cell50 = findHPoint(values[i,:], startCell, 0.50)
        cell10 = findHPoint(values[i,:], startCell, 0.10)

        tdrift[i] = (cell99-startCell)*dtimes[i]
        tdrift50[i] = (cell50-startCell)*dtimes[i]
        tdrift10[i] = (cell10-startCell)*dtimes[i]
    
    return tdrift, tdrift50, tdrift10




def getTDriftInterpolate(times, values, startTime, dtimes):
    tdrift = np.zeros(values.shape[0])
    tdrift50 = np.zeros(values.shape[0])
    tdrift10 = np.zeros(values.shape[0])

    for i in range(values.shape[0]):
        f1 = interp1d(times[i, :], values[i, :], fill_value="extrapolate")
        
        startCell = int(startTime[i]/dtimes[i])
        
        # base = startTime[i] 
        # top = times[i, np.argmax(values[i, :])]
        # testX = np.linspace(base, top, 1000)
        # testY = f1(testX)
        def getClosest(perc):
            baseOfPeak = values[i, startCell]
            base = startTime[i]
            top = times[i, np.argmax(values[i, :])] + 10
            testX = np.linspace(base, top, 20000)
            testY = f1(testX)
            percVal = perc*(np.max(values[i, :]) - baseOfPeak) + baseOfPeak
            diff = np.abs(testY - percVal)
            closest = np.argmin(diff)
            return testX[closest]

        tdrift[i] = getClosest(0.999)-startTime[i]
        tdrift50[i] = getClosest(0.5)-startTime[i]
        tdrift10[i] = getClosest(0.15)-startTime[i]
    return tdrift, tdrift50, tdrift10

# plt.plot(times[i, :], values[i, :], '.')
# plt.plot(times[i, :], f1(times[i, :]), '--')
# plt.hlines(perc50, times[i, 1], times[i, -1])
# plt.show()
# print(testX[closest])