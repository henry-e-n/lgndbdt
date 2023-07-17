import numpy as np
from scipy.interpolate import interp1d

def get_TDRIFT_interpolate(times, values, startTime, dtimes):
    """
    Function: Determine the drift time values of a waveform.

    Parameters:
        - times: Times array.
        - values: Waveform values:
        - startTime: Start of the rise times:
        - dtimes: dt of time samples.
    
    Returns: 
        - tdrift
        - tdrift50
        - tdrift10
    """
    
    tdrift = np.zeros(values.shape[0])
    tdrift50 = np.zeros(values.shape[0])
    tdrift10 = np.zeros(values.shape[0])

    for i in range(values.shape[0]):
        f1 = interp1d(times, values[i, :], fill_value="extrapolate")
        
        startCell = int(startTime[i]/dtimes[i])
        
        # base = startTime[i] 
        # top = times[i, np.argmax(values[i, :])]
        # testX = np.linspace(base, top, 1000)
        # testY = f1(testX)
        def getClosest(perc):
            baseOfPeak = values[i, startCell]
            base = startTime[i]
            top = times[np.argmax(values[i, :])] + 10
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
