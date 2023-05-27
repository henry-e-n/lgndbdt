import numpy as np
import tqdm
from utilities.global_config import terminalCMAP

def AvsE(values, dtimes, plots = [], numWF = 2500, currentEstimator = 100):
    """
    Arguments:
        - ntimes array - [#Waveforms, #TimeCells]
        - values - [#Waveforms, #TimeCells] - choose one waveform
        - dtimes - [#Waveforms, #TimeCells]
    
    Plots should either be:
        - False - no plots to be printed
        - E - energy array same length as times
    """
    dWindow = int(currentEstimator/dtimes[0])
    wfdLen = values.shape[1]
    row = np.zeros(wfdLen)
    row[0] = -1
    row[dWindow] = 1
    tensor = np.tile(row, (wfdLen, 1))
    for i in range(wfdLen):
        tensor[i] = np.roll(tensor[i], i)
    
    tensor = np.triu(tensor, -1)
    maxA = np.zeros(numWF)
    # plt.imshow(tensor[:100, :100])
    # plt.show()
    for w in tqdm(range(numWF),
                  desc = "Getting Current Amplitude.....",
                  colour = terminalCMAP[1]):
        maxA[w] = np.max(np.dot(values[w, :], tensor.T))
    return maxA