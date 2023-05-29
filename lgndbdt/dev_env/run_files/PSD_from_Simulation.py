# This function takes in an array of waveforms as simulated by Julia (or other software)
# And returns a file that can be fed into the PSD_extraction pipeline

# PSD extraction typically requires a paramArr with the 

import numpy as np
from lgndbdt.dev_env.extraction_utilities.DCR import find_percent


def sim_to_param(event_array):
    times  = event_array[:,0]
    values = event_array[:,1]
    num_events = np.shape(event_array[0,:])[0]

    
    dt_array = np.ones(num_events)*(times[0][1]-times[0][0])
    t0_array = np.zeros(num_events)
    
    tp_0 = []
    for ev in range(num_events):
        baseline = np.mean(values[ev][1:100])
        ind = find_percent(values[ev], percent = .1)
        tp_0 = np.append(tp_0, times[ev][ind])
    tp_0_array = np.array(tp_0)
    
    trapEmax_array = np.amax(values, 1)
    
    paramArr = [trapEmax_array, tp_0_array, t0_array, dt_array, values]
    paramStrings = ["trapEmax", "tp_0", "t0", "dt", "values"]
    paramDict = dict(zip(paramStrings, paramArr))
    return paramArr, paramDict