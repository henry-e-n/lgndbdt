# This function takes in an array of waveforms as simulated by Julia (or other software)
# And returns a file that can be fed into the PSD_extraction pipeline

# PSD extraction typically requires a paramArr with the 

import numpy as np

def sim_to_param(event_array):
    times  = event_array[0,:]
    values = event_array[1,:]
    num_events = np.shape(event_array[0,:])[1]

    print(num_events)

    dt_array = np.ones(len(times[0][1]-times[0][1]))
    d0_array = np.zeros(num_events)

    print(dt_array, d0_array)
    return