import numpy as np


# Function that splits data randomly with specified split proportion
def dataSplit(data, testSplit = 0.3):
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    
    trainIndex = indices[int(len(indices)*testSplit):] # takes last 70% of shuffled indices for training
    testIndex = indices[:int(len(indices)*testSplit)] # takes first 30% of shuffled indices for testing

    dataTrain = data[trainIndex] # assigns arrays corresponding to randomly split signal data 
    dataTest = data[testIndex]
    
    return dataTrain, dataTest

