import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTENC
from utilities.global_config import cmapNormal


############################################################################################################################
############################################################################################################################

# Function that splits data randomly with specified split proportion
def split_data(data, testSplit = 0.3):
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    
    trainIndex = indices[int(len(indices)*testSplit):] # takes last 70% of shuffled indices for training
    testIndex = indices[:int(len(indices)*testSplit)] # takes first 30% of shuffled indices for testing

    dataTrain = data[trainIndex] # assigns arrays corresponding to randomly split signal data 
    dataTest = data[testIndex]
    
    return dataTrain, dataTest


# Distribution Matching Function
def match_data(signalData, bkgData, selectDict, varname, increment, plots=False, plotPath = "", show = True):
    """
    This function executes a distribution matching algorithm
    Distribution matching is used to match the number and distribution of entries in the signal and background datasets for each variable
    This is necessary to ...
    This function accomplishes this by:
    1. Calculating the number of entries in each 'bin'
    2. Pulling from each set the minimum number of entries for each bin
        - Such that the dataset (signal or background) that has more entries for that particular bin will only keep the number of entries equal to that of the other dataset
    
    This function needs to be run for each variable that we need to match by feeding the output datasets from one run into the next
    """
    initShape = [np.shape(signalData), np.shape(bkgData)]
    if (len(signalData) == 0) or (len(bkgData) == 0):
        return signalData, bkgData
    sigIndex = []
    bkgIndex = []
    index = int(selectDict[varname]) # Finds the index of the variable name in the feature name list found in cell 2
    #min = max( min(signal[variable]), min(bkg[variable]) )

    minSelectVal = max(np.min(signalData[:, index]), np.min(bkgData[:, index])) # Takes the max between the minimum value of signal or background for the particular variable
    maxSelectVal = min(np.max(signalData[:, index]), np.max(bkgData[:, index])) # Takes the max between the minimum value of signal or background for the particular variable
    # print(minSelectVal, maxSelectVal)

    if varname == "/DCR":
        minSelectVal = max(minSelectVal, -3.0)#-30.0) # Sets min DCR value to -30
        maxSelectVal = min(maxSelectVal, 3.0)#10.0) # Set max DCR value to 10
        #print(f"Hit DCR Edge, min is {minSelectVal}, and max is {maxSelectVal}")
    elif varname == "/noise":
        minSelectVal = max(minSelectVal, 0.0) 
        maxSelectVal = min(maxSelectVal, 0.008)
    elif varname == "/noiseTail":
        minSelectVal = max(minSelectVal, 0.0) 
        maxSelectVal = min(maxSelectVal, 0.008) 
    elif varname == "/LQ80":
        minSelectVal = max(minSelectVal, 0.0) # Sets min noise value to -10
        maxSelectVal = min(maxSelectVal, 1500) # Set max noise value to 10    

    for entrySelectVal in np.arange(minSelectVal, maxSelectVal + increment, increment):
        # For each individual dx of the range 
        #   1. Find how many signals and backgrounds are in that dx
        #   2. Appends the lesser number of qualifying events between the two
        #       If one or both have no qualifying events, does not append
        # This effectively only keeps the overlap of the histograms. 
        # A for loop running over entire range of variable data with <increment> step size
        # Returns the indices where signal data is in Range with the current value of entrySelectVal
        sigInRange = np.where(np.logical_and(signalData[:, index]>=entrySelectVal, signalData[:, index]<entrySelectVal+increment))[0]
        bkgInRange = np.where(np.logical_and(bkgData[:, index]>=entrySelectVal, bkgData[:, index]<entrySelectVal+increment))[0]
        
        minEntry = min(len(sigInRange), len(bkgInRange)) # the minimum between the number of points in range for signal and bkg
        if minEntry == 0: # Why is this Necessary ###########
            continue
        
        # appends <minEntry> number of random entries from sigInRange to sigIndex (without repeat)
        sigIndex += list(np.random.choice(sigInRange, minEntry, replace=False)) 
        bkgIndex += list(np.random.choice(bkgInRange, minEntry, replace=False))
        
    rg = np.arange(max(minSelectVal, -10), maxSelectVal+2*increment, increment) # range from min value to max value +2*increment (with built in lower edge)


    # Plots 
    if plots == True:
        plt.figure()
        plt.hist(signalData[:,index], bins = rg, color=cmapNormal(0.2), linestyle="-", histtype="step", label="DEP Before Matching")
        plt.hist(bkgData[:,index], bins = rg, color=cmapNormal(0.8), linestyle="-", histtype="step", label="SEP Before Matching")
        plt.hist(signalData[sigIndex,index], bins = rg, color=cmapNormal(0.4), alpha=0.5, label="Distribution After Matching")
        plt.ylabel("Counts")
        # plt.xlabel(f"New # Waves {len(sigIndex)}")
        plt.xlabel(f"Parameter Value")
        if (np.mean(signalData[:, index])-2*np.std(signalData[:, index]))>-2:
            plt.xlim(np.mean(signalData[:, index])-2*np.std(signalData[:, index]), np.mean(signalData[:, index])+2*np.std(signalData[:, index]))
        else:
            plt.xlim(np.mean(signalData[:, index])-2*np.std(signalData[:, index]), np.mean(signalData[:, index])+2*np.std(signalData[:, index]))
            # plt.xlim(-2, 2) # With standard scaler on
        plt.title(f"{varname}")
        lgd = plt.legend(bbox_to_anchor=(0.85, 0.75), loc = "lower center")
        plt.savefig(f"{plotPath}/{varname}DataMatching.pdf", dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')
        # if show == True:
        #     plt.show(block=False)
        plt.cla()
        plt.clf()
        plt.close()
    
    # assigns the signalData and bkgData as parts of whole - defined by indices chosen above
    signalData = signalData[sigIndex]
    bkgData = bkgData[bkgIndex]
    #print(entrySelectVal, sigInRange, bkgInRange)
    finShape = [np.shape(signalData), np.shape(bkgData)]
    #print(f"Initial shape: {initShape}, final shape: {finShape}")

    return signalData, bkgData

############################################################################################################################

# Data Conversion Function
# (removing energy)

# def convert_data(inputArray, useLongCalibration=False):
#     inputs, trueParam, runs = np.split(inputArray, [-3, -1], axis = 1)
#     inputs = np.delete(inputs, selectCriteria.index("Final_Energy"), axis=1) # Deletes Final Energy Column
    
#     inputs[inputs[:,fname.index("channel")] % 2 == 1, fname.index("channel")] -= 1 # Changes odd channels to even - Duplicate of above??
#     #print(np.unique(inputs[:, fname.index("dettype")])) # Prints the unique detector types should be 1 or 2 for PPC and ICPC
#     runs = runs.flatten() # Not Used???
    
#     return np.concatenate([inputs, trueParam], axis=-1)

############################################################################################################################

# # Data Augmentation Function
# def augment_ICPC(xInput, fname):
#     yTest = xInput[:, fname.index("dettype")] # Array of detector types
#     xTest = np.delete(xInput, fname.index("dettype"), axis = -1) # Removes detector type from input array
    
#     smICPC = SMOTENC(categorical_features = [fname.index("channel")-1, fname.index("ds")], k_neighbors = 10)
#     xRes, yRes = smICPC.fit_resample(xTest, yTest)
#     xOut = np.insert(xRes, fname.index("dettype"), yRes, axis=-1)
    
#     return xOut    

############################################################################################################################

# Data Cleaning function
# Remove non-physical - infinite, nan etc
def remove_infinite(inputArr):
    numSample, numFeature = inputArr.shape
    finite = np.sum(np.isfinite(inputArr), axis=-1) # Sums up number of columns (aka features) with finite values
    finiteRows =  finite==numFeature # if number of finite features in the row == number of features - aka all features finite - is True
    return inputArr[finiteRows] # returns only rows with all finite values
    
############################################################################################################################


############################################################################################################################
from utilities.h5_utils  import paramExtract


def getRaw(filename, fpath, train_features):
        file, names, paramArr = paramExtract(filename, fpath, False)
        print(file)
        dataDict = []
        dataArr = np.zeros((len(train_features), paramArr[0].shape[0]))
        select = []
        counter = 0
        wfd = np.zeros(2, dtype = object)
        for i in range(len(paramArr)):
            if np.any(np.isin(train_features, paramArr[i].name)):
                dataDict.append([paramArr[i].name, paramArr[i][:]])
                dataArr[counter, :] = paramArr[i]
                select.append([paramArr[i].name, counter])
                counter += 1
            if np.any(np.isin("/times", paramArr[i].name)):
                wfd[0] = paramArr[i]
            if np.any(np.isin("/wfdCorr", paramArr[i].name)):
                wfd[1] = paramArr[i]
        dataDictionary = dict(dataDict)
        selectDictionary = dict(select)
        dataArr = np.stack(dataArr, 1)
        print(f"Returned {fpath}{filename}")
        file.close()
        return dataArr, selectDictionary