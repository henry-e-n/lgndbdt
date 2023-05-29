import lightgbm as lgb
import numpy    as np
import os
# import sys
# import gc
# import pandas as pd
import shap
import matplotlib.pyplot as plt
# from matplotlib import cm
from sklearn.preprocessing import StandardScaler, RobustScaler

from tqdm import tqdm
from imblearn.over_sampling import SMOTE

from utilities.get_files import get_save_paths
from utilities.h5_utils  import paramExtract
from BDT_utilities.Visualization import *
from BDT_utilities.train_functions import split_data, match_data
print("Finished Import")

learning_rate        = 0.07442318529884213 #args.learning_rate
num_leaves           = 73 #args.num_leaves
max_bin              = 542 #args.max_bin

randSeed = 27
np.random.seed(randSeed)

def BDT_train(detector_name, target_peak, source_location, train_features, match_features, match_step, bdt_thresh = 0.55, avse_thresh = 969, split_ratio = 0.3, validate="split", augment = True, plots=True):
    top_file_save_path, top_plot_save_path = get_save_paths(detector_name, "top")
    side_file_save_path, side_plot_save_path = get_save_paths(detector_name, "side")
    plot_save_path = top_plot_save_path

    # Validate = "Full" for validation on all data
    isExist = os.path.exists(f"{plot_save_path}/{source_location}/")
    if not isExist:
        os.makedirs(f"{plot_save_path}/{source_location}/")
        print(f"{plot_save_path}/{source_location}/ directory was created!")

    ###################################################################
    # Data Type Preparation
    ###################################################################

    filename        = f"{detector_name}_PSDs_"

    def getRaw(filename, fpath):
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

    def augment_ICPC(top_data, side_data):
        yTest = np.array([1]*len(top_data) + [0]*len(side_data))  # Array of detector types
        xTest = np.concatenate((top_data, side_data), axis=0) # Removes detector type from input array
        
        smICPC = SMOTE(k_neighbors = 10)
        xRes, yRes = smICPC.fit_resample(xTest, yTest)
        top_augmented = xRes[yRes==1]
        side_augmented = xRes[yRes==0]
        return top_augmented, side_augmented

    
    
    sigRAWTop, selectDict = getRaw(f"{filename}DEP_top.lh5", f"{top_file_save_path}")
    bkgRAWTop, selectDict = getRaw(f"{filename}{target_peak}_top.lh5", f"{top_file_save_path}")
    sigRAWSide, selectDict = getRaw(f"{filename}DEP_side.lh5", f"{side_file_save_path}")
    bkgRAWSide, selectDict = getRaw(f"{filename}{target_peak}_side.lh5", f"{side_file_save_path}")
    
    def scaleData(signalRAW, backgroundRAW):
        scaler = RobustScaler()
        sigbkgRAW = scaler.fit_transform(np.concatenate((signalRAW, backgroundRAW)))
        signalRAW = sigbkgRAW[:len(signalRAW)]
        backgroundRAW = sigbkgRAW[len(signalRAW):]
        return signalRAW, backgroundRAW
    
    print(sigRAWTop.shape, bkgRAWTop.shape, sigRAWSide.shape, bkgRAWSide.shape)
    
#     sigRAWSS, bkgRAWSS = scaleData(np.concatenate((sigRAWTop, sigRAWSide)), np.concatenate((bkgRAWTop, bkgRAWSide)))
#     sigRAWTop, sigRAWSide = [sigRAWSS[:len(sigRAWTop), :], sigRAWSS[len(sigRAWTop):, :]]
#     bkgRAWTop, bkgRAWSide = [bkgRAWSS[:len(bkgRAWTop), :], bkgRAWSS[len(bkgRAWTop):, :]]
    
    print(sigRAWTop.shape, bkgRAWTop.shape, sigRAWSide.shape, bkgRAWSide.shape)
    
    if source_location == "mix":
        print(f"Runs include a mix of data from source location on the top, and on the side\nTop Data Size (sig, bkg) {sigRAWTop.shape}, {bkgRAWTop.shape}\nSide Data Size (sig, bkg) {sigRAWSide.shape}, {bkgRAWSide.shape}")
        sigRAW = np.concatenate((sigRAWTop, sigRAWSide))
        bkgRAW = np.concatenate((bkgRAWTop, bkgRAWSide))
        if augment:
            sigTopAug, sigSideAug = augment_ICPC(sigRAWTop, sigRAWSide)
            bkgTopAug, bkgSideAug = augment_ICPC(bkgRAWTop, bkgRAWSide)
            sigAUG = np.concatenate((sigTopAug, sigSideAug))
            bkgAUG = np.concatenate((bkgTopAug, bkgSideAug))
            print(f"Augmenting Side dataset - original sig shape: {sigRAWSide.shape}, augmented sig shape: {sigSideAug.shape}")
            print(f"Augmenting Side dataset - original bkg shape: {bkgRAWSide.shape}, augmented bkg shape: {bkgSideAug.shape}")
        else:
            sigAUG = sigRAW
            bkgAUG = bkgRAW
        for dictKey in selectDict.keys():
            sourceLoc_distCheck(sigRAWTop, bkgRAWTop, sigRAWSide, bkgRAWSide, selectDict, dictKey)
            plt.title(f"{dictKey} Distribution")
            plt.savefig(f"{plot_save_path}/{source_location}/{dictKey}DistributionBoxPlot.pdf",dpi=300, transparent=True)
    
    elif source_location == "side":
        print("Source Loc Side")
        if augment:
            print("AUGMENT")
            sigTopAug, sigSideAug = augment_ICPC(sigRAWTop, sigRAWSide)
            bkgTopAug, bkgSideAug = augment_ICPC(bkgRAWTop, bkgRAWSide)
            sigAUG = sigSideAug
            bkgAUG = bkgSideAug

            sigRAW = sigRAWSide
            bkgRAW = bkgRAWSide
            print(f"Augmenting Side dataset - original sig shape: {sigRAWSide.shape}, augmented sig shape: {sigSideAug.shape}")
            print(f"Augmenting Side dataset - original bkg shape: {bkgRAWSide.shape}, augmented bkg shape: {bkgSideAug.shape}")
        else:
            sigRAW = sigRAWSide
            bkgRAW = bkgRAWSide
            sigAUG = sigRAW
            bkgAUG = bkgRAW
        
    if source_location == "top":
        print("Source Loc Top")
        sigRAW = np.copy(sigRAWTop)
        bkgRAW = np.copy(bkgRAWTop)
        sigAUG = np.copy(sigRAW)
        bkgAUG = np.copy(bkgRAW)
        print(sigRAW-sigAUG)
    ###################################################################
    # DATA MATCHING
    ###################################################################
    print("--------------- Running Distribution Matching ---------------")
    print("-------------------------------------------------------------")
    
    # sigSave, sigPDM = split_data(sigRAW, split_ratio)
    # bkgSave, bkgPDM = split_data(bkgRAW, split_ratio)

    # print(f"Incoming dataset size \n \
    #         SS shape {len(sigRAW)} - Split to {len(sigSave), len(sigPDM)} \n \
    #         MS shape {len(bkgRAW)} - Split to {len(bkgSave), len(bkgPDM)}")

    sigSave = sigAUG
    bkgSave = bkgAUG
    
    print(f"Size before Distribution Matching Signal: {sigSave.shape}, Background: {bkgSave.shape}")
    for i in range(len(match_features)):
        print(f"Distribution Matching {match_features[i]}")
        sigSave, bkgSave = match_data(sigSave, bkgSave, selectDict, match_features[i], match_step[i], plots, plotPath = plot_save_path, show = False)
    print(f"Size after Distribution Matching Signal: {sigSave.shape}, Background: {bkgSave.shape}")

    sigs = sigSave
    bkgs = bkgSave

    ###################################################################
    # TRAINING PREPARATION
    ###################################################################
    print("----------------------- Training Prep -----------------------")
    print("-------------------------------------------------------------")

    signalTrain, signalTest = split_data(sigs, split_ratio)
    sigLabelTrain           = np.ones(signalTrain.shape[0]) # Labels all training signals as signals (1)
    sigLabelTest            = np.ones(signalTest.shape[0]) # Labels all testing signals as signals (1)

    bkgTrain, bkgTest = split_data(bkgs, split_ratio)  # assigns arrays corresponding to randomly split signal data 
    bkgLabelTrain     = np.zeros(bkgTrain.shape[0])
    bkgLabelTest      = np.zeros(bkgTest.shape[0])

    # Combining then randomizing signal and background data for training

    xTrain = np.concatenate([signalTrain, bkgTrain], axis = 0) # combines input data
    yTrain = np.concatenate([sigLabelTrain, bkgLabelTrain], axis = 0) # combines label data

    trainIndex = np.arange(len(xTrain)) # Array assures same shuffling between input and label data
    np.random.shuffle(trainIndex)

    xTrain = xTrain[trainIndex]
    yTrain = yTrain[trainIndex]

    xTestStraight = np.concatenate([signalTest, bkgTest], axis = 0) # combines test input - no need to shuffle as it is for testing only
    yTestStraight = np.concatenate([sigLabelTest, bkgLabelTest], axis = 0) # combines test label

    sigTest   = np.where(yTestStraight == 1)[0] # Array of signal indeces
    backgTest = np.where(yTestStraight == 0)[0]

    minTestEntry = min(len(sigTest), len(backgTest)) # Takes lesser beween num of sig and num of bkg

    indexTest = list(np.random.choice(sigTest, minTestEntry, replace=False)) + list(np.random.choice(backgTest, minTestEntry, replace=False)) # list of random pull from sigTest, size of minEntryTest, list of random pull from background
    #np.random.shuffle(indexTest) # Randomizes the order of the test datasets 
    # This should do the same as the randomization cell but double check
    xTestShuffle = xTestStraight[indexTest]
    yTestShuffle = yTestStraight[indexTest]

    if len(xTestShuffle)%2==1: # If len of array is odd
        xTestShuffle = xTestShuffle[:-1] # only index to 2nd to last (making even)
        yTestShuffle = yTestShuffle[:-1]
        
    xVal, xTest = np.split(xTestShuffle, 2) # splits array evenly in 2 between validation and test set
    yVal, yTest = np.split(yTestShuffle, 2)

    ###################################################################
    # TRAINING
    ###################################################################
    print("------------------------- Training --------------------------")
    print("-------------------------------------------------------------")

    lgbTrain = lgb.Dataset(xTrain, yTrain, free_raw_data=False, feature_name = list(train_features))
    lgbEval  = lgb.Dataset(xVal, yVal, reference=lgbTrain, free_raw_data=False, feature_name = list(train_features))

    # Defines the hyperparameters of the BDT 
    params={"num_iterations": 2000, "learning_rate": learning_rate,
            "num_leaves": num_leaves, "bagging_freq": 62, "min_data_in_leaf": 26,
            "drop_rate": 0.3299436689754462, "min_gain_to_split": 0.5355479139953352,
            "max_bin": max_bin, "boosting": "goss", "objective": "binary", "metric": "binary_logloss", "verbose": -100}
    evals_result = {}

    # Performs the training on the dataset
    gbm = lgb.train(params, 
                    lgbTrain,
                    feature_name = list(train_features), 
                    valid_sets   = lgbEval,
                    callbacks    = [lgb.early_stopping(10), lgb.log_evaluation(20), lgb.record_evaluation(evals_result)])

    explainer = shap.TreeExplainer(gbm)
    gbm.save_model('BDT_unblind.txt') # Saves the BDT model as txt file

    gbm = lgb.Booster(model_file='BDT_unblind.txt')  # init model
    if plots:
        TrainingMetric(evals_result) #####################
        plt.savefig(f"{plot_save_path}/TrainingMetric.pdf", dpi=100, transparent=True)

    return