import lightgbm as lgb
import numpy as np
import os
import sys
import gc
import pandas as pd
import shap
import matplotlib.pyplot as plt
from matplotlib import cm

from tqdm import tqdm
from time import time
from imblearn.over_sampling import SMOTE


from lgndbdt.extraction_utils.config        import *
from lgndbdt.extraction_utils.h5Extract     import *
from lgndbdt.ML_utils.BDTPrep       import *
from lgndbdt.ML_utils.BDTTrain      import *
from lgndbdt.ML_utils.plot_legacy   import summary_legacy
from lgndbdt.ML_utils.Visualization import *
from lgndbdt.ML_utils.MultiVarCorr import *

print("Finished Import")

learning_rate        = 0.07442318529884213 #args.learning_rate
num_leaves           = 73 #args.num_leaves
max_bin              = 542 #args.max_bin

randSeed = 27
np.random.seed(randSeed)

def run_BDT(bdt_thresh = 0.55, avse_thresh = 969, SEPorFEP="SEP", sourceLoc = "top", validate="split", augment = True, plots=True):
    # Validate = "Full" for validation on all data
    ###################################################################
    # Data Type Preparation
    ###################################################################

    filename        = f"{detName}_PSDs_{targetPeak[:-3]}"
    fpath           = f"{psdPath}"

    print(filename)
    print(fpath)

    def getRaw(filename, fpath):
        file, names, paramArr = paramExtract(filename, fpath, False)
        dataDict = []
        dataArr = np.zeros((len(fname), paramArr[0].shape[0]))
        select = []
        counter = 0
        wfd = np.zeros(2, dtype = object)
        for i in range(len(paramArr)):
            if np.any(np.isin(fname, paramArr[i].name)):
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

    
    
    sigRAWTop, selectDict = getRaw(f"{filename}topDEP.lh5", f"{fpath}")
    bkgRAWTop, selectDict = getRaw(f"{filename}top{SEPorFEP}.lh5", f"{fpath}")
    sigRAWSide, selectDict = getRaw(f"{filename}sideDEP.lh5", f"{fpath}")
    bkgRAWSide, selectDict = getRaw(f"{filename}side{SEPorFEP}.lh5", f"{fpath}")
        
    if sourceLoc == "mix":
        print(f"Runs include a mix of data from source location on the top, and on the side\nTop Data Size (sig, bkg) {sigRAWTop.shape}, {bkgRAWTop.shape}\nSide Data Size (sig, bkg) {sigRAWSide.shape}, {bkgRAWSide.shape}")
        sigRAW = np.concatenate((sigRAWTop, sigRAWSide))
        bkgRAW = np.concatenate((bkgRAWTop, bkgRAWSide))

        if augment:
            sigTopAug, sigSideAug = augment_ICPC(sigRAWTop, sigRAWSide)
            bkgTopAug, bkgSideAug = augment_ICPC(bkgRAWTop, bkgRAWSide)
            sigAUG = np.concatenate((sigTopAug, sigSideAug))
            bkgAUG = np.concatenate((bkgTopAug, bkgSideAug))
        else:
            sigAUG = sigRAW
            bkgAUG = bkgRAW
        for dictKey in selectDict.keys():
            sourceLoc_distCheck(sigRAWTop, bkgRAWTop, sigRAWSide, bkgRAWSide, selectDict, dictKey)
        
    elif sourceLoc == "side":
        if augment:
            sigTopAug, sigSideAug = augment_ICPC(sigRAWTop, sigRAWSide)
            bkgTopAug, bkgSideAug = augment_ICPC(bkgRAWTop, bkgRAWSide)
            sigAUG = sigSideAug
            bkgAUG = bkgSideAug
            sigRAW = sigSideAug
            bkgRAW = bkgSideAug
        else:
            sigRAW = sigRAWSide
            bkgRAW = bkgRAWSide
            sigAUG = sigRAW
            bkgAUG = bkgRAW
        
    elif sourceLoc == "top":
        sigRAW = sigRAWTop
        bkgRAW = bkgRAWTop
        sigAUG = sigRAW
        bkgAUG = bkgRAW

    ###################################################################
    # DATA MATCHING
    ###################################################################
    print("--------------- Running Distribution Matching ---------------")
    print("-------------------------------------------------------------")

    sigSave, sigPDM = dataSplit(sigRAW, 0.3)
    bkgSave, bkgPDM = dataSplit(bkgRAW, 0.3)


    sigSave, sigAUGPDM = dataSplit(sigAUG, 0.3)
    bkgSave, bkgAUGPDM = dataSplit(bkgAUG, 0.3)

    print(f"Size before Distribution Matching Signal: {sigSave.shape}, Background: {bkgSave.shape}")
    for i in range(len(distMatch)):
        print(f"Distribution Matching {distMatch[i]}")
        sigSave, bkgSave = match_data(sigSave, bkgSave, selectDict, distMatch[i], distStep[i], plots, show = False)
    print(f"Size after Distribution Matching Signal: {sigSave.shape}, Background: {bkgSave.shape}")

    sigs = sigSave
    bkgs = bkgSave

    ###################################################################
    # TRAINING PREPARATION
    ###################################################################
    print("----------------------- Training Prep -----------------------")
    print("-------------------------------------------------------------")

    signalTrain, signalTest = dataSplit(sigs, 0.3)
    sigLabelTrain           = np.ones(signalTrain.shape[0]) # Labels all training signals as signals (1)
    sigLabelTest            = np.ones(signalTest.shape[0]) # Labels all testing signals as signals (1)

    bkgTrain, bkgTest = dataSplit(bkgs, 0.3)  # assigns arrays corresponding to randomly split signal data 
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

    lgbTrain = lgb.Dataset(xTrain, yTrain, free_raw_data=False, feature_name = list(fname))
    lgbEval  = lgb.Dataset(xVal, yVal, reference=lgbTrain, free_raw_data=False, feature_name = list(fname))

    # Defines the hyperparameters of the BDT 
    params={"num_iterations": 2000, "learning_rate": learning_rate,
            "num_leaves": num_leaves, "bagging_freq": 62, "min_data_in_leaf": 26,
            "drop_rate": 0.3299436689754462, "min_gain_to_split": 0.5355479139953352,
            "max_bin": max_bin, "boosting": "goss", "objective": "binary", "metric": "binary_logloss", "verbose": -100}
    evals_result = {}

    # Performs the training on the dataset
    gbm = lgb.train(params, 
                    lgbTrain,
                    feature_name = list(fname), 
                    valid_sets   = lgbEval,
                    callbacks    = [lgb.early_stopping(10), lgb.log_evaluation(20), lgb.record_evaluation(evals_result)])

    explainer = shap.TreeExplainer(gbm)
    gbm.save_model('BDT_unblind.txt') # Saves the BDT model as txt file

    gbm = lgb.Booster(model_file='BDT_unblind.txt')  # init model
    if plots:
        TrainingMetric(evals_result) #####################

    ###################################################################
    # EVALUATION AND VISUALIZATION
    ###################################################################
    print("---------------- Evaluation and Visualization ---------------")
    print("-------------------------------------------------------------")

    ######################################
    if plots:
        for i in tqdm(range(5), 
                        desc   ="Running Visualization................", 
                        colour = terminalCMAP[1]):
            if i == 0:
                if validate=="Full":
                    minSize = np.min([sigRAW.shape[0], bkgRAW.shape[0]])
                    np.random.shuffle(sigRAW)
                    np.random.shuffle(bkgRAW)           
                    signalData   = sigRAW[:minSize, :]
                    bkgData      = bkgRAW[:minSize, :]
                else:
                    # Using split raw data
                    minSize = np.min([sigPDM.shape[0], bkgPDM.shape[0]])
                    np.random.shuffle(sigPDM)
                    np.random.shuffle(bkgPDM)
                    signalData   = sigPDM[:minSize, :]
                    bkgData      = bkgPDM[:minSize, :]


                X_test = np.concatenate([signalData,bkgData], axis=0)
                Y_test = np.array([1]*len(signalData) + [0] * len(bkgData))
                params = {"num_iterations": 1, "learning_rate": 0.15967607193274216, "num_leaves": 688, "bagging_freq": 34, "bagging_fraction": 0.9411410478379901, "min_data_in_leaf": 54, "drop_rate": 0.030050388917525712, "min_gain_to_split": 0.24143821598351703, "max_bin": 454, "boosting": "dart", "objective": "binary", "metric": "binary_logloss", "verbose": -1}

                lgb_train = lgb.Dataset(X_test[:,:len(fname)], Y_test,free_raw_data=False, feature_name = list(fname))
                MSBDT     = lgb.Booster(model_file='BDT_unblind.txt')
                # params["num_iterations"] = 1

                gbm = lgb.train(params, 
                                lgb_train) 

                MSBDTstr  = MSBDT.model_to_string()
                explainer = shap.TreeExplainer(gbm.model_from_string(MSBDTstr))
                
                y_pred = gbm.predict(X_test[:,:len(fname)], num_iteration=gbm.best_iteration)
                np.save("Y_test.npy", Y_test)
                np.save("Y_pred.npy", y_pred)

                BDTDistrib(y_pred, Y_test)
                plt.cla()
                plt.clf()
                plt.close()
            elif i == 1:
                Pos_sample = X_test[Y_test == 1,:len(fname)]
                Neg_sample = X_test[Y_test == 0,:len(fname)]
                np.random.shuffle(Pos_sample)
                np.random.shuffle(Neg_sample)

                sample = np.concatenate([Pos_sample[:10000], Neg_sample[:10000]],axis=0)
                shap_values = explainer.shap_values(sample)
                # Returns a list of matrices (# outputs, # samples x, # features)
                BDTSummary(shap_values, sample)
            elif i == 2 and np.any(np.isin("/AvsE_c", fname)):
                explainer  = shap.TreeExplainer(gbm)
                sample_sig = (y_pred>bdt_thresh) & (Y_test == 1) & (X_test[:,selectDict["/AvsE_c"]]<avse_thresh)# & cselector
                sample_bkg  = (y_pred<bdt_thresh) & (Y_test == 0) & (X_test[:,selectDict["/AvsE_c"]]>avse_thresh)# & cselector

                sample_selector = sample_sig|sample_bkg
                evnew = X_test[sample_selector,:len(fname)]
                np.random.shuffle(evnew)
                evnew = evnew[:10000]
                shap_valuesDist = explainer.shap_values(evnew)
                make_dist_plot(evnew,shap_valuesDist[1],selectDict, "/tdrift10", "/AvsE_c")
                make_dist_plot(evnew,shap_valuesDist[1],selectDict, "/tdrift", "/AvsE_c")
                make_dist_plot(evnew,shap_valuesDist[1],selectDict, "/tdrift50", "/AvsE_c"),
                make_dist_plot(evnew,shap_valuesDist[1],selectDict, "/tdrift", "/AvsE_c", point=True),
            elif i == 3 and np.any(np.isin("/AvsE_c", fname)):
                index = 0
                ROIdata = evnew
                ROIdata     = ROIdata[ROIdata[:,selectDict["/tdrift"]] < 600]
                sample      = ROIdata[index,:len(fname)].reshape(1,-1)
                shap_values = explainer.shap_values(sample)
                plot_SHAP_force(explainer, shap_values[1][0])
            elif i == 4:
                
                if sourceLoc == "mix":
                    sig_sideband_RawTop, selectDict = getRaw(f"{filename}topDEP_sideband.lh5", f"{fpath}")
                    bkg_sideband_RawTop, selectDict = getRaw(f"{filename}top{SEPorFEP}_sideband.lh5", f"{fpath}")
                    sig_sideband_RawSide, selectDict = getRaw(f"{filename}sideDEP_sideband.lh5", f"{fpath}")
                    bkg_sideband_RawSide, selectDict = getRaw(f"{filename}side{SEPorFEP}_sideband.lh5", f"{fpath}")
                    print(f"SIDEBAND DATA: Runs include a mix of data from source location on the top, and on the side\nTop Data Size (sig, bkg) {sig_sideband_RawTop.shape}, {bkg_sideband_RawTop.shape}\nSide Data Size (sig, bkg) {sig_sideband_RawSide.shape}, {bkg_sideband_RawSide.shape}")
                    sig_sideband_RAW = np.concatenate((sig_sideband_RawTop, sig_sideband_RawSide))
                    bkg_sideband_RAW = np.concatenate((bkg_sideband_RawTop, bkg_sideband_RawSide))
                else:
                    sig_sideband_RAW, selectDict = getRaw(f"{filename}{sourceLoc}DEP_sideband.lh5", f"{fpath}")
                    bkg_sideband_RAW, selectDict = getRaw(f"{filename}{sourceLoc}{SEPorFEP}_sideband.lh5", f"{fpath}")
                
                if validate=="Full":
                    sig_sideband_Ratio = sig_sideband_RAW
                    bkg_sideband_Ratio = bkg_sideband_RAW
                else:
                    sig_sideband_Save, sig_sideband_Ratio = dataSplit(sig_sideband_RAW, 0.3)
                    bkg_sideband_Save, bkg_sideband_Ratio = dataSplit(bkg_sideband_RAW, 0.3)

                np.random.shuffle(sig_sideband_Ratio)
                np.random.shuffle(bkg_sideband_Ratio)
                
                MSBDT     = lgb.Booster(model_file='BDT_unblind.txt')
                gbm = lgb.train(params, 
                                lgb_train) 
                MSBDTstr  = MSBDT.model_to_string()
                explainer = shap.TreeExplainer(gbm.model_from_string(MSBDTstr))
                
                sig_sideband_pred = gbm.predict(sig_sideband_Ratio, num_iteration=gbm.best_iteration)
                bkg_sideband_pred = gbm.predict(bkg_sideband_Ratio, num_iteration=gbm.best_iteration)
                
                if validate!="Full":
                    sig_sp_frac, sig_sideband_pred = dataSplit(sig_sideband_pred, 0.3)
                    bkg_sp_frac, bkg_sideband_pred = dataSplit(bkg_sideband_pred, 0.3)
                    

                result = list(filter(lambda x: "A_" in x, selectDict))
                
                if validate=="Full":
                    sigavse = sigRAW[:,selectDict[result[0]]]
                    bkgavse = bkgRAW[:,selectDict[result[0]]]
                else:
                    # small set validation
                    sigavse = sigPDM[:,selectDict[result[0]]]
                    bkgavse = bkgPDM[:,selectDict[result[0]]]
                
                side_pred = np.concatenate((sig_sideband_pred, bkg_sideband_pred))
                side_test = np.array([1]*len(sig_sideband_pred) + [0]*len(bkg_sideband_pred))
                BDTDistrib(y_pred, Y_test, side_pred, side_test)
                plt.title(f"BDT Distribution - {sourceLoc} data", fontsize = 40)
                plt.savefig(f"{plotPath}/BDT_{sourceLoc}_+Sideband_distribution.png",dpi=300, transparent=True)
                
                tpr, fpr = getROC_sideband(Y_test, y_pred, sig_sideband_pred, bkg_sideband_pred, sigavse, bkgavse)
                plt.title(f"ROC performance - {sourceLoc} data", fontsize = 40) #, fontsize = 24, pad = 15, fontstyle='italic')
                plt.savefig(f"{plotPath}/ROC_{sourceLoc}_sideband.png",dpi=300, transparent=False)
                plt.cla()
                plt.clf()
                plt.close()

    return
