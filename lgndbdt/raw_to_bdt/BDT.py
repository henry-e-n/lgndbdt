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
from imblearn.over_sampling import SMOTENC


from extraction_utils.config        import *
from extraction_utils.h5Extract     import *
from ML_utils.BDTPrep       import *
from ML_utils.BDTTrain      import *
from ML_utils.plot_legacy   import summary_legacy
from ML_utils.Visualization import *
from ML_utils.MultiVarCorr import *

print("Finished Import")

import argparse                  # allows us to deal with arguments to main()
from argparse import RawTextHelpFormatter
# PARSER ARGUMENTS
parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter, 
                                     description = "If the data is already clean, set clean to false and fname to the clean data name")
parser.add_argument("learning_rate", type=float,
                    help="Learning Rate of BDT",
                    default = 0.07442318529884213, nargs='?')
parser.add_argument("num_leaves", type=int,
                    help="Number of Leaves of BDT",
                    default = 73, nargs='?')
parser.add_argument("max_bin", type=int,
                    help="Max Bins of BDT",
                    default = 542, nargs='?')                

args                 = parser.parse_args()
learning_rate        = args.learning_rate
num_leaves           = args.num_leaves
max_bin              = args.max_bin

def run_BDT():
    ###################################################################
    # Data Type Preparation
    ###################################################################

    filename        = f"{detName}_Clean_StandardAnalysis.lh5"
    fpath           = f"{savePath}"

    def getRaw(filename, fpath):
        file, names, paramArr = paramExtract(filename, fpath, False)
        dataDict = []
        dataArr = np.zeros((len(fname), paramArr[0].shape[0]))
        select = []
        counter = 0
        wfd = np.zeros(2, dtype = object)
        avse = np.zeros(paramArr[0].shape[0])
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
            if np.any(np.isin("/E", paramArr[i].name)):
                avse = paramArr[i][:]
        dataDictionary = dict(dataDict)
        selectDictionary = dict(select)
        dataArr = np.stack(dataArr, 1)
        print(f"Returned {fpath}{filename}")#, shape {dataArr.shape}")
        return dataArr, dataDictionary, wfd, avse, selectDictionary

    sigRaw, sigDict, sigWFD, sigavse, selectDict = getRaw(filename, f"{fpath}DEP/")
    bkgRaw, bkgDict, bkgWFD, bkgavse, selectDict = getRaw(filename, f"{fpath}FEP/")

    ###################################################################
    # DATA MATCHING
    ###################################################################
    print("--------------- Running Distribution Matching ---------------")
    print("-------------------------------------------------------------")

    sigSave, sigPDM = dataSplit(sigRaw, 0.3)
    bkgSave, bkgPDM = dataSplit(bkgRaw, 0.3)

    print(f"Signal Size before Distribution Matching {sigSave.shape}")
    for i in range(len(distMatch)):
        sigSave, bkgSave = match_data(sigSave, bkgSave, selectDict, distMatch[i], distStep[i], True, show = False)
    # sigSave, bkgSave = match_data(sigSave, bkgSave, selectDict, "/tdrift", 30, True, show = False)
    # sigSave, bkgSave = match_data(sigSave, bkgSave, selectDict, "/tdrift10", 20, True, show = False)
    # sigSave, bkgSave = match_data(sigSave, bkgSave, selectDict, "/tdrift50", 20, True, show = False)
    # sigSave, bkgSave = match_data(sigSave, bkgSave, selectDict, "/noise", 0.00002, True, show = False)
    # sigSave, bkgSave = match_data(sigSave, bkgSave, selectDict, "/noiseTail", 0.00002, True, show = False)
    # sigSave, bkgSave = match_data(sigSave, bkgSave, selectDict, "/LQ80", 10, True, show = False)
    print(f"Signal Size after Distribution Matching {sigSave.shape}")

    sigs = sigSave
    bkgs = bkgSave

    ###################################################################
    # TRAINING PREPARATION
    ###################################################################
    print("----------------------- Training Prep -----------------------")
    print("-------------------------------------------------------------")

    signalTrain, signalTest = dataSplit(sigs, 0.3)
    sigLabelTrain = np.ones(signalTrain.shape[0]) # Labels all training signals as signals (1)
    sigLabelTest = np.ones(signalTest.shape[0]) # Labels all testing signals as signals (1)

    bkgTrain, bkgTest = dataSplit(bkgs, 0.3)  # assigns arrays corresponding to randomly split signal data 
    bkgLabelTrain = np.zeros(bkgTrain.shape[0])
    bkgLabelTest = np.zeros(bkgTest.shape[0])

    # Combining then randomizing signal and background data for training

    xTrain = np.concatenate([signalTrain, bkgTrain], axis = 0) # combines input data
    yTrain = np.concatenate([sigLabelTrain, bkgLabelTrain], axis = 0) # combines label data

    trainIndex = np.arange(len(xTrain)) # Array assures same shuffling between input and label data
    np.random.shuffle(trainIndex)

    xTrain = xTrain[trainIndex]
    yTrain = yTrain[trainIndex]

    xTestStraight = np.concatenate([signalTest, bkgTest], axis = 0) # combines test input - no need to shuffle as it is for testing only
    yTestStraight = np.concatenate([sigLabelTest, bkgLabelTest], axis = 0) # combines test label

    sigTest = np.where(yTestStraight == 1)[0] # Array of signal indeces
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
    lgbEval = lgb.Dataset(xVal, yVal, reference=lgbTrain, free_raw_data=False, feature_name = list(fname))

    # Defines the hyperparameters of the BDT 
    params={"num_iterations": 2000, "learning_rate": learning_rate,
            "num_leaves": num_leaves, "bagging_freq": 62, "min_data_in_leaf": 26,
            "drop_rate": 0.3299436689754462, "min_gain_to_split": 0.5355479139953352,
            "max_bin": max_bin, "boosting": "goss", "objective": "binary", "metric": "binary_logloss", "verbose": -100}
    evals_result = {}

    # Performs the training on the dataset
    gbm = lgb.train(params, 
                    lgbTrain,
                    feature_name=list(fname), 
                    valid_sets=lgbEval,
                    early_stopping_rounds=10,
                    evals_result=evals_result,
                    verbose_eval = 20) 

    explainer = shap.TreeExplainer(gbm)
    gbm.save_model('BDT_unblind.txt') # Saves the BDT model as txt file

    gbm = lgb.Booster(model_file='BDT_unblind.txt')  # init model
    TrainingMetric(evals_result) #####################

    ###################################################################
    # TREE VISUALIZATION
    ###################################################################
    TreeVis(gbm)#################

    ###################################################################
    # EVALUATION AND VISUALIZATION
    ###################################################################
    print("---------------- Evaluation and Visualization ---------------")
    print("-------------------------------------------------------------")

    ######################################

    for i in tqdm(range(11), 
                    desc   ="Running Visualization................", 
                    colour = terminalCMAP[1]):
        if i == 0:
            signalData = sigPDM
            bkgData = bkgPDM

            X_test = np.concatenate([signalData,bkgData],axis=0)
            Y_test = np.array([1]*len(signalData) + [0] * len(bkgData))
            params = {"num_iterations": 1, "learning_rate": 0.15967607193274216, "num_leaves": 688, "bagging_freq": 34, "bagging_fraction": 0.9411410478379901, "min_data_in_leaf": 54, "drop_rate": 0.030050388917525712, "min_gain_to_split": 0.24143821598351703, "max_bin": 454, "boosting": "dart", "objective": "binary", "metric": "binary_logloss", "verbose": -100}

            lgb_train = lgb.Dataset(X_test[:,:len(fname)], Y_test,free_raw_data=False, feature_name = list(fname))
            MSBDT = lgb.Booster(model_file='BDT_unblind.txt', silent=True)
            params["num_iterations"] = 1

            gbm = lgb.train(params, 
                            lgb_train) 

            MSBDTstr = MSBDT.model_to_string()
            explainer = shap.TreeExplainer(gbm.model_from_string(MSBDTstr))
            
            y_pred = gbm.predict(X_test[:,:len(fname)], num_iteration=gbm.best_iteration)

            BDTDistrib(y_pred, Y_test)
        elif i == 1:
            Pos_sample = X_test[Y_test == 1,:len(fname)]
            Neg_sample = X_test[Y_test == 0,:len(fname)]
            np.random.shuffle(Pos_sample)
            np.random.shuffle(Neg_sample)

            sample = np.concatenate([Pos_sample[:10000], Neg_sample[:10000]],axis=0)
            shap_values = explainer.shap_values(sample)
            # Returns a list of matrices (# outputs, # samples x, # features)
            BDTSummary(shap_values, sample)
        elif i == 2:
            shap_val = np.array(shap_values)[0]
            pcaRes, pcaNames = biVarCorr(shap_val, fname)
            printBVC(pcaRes, pcaNames)
            # pcaMat = multiVarCorr(shap_val, 2)
            # printMVC(pcaMat)
        elif i == 3:
            # Covariance Matrices
            # Define Outperforming events
            bdt_thresh = 0.55
            avse_thresh = 969 #-1 # How to set Cut
            explainer = shap.TreeExplainer(gbm)
            sample_sig = (y_pred>bdt_thresh) & (Y_test == 1) & (X_test[:,selectDict["/AvsE_c"]]<avse_thresh)# & cselector
            # Get Sig Outperforming SHAP
            shap_sig = explainer.shap_values(X_test[sample_sig,:len(fname)])
            # Get BDT and AvsE score 
            outSigBDT = y_pred[sample_sig]
            outSigAvsE = X_test[sample_sig,selectDict["/AvsE_c"]]
            # Transform SHAP to array
            shap_sigArr = np.array(shap_sig[0], dtype=float)
            # Add BDT
            shap_sigArr = np.insert(shap_sigArr, -1, outSigBDT, axis=1)
            # Add AvsE
            shap_sigArr = np.insert(shap_sigArr, -1, outSigAvsE, axis=1)
            covName = np.append(fname, ["BDT", "A/E"])
            covSIG = np.corrcoef(shap_sigArr.T)
            plot_covariance(covSIG, "Signal Covariance", covName)

        elif i == 4:
            sample_bkg = (y_pred<bdt_thresh) & (Y_test == 0) & (X_test[:,selectDict["/AvsE_c"]]>avse_thresh)# & cselector
            shap_bkg = explainer.shap_values(X_test[sample_bkg,:len(fname)])
            outBkgBDT = y_pred[sample_bkg]
            outBkgAvsE = X_test[sample_bkg,selectDict["/AvsE_c"]]
            shap_bkgArr = np.array(shap_bkg[0], dtype=float)
            shap_bkgArr = np.insert(shap_bkgArr, -1, outBkgBDT, axis=1)
            shap_bkgArr = np.insert(shap_bkgArr, -1, outBkgAvsE, axis=1)
            covBKG = np.corrcoef(shap_bkgArr.T)
            plot_covariance(covBKG, "Background Covariance", covName)

        elif i == 5:
            # sigsave = sigRaw
            # bkgsave = bkgRaw

            # bdt_thresh = 0.55
            # avse_thresh = 969 #-1 # How to set Cut
            # explainer = shap.TreeExplainer(gbm)

            # sample_selector1 = (y_pred>bdt_thresh) & (Y_test == 1) & (X_test[:,selectDict["/AvsE_c"]]<avse_thresh)# & cselector
            # sample_selector2 = (y_pred<bdt_thresh) & (Y_test == 0) & (X_test[:,selectDict["/AvsE_c"]]>avse_thresh)# & cselector
            sample_selector = sample_sig|sample_bkg
            evnew = X_test[sample_selector,:len(fname)]
            np.random.shuffle(evnew)
            evnew = evnew[:10000]
            shap_valuesDist = explainer.shap_values(evnew)
            make_dist_plot(evnew,shap_valuesDist[1],selectDict, "/tdrift10", "/AvsE_c")
        elif i == 6:
            make_dist_plot(evnew,shap_valuesDist[1],selectDict, "/tdrift", "/AvsE_c")
        elif i == 7:
            make_dist_plot(evnew,shap_valuesDist[1],selectDict, "/tdrift50", "/AvsE_c"),
        elif i == 8:
            make_dist_plot(evnew,shap_valuesDist[1],selectDict, "/tdrift", "/AvsE_c", point=True),
        elif i == 9:
            index = 0
            ROIdata = evnew #X_test[sample_selector]

            ROIdata = ROIdata[ROIdata[:,selectDict["/tdrift"]] < 600]
            sample=ROIdata[index,:len(fname)].reshape(1,-1)
            shap_values = explainer.shap_values(sample)
            plot_SHAP_force(explainer, shap_values[1][0])
        elif i == 10:
            plot_ROC(sigavse, bkgavse, Y_test, y_pred, sigRaw, bkgRaw, selectDict)
    return

if __name__ == "__main__":
    run_BDT()