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
from extraction_utils.CleanData     import *
from ML_utils.plot_legacy   import summary_legacy
from ML_utils.Visualization import *
from ML_utils.MultiVarCorr import singVarCorr, multiVarCorr
from lgndbdt.ML_utils.MultiVarCorr import biVarCorr

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


    sigSave, sigPDM = dataSplit(sigRaw, 0.3)
    bkgSave, bkgPDM = dataSplit(bkgRaw, 0.3)
    ###################################################################
    # TRAINING PREPARATION
    ###################################################################
    gbm = lgb.Booster(model_file='BDT_unblind.txt')  # init model
    
    ###################################################################
    # EVALUATION AND VISUALIZATION
    ###################################################################
    print("---------------- Evaluation and Visualization ---------------")
    print("-------------------------------------------------------------")

    ######################################

    for i in tqdm(range(7), 
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
            shap_valuesFull = explainer.shap_values(sample)
            # Returns a list of matrices (# outputs, # samples x, # features)
            BDTSummary(shap_valuesFull, sample)

        elif i == 2:
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

        elif i == 3:
            sample_bkg = (y_pred<bdt_thresh) & (Y_test == 0) & (X_test[:,selectDict["/AvsE_c"]]>avse_thresh)# & cselector
            shap_bkg = explainer.shap_values(X_test[sample_bkg,:len(fname)])
            outBkgBDT = y_pred[sample_bkg]
            outBkgAvsE = X_test[sample_bkg,selectDict["/AvsE_c"]]
            shap_bkgArr = np.array(shap_bkg[0], dtype=float)
            shap_bkgArr = np.insert(shap_bkgArr, -1, outBkgBDT, axis=1)
            shap_bkgArr = np.insert(shap_bkgArr, -1, outBkgAvsE, axis=1)
            covBKG = np.corrcoef(shap_bkgArr.T)
            plot_covariance(covBKG, "Background Covariance", covName)

        elif i == 4:
            sample_selector = sample_sig|sample_bkg
            evnew = X_test[sample_selector,:len(fname)]
            np.random.shuffle(evnew)
            evnew = evnew[:10000]
            # shap_valuesDist = explainer.shap_values(evnew)
            # make_dist_plot(evnew,shap_valuesDist[1],selectDict, "/tdrift10", "/AvsE_c")
            index = 0
            ROIdata = evnew #X_test[sample_selector]

            ROIdata = ROIdata[ROIdata[:,selectDict["/tdrift"]] < 600]
            sample=ROIdata[index,:len(fname)].reshape(1,-1)
            shap_values = explainer.shap_values(sample)
            plot_SHAP_force(explainer, shap_values[1][0])
        elif i == 5:
            plot_ROC(sigavse, bkgavse, Y_test, y_pred, sigRaw, bkgRaw, selectDict)
        elif i == 6:
            shap_val = np.array(shap_valuesFull)[0]
            # pcaMat = singVarCorr(shap_val, 2)
            # printMVC(pcaMat)
            pcaRes, pcaNames = biVarCorr(shap_val, fname)


    return

if __name__ == "__main__":
    run_BDT()