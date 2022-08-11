import lightgbm as lgb
import numpy as np
import os
import sys
import gc
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import tqdm

from tqdm import tqdm
from time import time
from imblearn.over_sampling      import SMOTENC

from matplotlib import cm

module_path = [os.path.abspath(os.path.join("C:/Users/henac/GitHub/ENAP_Personal/LegendMachineLearning/AnalysisExtraction"))]
for path in module_path:
    if path not in sys.path:
        sys.path.append(path)

from extraction_utils.config     import *
from extraction_utils.h5Extract  import *
from ML_utils.BDTPrep            import *
from ML_utils.BDTTrain           import *
from extraction_utils.RawToTrain import *
from ML_utils.plot_legacy        import summary_legacy

randSeed = 27
np.random.seed(randSeed)
# random.seed(randSeed)

matplotlib.use('Agg')
###################################################################
# Data Type Preparation
###################################################################
def main(distribList):
    np.random.seed(13)


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
        # print(f"Returned {fpath}{filename}")#, shape {dataArr.shape}")
        return dataArr, dataDictionary, wfd, avse, selectDictionary

    sigRaw, sigDict, sigWFD, sigavse, selectDict = getRaw(filename, f"{fpath}DEP/")
    bkgRaw, bkgDict, bkgWFD, bkgavse, selectDict = getRaw(filename, f"{fpath}FEP/")

    ###################################################################
    # DATA MATCHING
    ###################################################################

    sigSave, sigPDM = dataSplit(sigRaw, 0.3)
    bkgSave, bkgPDM = dataSplit(bkgRaw, 0.3)

    for i in range(len(distribList)):
        feat = distribList[i]
        if feat == "/DCR":
            dFeat = 0.05
        if feat == "/tdrift":
            dFeat = 30
        if feat == "/tdrift10":
            dFeat = 20
        if feat == "/tdrift50":
            dFeat = 20
        if feat == "/noise":
            dFeat = 0.00002
        if feat == "/noiseTail":
            dFeat = 0.00002
        if feat == "/LQ80":
            dFeat = 10
        sigSave, bkgSave = match_data(sigSave, bkgSave, selectDict, feat, dFeat, True, show = False)
    
    sigs = sigSave
    bkgs = bkgSave
    


    ###################################################################
    # TRAINING PREPARATION
    ###################################################################

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

    lgbTrain = lgb.Dataset(xTrain, yTrain, free_raw_data=False, feature_name = list(fname))
    lgbEval = lgb.Dataset(xVal, yVal, reference=lgbTrain, free_raw_data=False, feature_name = list(fname))

    # Defines the hyperparameters of the BDT 
    params={"num_iterations": 2000, "learning_rate": 0.07442318529884213,
            "num_leaves": 73, "bagging_freq": 62, "min_data_in_leaf": 26,
            "drop_rate": 0.3299436689754462, "min_gain_to_split": 0.5355479139953352,
            "max_bin": 542, "boosting": "goss", "objective": "binary", "metric": "binary_logloss", "verbose": -100, "seed" : randSeed, "deterministic" : True}
    evals_result = {}

    # Performs the training on the dataset
    gbm = lgb.train(params, 
                    lgbTrain,
                    feature_name=list(fname), 
                    valid_sets=lgbEval,
                    early_stopping_rounds=15,
                    evals_result=evals_result,
                    verbose_eval=False) 

    explainer = shap.TreeExplainer(gbm)
    gbm.save_model('BDT_unblind.txt') # Saves the BDT model as txt file

    ###################################################################
    # EVALUATION AND VISUALIZATION
    ###################################################################

    signalData = sigPDM
    bkgData = bkgPDM

    X_test = np.concatenate([signalData,bkgData],axis=0)
    Y_test = np.array([1]*len(signalData) + [0] * len(bkgData))

    # Is this code necesssary??

    # params = {"num_iterations": 1, "learning_rate": 0.15967607193274216, "num_leaves": 688, "bagging_freq": 34, "bagging_fraction": 0.9411410478379901, "min_data_in_leaf": 54, "drop_rate": 0.030050388917525712, "min_gain_to_split": 0.24143821598351703, "max_bin": 454, "boosting": "dart", "objective": "binary", "metric": "binary_logloss", "verbose": -100}

    # lgb_train = lgb.Dataset(X_test[:,:len(fname)], Y_test,free_raw_data=False, feature_name = list(fname))
    # MSBDT = lgb.Booster(model_file='BDT_unblind.txt')
    # params["num_iterations"] = 1

    # gbm = lgb.train(params, 
    #                 lgb_train) 

    # MSBDTstr = MSBDT.model_to_string()

    # replace with
    MSBDT = lgb.Booster(model_file='BDT_unblind.txt')
    MSBDTstr = MSBDT.model_to_string()

    explainer = shap.TreeExplainer(gbm.model_from_string(MSBDTstr))
    y_pred = gbm.predict(X_test[:,:len(fname)], num_iteration=gbm.best_iteration)

    # ROC CURVES #
    # cleanSig = np.delete(sigavse, np.argwhere(np.isnan(sigavse)))
    # cleanBkg = np.delete(bkgavse, np.argwhere(np.isnan(bkgavse)))

    # avseOriginal = np.concatenate((cleanSig,cleanBkg))
    # avseOgLabels = np.concatenate((np.ones(len(cleanSig)), np.zeros(len(cleanBkg))))

    # BDTfpr, BDTtpr, BDTthresholds = roc_curve(Y_test, y_pred)
    # ogfpr, ogtpr, ogthresholds    = roc_curve(avseOgLabels, avseOriginal)
    BDTauc = roc_auc_score(Y_test, y_pred)
    
    return np.round(BDTauc, 5), np.shape(sigs)

def run_SDS():
    featureList = distMatch
    import itertools
    combos = []
    for L in range(0, len(featureList) + 1):
        for subset in itertools.combinations(featureList, L):
            combos.append(subset)

    bdtAUC   = []
    finSize  = []
    bigStart = time()
    for n in tqdm(combos):
        auc, szF = main(n)
        bdtAUC.append(auc)
        finSize.append(szF[0])

    sysDist = np.stack([bdtAUC, finSize])
    # np.savetxt("Plots/bdtAUC.csv", bdtAUC)
    # np.savetxt("Plots/finSize.csv", finSize)
    # np.save("Plots/SYSDIST", sysDist)

    bigEnd = time()
    print(f"Total Run Time : {bigEnd - bigStart}")

    data = np.stack([combos, sysDist[0,:], sysDist[1,:]])
    labelData = data[0, :]
    auc = data[1,:]
    finalSize = data[2,:]*2

    maxFS = np.argmax(finalSize)
    maxAUC = np.argmax(auc)

    aucStandard = auc[-1]

    new_tick_locations = np.array([0.78, 0.82, 0.84])

    def tick_func(X, standard):
        V = ((X-standard)/standard)*100
        return ["%.3f" % z for z in V]

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()

    ax1.plot(auc, finalSize, '.')
    ax1.plot(auc[-1], finalSize[-1], '*', color='purple', label = "Standard All Matched")
    ax1.plot(auc[maxFS], finalSize[maxFS], 's', color = 'g', label = labelData[maxFS])
    ax1.plot(auc[maxAUC], finalSize[maxAUC], 'o', color='r', label = labelData[maxAUC])
    ax1.set_xlabel(r"Raw ROC AUC score")
    ax1.set_ylabel(r"Total Number of Events (Sig + Bkg)")
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(new_tick_locations)
    ax2.set_xticklabels(tick_func(new_tick_locations, aucStandard))
    ax2.set_xlabel(r"Percent Increase from Standard")
    lgd = ax1.legend(bbox_to_anchor=(0.5, -0.40), loc = "lower center")
    plt.savefig(f"{plotPath}/SystematicDistribution.jpg", dpi=100, bbox_extra_artists=(lgd,), bbox_inches='tight')


    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    ax1.hist2d(auc, finalSize, cmap = plt.get_cmap("Blues"), bins=(np.linspace(0.78, 0.86, 40), np.linspace(9000, 27000, 40)))
    ax1.plot(auc[-1], finalSize[-1], '*', color='purple', label = "Standard All Matched")
    ax1.plot(auc[maxFS], finalSize[maxFS], 's', color = 'g', label = labelData[maxFS])
    ax1.plot(auc[maxAUC], finalSize[maxAUC], 'o', color='r', label = labelData[maxAUC])
    ax1.set_xlabel(r"Raw ROC AUC score")
    ax1.set_ylabel(r"Total Number of Events (Sig + Bkg)")
    lgd = ax1.legend(bbox_to_anchor=(0.5, -0.40), loc = "lower center")
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(new_tick_locations)
    ax2.set_xticklabels(tick_func(new_tick_locations, aucStandard))
    ax2.set_xlabel(r"Percent Increase from Standard")
    lgd = ax1.legend(bbox_to_anchor=(0.5, -0.40), loc = "lower center")
    plt.savefig(f"{plotPath}/SystematicDistributionHist2d.jpg", dpi=100, bbox_extra_artists=(lgd,), bbox_inches='tight')
