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

def BDT_train(detector_name, target_peak, source_location, train_features, match_features, match_step, bdt_thresh = 0.55, avse_thresh = 969, validate="split", augment = True, plots=True):
    file_save_path, plot_save_path = get_save_paths(detector_name, source_location)

    # Validate = "Full" for validation on all data
    isExist = os.path.exists(f"{plot_save_path}/{source_location}/")
    if not isExist:
        os.makedirs(f"{plot_save_path}/{source_location}/")
        print(f"{plot_save_path}/{source_location}/ directory was created!")

    ###################################################################
    # Data Type Preparation
    ###################################################################

    filename        = f"{detector_name}_PSDs_{target_peak}"
    fpath           = f"{file_save_path}"

    print(filename)
    print(fpath)

    def getRaw(filename, fpath):
        file, names, paramArr = paramExtract(filename, fpath, False)
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

    
    
    sigRAWTop, selectDict = getRaw(f"{filename}DEP_top.lh5", f"{fpath}")
    bkgRAWTop, selectDict = getRaw(f"{filename}{target_peak}_top.lh5", f"{fpath}")
    sigRAWSide, selectDict = getRaw(f"{filename}DEP_side.lh5", f"{fpath}")
    bkgRAWSide, selectDict = getRaw(f"{filename}{target_peak}_side.lh5", f"{fpath}")
    
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
    
    sigSave, sigPDM = split_data(sigRAW, 0.3)
    bkgSave, bkgPDM = split_data(bkgRAW, 0.3)

    print(f"Incoming dataset size \n \
            SS shape {len(sigRAW)} - Split to {len(sigSave), len(sigPDM)} \n \
            MS shape {len(bkgRAW)} - Split to {len(bkgSave), len(bkgPDM)}")

    sigSave, sigAUGPDM = split_data(sigAUG, 0.3)
    bkgSave, bkgAUGPDM = split_data(bkgAUG, 0.3)
    
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

    signalTrain, signalTest = split_data(sigs, 0.3)
    sigLabelTrain           = np.ones(signalTrain.shape[0]) # Labels all training signals as signals (1)
    sigLabelTest            = np.ones(signalTest.shape[0]) # Labels all testing signals as signals (1)

    bkgTrain, bkgTest = split_data(bkgs, 0.3)  # assigns arrays corresponding to randomly split signal data 
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


    if plots:
        for i in tqdm(range(6), 
                desc   ="Running Visualization................", 
                colour = terminalCMAP[1]):
            if i == 0:
                if validate=="Full":
                    minSize = np.min([sigRAW.shape[0], bkgRAW.shape[0]])
                    np.random.shuffle(sigRAW)
                    np.random.shuffle(bkgRAW)           
                    signalData   = sigRAW[:minSize, :]
                    bkgData      = bkgRAW[:minSize, :]
                    print(f"Validating on full dataset {signalData.shape}")
                else:
                    # Using split raw data
                    minSize = np.min([sigPDM.shape[0], bkgPDM.shape[0]])
                    np.random.shuffle(sigPDM)
                    np.random.shuffle(bkgPDM)
                    signalData   = sigPDM[:minSize, :]
                    bkgData      = bkgPDM[:minSize, :]
                    print(f"Validating on Split dataset {signalData.shape}")


                X_test = np.concatenate([signalData,bkgData], axis=0)
                Y_test = np.array([1]*len(signalData) + [0] * len(bkgData))
                params = {"num_iterations": 1, "learning_rate": 0.15967607193274216, "num_leaves": 688, "bagging_freq": 34, "bagging_fraction": 0.9411410478379901, "min_data_in_leaf": 54, "drop_rate": 0.030050388917525712, "min_gain_to_split": 0.24143821598351703, "max_bin": 454, "boosting": "dart", "objective": "binary", "metric": "binary_logloss", "verbose": -1}

                lgb_train = lgb.Dataset(X_test[:,:len(train_features)], Y_test,free_raw_data=False, feature_name = list(train_features))
                MSBDT     = lgb.Booster(model_file='BDT_unblind.txt')
                # params["num_iterations"] = 1

                gbm = lgb.train(params, 
                                lgb_train) 

                MSBDTstr  = MSBDT.model_to_string()
                explainer = shap.TreeExplainer(gbm.model_from_string(MSBDTstr))
                
                y_pred = gbm.predict(X_test[:,:len(train_features)], num_iteration=gbm.best_iteration)
                np.save("Y_test.npy", Y_test)
                np.save("Y_pred.npy", y_pred)

                BDTDistrib(y_pred, Y_test)
                plt.title("BDT Result Distribution", fontsize = 40)
                plt.savefig(f"{plot_save_path}/{source_location}/BDT_distribution.pdf",dpi=300, transparent=True)
                plt.cla()
                plt.clf()
                plt.close()
            elif i == 1:
                Pos_sample = X_test[Y_test == 1,:len(train_features)]
                Neg_sample = X_test[Y_test == 0,:len(train_features)]
                np.random.shuffle(Pos_sample)
                np.random.shuffle(Neg_sample)

                sample = np.concatenate([Pos_sample[:10000], Neg_sample[:10000]],axis=0)
                shap_values = explainer.shap_values(sample)
                # Returns a list of matrices (# outputs, # samples x, # features)
                BDTSummary(shap_values, sample, train_features)
                plt.title(f"BDT SHAP Feature Importance ({source_location})")
                plt.savefig(f"{plot_save_path}/{source_location}/bdt_summary.pdf",dpi=300, bbox_inches = 'tight', pad_inches = 0.3, transparent=True)

            elif i == 2:
                # selectDictKeys = selectDict.keys()
                # print(selectDictKeys)
                # print(np.any(np.isin("/A_", selectDictKeys)))
                # if np.any(np.isin("/A_", selectDictKeys)):
                #     print("ITS HERE !!!")
                #     print(np.isin("/A_", selectDictKeys))
                #     print(selectDictKeys[np.isin("/A_", selectDictKeys)])

                explainer  = shap.TreeExplainer(gbm)
                sample_sig = (y_pred>bdt_thresh) & (Y_test == 1) & (X_test[:,selectDict["/A_DAQE"]]<avse_thresh)# & cselector
                sample_bkg  = (y_pred<bdt_thresh) & (Y_test == 0) & (X_test[:,selectDict["/A_DAQE"]]>avse_thresh)# & cselector

                sample_selector = sample_sig|sample_bkg
                evnew = X_test[sample_selector,:len(train_features)]
                np.random.shuffle(evnew)
                evnew = evnew[:10000]
                shap_valuesDist = explainer.shap_values(evnew)
                make_dist_plot(evnew,shap_valuesDist[1],selectDict, "/TDRIFT10", "/A_DAQE", plot_save_path)
                make_dist_plot(evnew,shap_valuesDist[1],selectDict, "/TDRIFT", "/A_DAQE", plot_save_path)
                make_dist_plot(evnew,shap_valuesDist[1],selectDict, "/TDRIFT50", "/A_DAQE", plot_save_path)
                make_dist_plot(evnew,shap_valuesDist[1],selectDict, "/LQ80", "/A_DAQE", plot_save_path)
                make_dist_plot(evnew,shap_valuesDist[1],selectDict, "/DCR", "/A_DAQE", plot_save_path)
                # make_dist_plot(evnew,shap_valuesDist[1],selectDict, "/NOISE", "/A_DAQE"),
                # make_dist_plot(evnew,shap_valuesDist[1],selectDict, "/NOISETAIL", "/A_DAQE"),
                make_dist_plot(evnew,shap_valuesDist[1],selectDict, "/TDRIFT", "/A_DAQE", plot_save_path, point=True)
            elif i == 3:
                # index = 0
                # ROIdata = evnew
                # ROIdata     = ROIdata[ROIdata[:,selectDict["/tdrift"]] < 600]
                # sample      = ROIdata[index,:len(train_features)].reshape(1,-1)
                # shap_values = explainer.shap_values(sample)
                # print(np.shape(explainer.expected_value), np.shape(shap_values), shap_values[1][1])
                # shapFP = shap.force_plot(explainer.expected_value[1], shap_values[1][1], train_features, matplotlib = True, show=False, plot_cmap = "PkYg", text_rotation=45)
                # for n in range(25):
                #     plot_SHAP_force(explainer, shap_values[1][-n])
                #     plt.savefig(f"{plot_save_path}/{source_location}/ForcePlots/ForcePlot{-n}.pdf",dpi=300, bbox_inches = 'tight', pad_inches = 0.3, transparent=True)
                
                plot_SHAP_force(explainer, shap_values[1][1], train_features)
                plt.savefig(f"{plot_save_path}/{source_location}/ForcePlot.pdf",dpi=300, bbox_inches = 'tight', pad_inches = 0.3, transparent=True)
                
                # plot_SHAP_force(explainer, shap_values[1][2])
                # plt.savefig(f"{plot_save_path}/ForcePlot2.pdf",dpi=300, bbox_inches = 'tight', pad_inches = 0.3, transparent=True)
                # plot_SHAP_force(explainer, shap_values[1][3])
                # plt.savefig(f"{plot_save_path}/ForcePlot3.pdf",dpi=300, bbox_inches = 'tight', pad_inches = 0.3, transparent=True)
            
            elif i == 4:
                
                if source_location == "mix":
                    sig_sideband_RawTop, selectDict = getRaw(f"{filename}topDEP_sideband.lh5", f"{fpath}")
                    bkg_sideband_RawTop, selectDict = getRaw(f"{filename}top{target_peak}_sideband.lh5", f"{fpath}")
                    sig_sideband_RawSide, selectDict = getRaw(f"{filename}sideDEP_sideband.lh5", f"{fpath}")
                    bkg_sideband_RawSide, selectDict = getRaw(f"{filename}side{target_peak}_sideband.lh5", f"{fpath}")
                    print(f"SIDEBAND DATA: Runs include a mix of data from source location on the top, and on the side\nTop Data Size (sig, bkg) {sig_sideband_RawTop.shape}, {bkg_sideband_RawTop.shape}\nSide Data Size (sig, bkg) {sig_sideband_RawSide.shape}, {bkg_sideband_RawSide.shape}")
                    sig_sideband_RAW = np.concatenate((sig_sideband_RawTop, sig_sideband_RawSide))
                    bkg_sideband_RAW = np.concatenate((bkg_sideband_RawTop, bkg_sideband_RawSide))
                else:
                    sig_sideband_RAW, selectDict = getRaw(f"{filename}{source_location}DEP_sideband.lh5", f"{fpath}")
                    bkg_sideband_RAW, selectDict = getRaw(f"{filename}{source_location}{target_peak}_sideband.lh5", f"{fpath}")
                
                print(f"Sideband Comparison (RAW)\n \
                        SS Peak Size {len(sigRAW)} - SS Sideband size {len(sig_sideband_RAW)} - \u03C4 = 4, {1/4*len(sig_sideband_RAW)}\n \
                        MS Peak Size {len(bkgRAW)} - MS Sideband size {len(bkg_sideband_RAW)} - \u03C4 = 4, {1/4*len(bkg_sideband_RAW)}")

                if validate=="Full":
                    sig_sideband_Ratio = sig_sideband_RAW
                    bkg_sideband_Ratio = bkg_sideband_RAW
                    print(f"Ratio loss of signal data {len(signalData)/len(sigRAW)}")
                    print(f"Ratio loss of bkg data {len(bkgData)/len(bkgRAW)}")
                    lossRatio_SIG = len(signalData)/len(sigRAW)
                    lossRatio_BKG = len(bkgData)/len(bkgRAW)

                else:
                    sig_sideband_Save, sig_sideband_Ratio = split_data(sig_sideband_RAW, 0.3)
                    bkg_sideband_Save, bkg_sideband_Ratio = split_data(bkg_sideband_RAW, 0.3)
                    print(f"Ratio loss of signal data {len(signalData)/len(sigPDM)}")
                    print(f"Ratio loss of bkg data {len(bkgData)/len(bkgPDM)}")
                    lossRatio_SIG = len(signalData)/len(sigPDM)
                    lossRatio_BKG = len(bkgData)/len(bkgPDM)


                np.random.shuffle(sig_sideband_Ratio)
                np.random.shuffle(bkg_sideband_Ratio)
                sig_sideband_Ratio = sig_sideband_Ratio[:int(lossRatio_SIG*len(sig_sideband_Ratio))]
                bkg_sideband_Ratio = bkg_sideband_Ratio[:int(lossRatio_BKG*len(bkg_sideband_Ratio))]

                MSBDT     = lgb.Booster(model_file='BDT_unblind.txt')
                gbm = lgb.train(params, 
                                lgb_train) 
                MSBDTstr  = MSBDT.model_to_string()
                explainer = shap.TreeExplainer(gbm.model_from_string(MSBDTstr))
                
                sig_sideband_pred = gbm.predict(sig_sideband_Ratio, num_iteration=gbm.best_iteration)
                bkg_sideband_pred = gbm.predict(bkg_sideband_Ratio, num_iteration=gbm.best_iteration)

                result = list(filter(lambda x: "A_" in x, selectDict))
                
                if validate=="Full":
                    sigavse = sigRAW[:,selectDict[result[0]]]
                    bkgavse = bkgRAW[:,selectDict[result[0]]]
                else:
                    # small set validation
                    sigavse = sigPDM[:,selectDict[result[0]]]
                    bkgavse = bkgPDM[:,selectDict[result[0]]]
                    
                # sigavse = sigPDM[:,selectDict[result[0]]]
                # bkgavse = bkgPDM[:,selectDict[result[0]]]
                
                side_pred = np.concatenate((sig_sideband_pred, bkg_sideband_pred))
                side_test = np.array([1]*len(sig_sideband_pred) + [0]*len(bkg_sideband_pred))
                BDTDistrib(y_pred, Y_test, side_pred, side_test)
                plt.title(f"BDT Distribution - {source_location} data", fontsize = 40)
                plt.savefig(f"{plot_save_path}/{source_location}/BDT_{source_location}_+Sideband_distribution.pdf",dpi=300, transparent=True)
                
                tpr, fpr = getROC_sideband(Y_test, y_pred, sig_sideband_pred, bkg_sideband_pred, sigavse, bkgavse)#, sideSigAvsE, sideBkgAvsE)
                plt.title(f"ROC performance - {source_location} data", fontsize = 40) #, fontsize = 24, pad = 15, fontstyle='italic')
                plt.savefig(f"{plot_save_path}/{source_location}/ROC_{source_location}_sideband.pdf",dpi=300, transparent=False)
                plt.cla()
                plt.clf()
                plt.close()
            elif i==5:
                os.environ["PATH"] += os.pathsep + '/global/homes/h/hnachman/.conda/pkgs/graphviz-2.50.0-h3cd0ef9_0/bin/'
                digraph = lgb.create_tree_digraph(MSBDT, 0, name="PlotTree", directory=f"{plot_save_path}/{source_location}/", format="pdf") # , renderer="cairo", formatter="cairo"
                digraph.render(directory=f"{plot_save_path}/{source_location}/", view=True)

    return