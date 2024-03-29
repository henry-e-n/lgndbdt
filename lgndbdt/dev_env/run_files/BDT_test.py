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

def BDT_test(detector_name, source_location, train_features, data_to_test, BDTmodel = "BDT_unblind.txt"):
    top_file_save_path, top_plot_save_path = get_save_paths(detector_name, "top")
    plot_save_path = top_plot_save_path


    MSBDT     = lgb.Booster(model_file=BDTmodel)
    MSBDTstr  = MSBDT.model_to_string()                
    explainer = shap.TreeExplainer(MSBDT.model_from_string(MSBDTstr))
                

    y_pred = MSBDT.predict(data_to_test[:,:len(train_features)], num_iteration=MSBDT.best_iteration)
    np.save("Y_pred.npy", y_pred)

    plt.hist(y_pred, histtype="step", linewidth = 3, color = "#13294B")
    plt.title("BDT Result Distribution", fontsize = 40)
    plt.savefig(f"BDT_DistributionResults.pdf",dpi=300, transparent=True)
    plt.cla()
    plt.clf()
    plt.close()

    # Pos_sample = data_to_test[:,:len(train_features)]
    # np.random.shuffle(Pos_sample)

    # sample = np.concatenate([Pos_sample],axis=0)
    # shap_values = explainer.shap_values(sample)
    # Returns a list of matrices (# outputs, # samples x, # features)
    # BDTSummary(shap_values, sample, train_features)
    # plt.title(f"BDT SHAP Feature Importance ({source_location})")
    # plt.savefig(f"{plot_save_path}/{source_location}/bdt_summary.pdf",dpi=300, bbox_inches = 'tight', pad_inches = 0.3, transparent=True)


    return