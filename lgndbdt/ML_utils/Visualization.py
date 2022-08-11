import matplotlib.pyplot as plt
import shap
import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score


from extraction_utils.config import plotPath, cmapNormal, fname, cmapDiv, cmapNormal_r
from ML_utils.plot_legacy import summary_legacy

def TrainingMetric(evals_result):
    shap.initjs()
    # print('Plotting metrics recorded during training...')
    ax = lgb.plot_metric(evals_result, metric='binary_logloss') # plot of log loss, should be smooth indicating the BDT was appropriately learning over iterations
    plt.savefig(f"{plotPath}/TrainingMetric.png", dpi=100, transparent=True)
    plt.cla()
    plt.clf()
    plt.close()
    return

def TreeVis(gbm):
    lgb.plot_tree(gbm, dpi=1000, show_info="data_percentage", figsize=(4,4))
    plt.savefig(f"{plotPath}/PlotTree.jpg", dpi=1000, transparent=True)
    plt.cla()
    plt.clf()
    plt.close()

    # graph = lgb.create_tree_digraph(gbm)#, dpi=700, show_info="data_percentage", figsize=(4,4))
    # graph.render(filename=f"{plotPath}/PlotTreeDiGraph", format='png', view=True)
    return

def BDTDistrib(y_pred, Y_test):
    plt.rcParams['font.size'] = 25
    plt.rcParams["figure.figsize"] = (12,8)
    rg=np.arange(0.0,1.0,0.0075)
    plt.hist(y_pred[Y_test==1], label="Signal", bins=rg, histtype="step",color=cmapNormal(0.2),linewidth=3)
    plt.hist(y_pred[Y_test==0], label="Background",bins=rg, histtype="step",color=cmapNormal(0.8),linewidth=3)
    plt.gca().ticklabel_format(axis="y",style="sci")
    plt.legend(loc="upper center",frameon=False)
    plt.xlabel("BDT output")
    plt.ylabel("# of events / 0.01 BDT Output(a.u.)")
    plt.title("BDT Distribtion")
    plt.savefig(f"{plotPath}/BDT_distribution.png",dpi=100, transparent=True)
    # plt.show()
    plt.cla()
    plt.clf()
    plt.close()
    return

def BDTSummary(shap_values, sample):
    plt.rcParams["figure.figsize"] = (12,8)
    summary_legacy(shap_values[1], sample, plot_type="dot", plot_size=(12,8), feature_names=fname,show=False, cmap=cmapNormal)
    plt.colorbar(fraction = 0.05)
    plt.title("BDT Summary")
    plt.savefig(f"{plotPath}/bdt_summary.png",dpi=300, bbox_inches = 'tight', pad_inches = 0.3, transparent=True)
    plt.cla()
    plt.clf()
    plt.close()
    return

def plot_covariance(covMat, saveName, covName = fname):
    plt.imshow(covMat, cmap=cmapDiv, vmin=-1, vmax=1)
    plt.xticks(np.arange(len(covName)), covName, rotation=60)
    plt.yticks(np.arange(len(covName)), covName)
    plt.title(f"{saveName}")
    plt.colorbar()
    plt.savefig(f"{plotPath}/{saveName.replace(' ', '')}.png",dpi=300, bbox_inches = 'tight', pad_inches = 0.3, transparent=True)
    plt.cla()
    plt.clf()
    plt.close()
    return

def make_dist_plot(data, shap, selectDict, var1, var2, point=False):
    # print(shap.shape)
    index1 = selectDict[var1]
    index2 = selectDict[var2]
    shapindex = index2
    plt.figure(figsize=(12,8))    
    xlow = np.mean(data[:, index1]) - 3*np.std(data[:, index1])
    xhi  = np.mean(data[:, index1]) + 3*np.std(data[:, index1])
    xlowfit = np.mean(data[:, index1]) - 1*np.std(data[:, index1])

    selector = (data[:,index1] > xlowfit) & (data[:,index2]>-10) & (data[:,index2]<1000)
    z = np.polyfit(data[selector,index1], data[selector,index2],deg=1)
    # print(f"FIT {z}")
    x = np.linspace(xlow, xhi, 10000)
    y = x * z[0] + z[1]
    
    # Plot
    ymin = np.min(data[:, index2]) - np.std(data[:, index2])
    ymax = np.max(data[:, index2]) + np.std(data[:, index2])

    if point == True:
        selection = (data[:, index1] < 600)
        selection = np.argwhere(selection)
        selection = selection[0]
        # print(selection, data[selection,index1], data[selection,index2])
        plt.scatter(data[selection,index1],data[selection,index2],c="r",cmap=cmapNormal, marker="d", linewidths=5)
        plt.scatter(data[:,index1],data[:,index2],c=shap[:,shapindex]+shap[:,index1],cmap=cmapNormal, zorder=10, alpha = 0.3)
        plt.plot(x,y, color="red",zorder=0,label="Linear Fit", alpha = 0.5)
    else:
        plt.scatter(data[:,index1],data[:,index2],c=shap[:,shapindex]+shap[:,index1],cmap=cmapNormal, zorder=10)
        plt.plot(x,y, color="red",zorder=0,label="Linear Fit")

    plt.ylim(ymin,ymax)
    plt.xlim(xlow,xhi)
    cbar = plt.colorbar()
    plt.xlabel(var1[1:])
    plt.ylabel(var2[1:])
    locs, labels = plt.yticks()
    plt.yticks(np.arange(0, 2000, step=500))
    cbar.ax.set_ylabel('SHAP Value of %s'%(var1))
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plotPath}/AvsE{var1[1:]}.png",dpi=200, transparent=True)
    plt.cla()
    plt.clf()
    plt.close()

    return

def plot_SHAP_force(explainer, shap_values):
    shap.force_plot(explainer.expected_value[1], shap_values, fname, matplotlib = True, show=False)
    plt.savefig(f"{plotPath}/ForcePlot.png",dpi=200, bbox_inches = 'tight', pad_inches = 0.3, transparent=True)
    plt.cla()
    plt.clf()
    plt.close()
    return


def plot_ROC(sigavse, bkgavse, Y_test, y_pred, sigRaw, bkgRaw, selectDict, inc_ext = True):
    cleanSig = np.delete(sigavse, np.argwhere(np.isnan(sigavse)))
    cleanBkg = np.delete(bkgavse, np.argwhere(np.isnan(bkgavse)))

    avseOriginal = np.concatenate((cleanSig,cleanBkg))
    avseOgLabels = np.concatenate((np.ones(len(cleanSig)), np.zeros(len(cleanBkg))))

    BDTfpr, BDTtpr, BDTthresholds = roc_curve(Y_test, y_pred)
    ogfpr, ogtpr, ogthresholds    = roc_curve(avseOgLabels, avseOriginal)
    BDTauc = roc_auc_score(Y_test, y_pred)
    ogauc  = roc_auc_score(avseOgLabels, avseOriginal)

    hlineBDT = np.argmin(np.abs(BDTtpr-0.76))
    hlineOG  = np.argmin(np.abs(ogtpr-0.76))

    if inc_ext:
        cleanSigExt = np.delete(sigRaw[:,selectDict["/AvsE_c"]], np.argwhere(np.isnan(sigavse)))
        cleanBkgExt = np.delete(bkgRaw[:,selectDict["/AvsE_c"]], np.argwhere(np.isnan(bkgavse)))
        avseExt = np.concatenate((cleanSigExt,cleanBkgExt))
        avseExtLabels = np.concatenate((np.ones(len(cleanSigExt)), np.zeros(len(cleanBkgExt))))

        Extfpr, Exttpr, Extthresholds    = roc_curve(avseExtLabels, avseExt)
        Extauc  = roc_auc_score(avseExtLabels, avseExt)
        hlineExt  = np.argmin(np.abs(Exttpr-0.76))
        # print(f"76% thresholds - BDT : {BDTthresholds[hlineBDT]}, PYGAMA : {ogthresholds[hlineOG]}, Extracted {Extthresholds[hlineExt]}")
    # else:
        # print(f"76% thresholds - BDT : {BDTthresholds[hlineBDT]}, PYGAMA : {ogthresholds[hlineOG]}")
    
    # plt.rcParams["font.size"]=24
    plt.figure(figsize=(20,12))
    plt.plot([0],[0],color="white",label = "Classifier             DEP           FEP        AUC    ")
    plt.plot(BDTfpr, BDTtpr, color = cmapNormal(0.6) , linestyle = "-", linewidth = 4, label = f"    BDT               75.0%        {np.round(100*BDTfpr[hlineBDT],1)}%    {np.round(BDTauc, 3)}")
    plt.plot(ogfpr , ogtpr , color = cmapNormal(0.01), linestyle = "--", linewidth = 4, label = f"Pygama A/E       75.0%        {np.round(100*ogfpr[hlineOG],1)}%    {np.round(ogauc, 3)}")
    if inc_ext:
        plt.plot(Extfpr , Exttpr , color = cmapNormal(0.8), linestyle = "-", linewidth = 4, label = f"Extracted A/E    75.0%        {np.round(100*Extfpr[hlineExt],1)}%    {np.round(Extauc, 3)}")
        plt.vlines(x = Extfpr[hlineExt]  , ymin = 0, ymax = Exttpr[hlineExt]  , linewidth = 3, color = cmapNormal(0.5), linestyles = 'dashed', alpha = 0.7)
    plt.hlines(y = BDTtpr[hlineBDT], xmin = 0, xmax = Extfpr[hlineExt]  , linewidth = 3, color = cmapNormal(0.5), linestyles = 'dashed', alpha = 0.7)
    plt.vlines(x = BDTfpr[hlineBDT], ymin = 0, ymax = BDTtpr[hlineBDT], linewidth = 3, color = cmapNormal(0.5), linestyles = 'dashed', alpha = 0.7)
    plt.vlines(x = ogfpr[hlineOG]  , ymin = 0, ymax = ogtpr[hlineOG]  , linewidth = 3, color = cmapNormal(0.5), linestyles = 'dashed', alpha = 0.7)
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.legend(loc="lower right")
    plt.xlabel("False Positivity Rate")
    plt.ylabel("True Positivity Rate")
    plt.suptitle("BDT ROC curve", fontsize = 30, fontweight = 15)
    plt.title("BDT performance vs traditional A/E cut on all events", fontsize = 24, pad = 15, fontstyle='italic')
    plt.savefig(f"{plotPath}/ROC3.png",dpi=200, transparent=True)
    plt.cla()
    plt.clf()
    plt.close()
