import matplotlib.pyplot as plt
import shap
import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.decomposition import PCA
import itertools

from extraction_utils.config import plotPath, cmapNormal, fname, cmapDiv, cmapNormal_r
from ML_utils.plot_legacy import summary_legacy
from matplotlib.colors import ListedColormap

forceCMAP = ListedColormap(["#13294B", "#EF426F"])

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
    print("SKIP TREE VIS")
    lgb.plot_tree(gbm, dpi=1000, show_info="data_percentage", figsize=(12,8))
    plt.savefig(f"{plotPath}/PlotTree.png", dpi=1000, transparent=True)
    plt.cla()
    plt.clf()
    plt.close()

    # graph = lgb.create_tree_digraph(gbm)#, dpi=700, show_info="data_percentage", figsize=(4,4))
    # graph.render(filename=f"{plotPath}/PlotTreeDiGraph", format='png', view=True)
    return

def BDTDistrib(y_pred, Y_test):
    # print("RUNNING BDTDISTRIB")
    plt.rcParams['font.size'] = 25
    plt.rcParams["figure.figsize"] = (15,16)
    rg=np.arange(0.0,1.01,0.01) # 0.0075
    
    # print(f"41 Vis: {len(y_pred[Y_test==1])}, {len(y_pred[Y_test==0])}")
    # print(f"41 Vis: {len(y_pred[y_pred<=0.5])}, {len(y_pred[y_pred>0.5])}")

    plt.hist(y_pred[Y_test==1], label="Signal", bins=rg, histtype="step", linewidth = 3, color = "#13294B")# color=cmapNormal(0.2),linewidth=3)
    plt.hist(y_pred[Y_test==0], label="Background",bins=rg, histtype="step", linewidth=3, color = "#EF426F") # , color=cmapNormal(0.8)
    plt.gca().ticklabel_format(axis="y",style="sci")
    plt.legend(loc="upper center",frameon=False)
    plt.xlabel("BDT output")
    plt.ylabel("# of events / 0.01 BDT Output(a.u.)")
    plt.title("BDT Result Distribution", fontsize = 40)
    plt.savefig(f"{plotPath}/BDT_distribution.png",dpi=300, transparent=True)
    # plt.show()
    plt.cla()
    plt.clf()
    plt.close()
    return

def BDTSummary(shap_values, sample):
    plt.rcParams["figure.figsize"] = (15,8)
    summary_legacy(shap_values[1], sample, plot_type="dot", plot_size=(15,8), feature_names=fname,show=False, cmap=cmapNormal)
    plt.colorbar(fraction = 0.05)
    plt.title("BDT SHAP Feature Importance Summary")
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
    # print("force_plot")
    # shap.force_plot(explainer.expected_value[1], shap_values, fname, matplotlib = False, show=True, plot_cmap = "PkYg", text_rotation=60)
    shapFP = shap.force_plot(explainer.expected_value[1], shap_values, fname, matplotlib = True, show=False, plot_cmap = "PkYg", text_rotation=45)
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

    plt.figure(figsize=(15,16))
    plt.plot([0],[0],color="white",label = " Classifier             DEP           FEP        AUC    ")
    plt.plot(BDTfpr, BDTtpr, color = "#EF426F" , linestyle = "-", linewidth = 4, label = f"     BDT               75.0%        {np.round(100*BDTfpr[hlineBDT],1)}%    {np.round(BDTauc, 3)}")
    plt.plot(ogfpr , ogtpr , color = "#13294B", linestyle = "--", linewidth = 4, label = f"Extracted A/E      75.0%        {np.round(100*ogfpr[hlineOG],1)}%    {np.round(ogauc, 3)}")
    if inc_ext:
        plt.plot(Extfpr , Exttpr , color = "#B87333", linestyle = "-.", linewidth = 4, label = f"Extracted A/E    75.0%        {np.round(100*Extfpr[hlineExt],1)}%    {np.round(Extauc, 3)}")
        plt.vlines(x = Extfpr[hlineExt]  , ymin = 0, ymax = Exttpr[hlineExt]  , linewidth = 3, color = cmapNormal(0.5), linestyles = 'dashed', alpha = 0.7)
    plt.hlines(y = BDTtpr[hlineBDT], xmin = 0, xmax = BDTfpr[hlineBDT]  , linewidth = 3, color = cmapNormal(0.5), linestyles = 'dashed', alpha = 0.7)
    plt.vlines(x = BDTfpr[hlineBDT], ymin = 0, ymax = BDTtpr[hlineBDT], linewidth = 3, color = cmapNormal(0.5), linestyles = 'dashed', alpha = 0.7)
    plt.vlines(x = ogfpr[hlineOG]  , ymin = 0, ymax = ogtpr[hlineOG]  , linewidth = 3, color = cmapNormal(0.5), linestyles = 'dashed', alpha = 0.7)
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.legend(loc="lower right")
    plt.xlabel("False Positivity Rate", fontsize = 40)
    plt.ylabel("True Positivity Rate", fontsize = 40)
    plt.title("BDT vs traditional A/E ROC performance", fontsize = 40) #, fontsize = 24, pad = 15, fontstyle='italic')
    plt.savefig(f"{plotPath}/ROC3.png",dpi=300, transparent=True)
    plt.cla()
    plt.clf()
    plt.close()

def getROC_sideband(peaks_known, peaks_pred, side_sig, side_bkg, sigavse, bkgavse):
    dx=0.05
    boundary_line = np.arange(0.0, 0.65, dx)
    print(boundary_line)
    tpr = []
    fpr = []
    tpr_unc = []
    pred_1 = peaks_pred[peaks_known==1] # predicted values that are known to be SS
    pred_0 = peaks_pred[peaks_known==0] # predicted values that are known to be MS
    
    N_sig = len(pred_1)
    B_sig = len(side_sig)
    N_bkg = len(pred_0)
    B_bkg = len(side_bkg)
    tau_sig = 1 # energy width ratio between the signal and background windows
    tau_bkg = 1
    for i in range(len(boundary_line)):
        Nc_sig = np.sum(pred_1>boundary_line[i])
        Bc_sig = np.sum(side_sig>boundary_line[i])

        Nc_bkg = np.sum(pred_0>boundary_line[i])
        Bc_bkg = np.sum(side_bkg>boundary_line[i])
        print(f"linePos, Nc_sig, Bc_sig, N_sig, B_sig, num, den, {Nc_sig, Bc_sig, N_sig, B_sig, Nc_sig-tau_sig*Bc_sig, N_sig-tau_sig*B_sig}")
        tprarr = (Nc_sig-tau_sig*Bc_sig)/(N_sig-tau_sig*B_sig)
        fprarr = (Nc_bkg-tau_bkg*Bc_bkg)/(N_bkg-tau_bkg*B_bkg)
        
        tpr = np.append(tpr, tprarr)
        fpr = np.append(fpr, fprarr)

        unc_LHS = (N_sig + tau_sig **2 * B_sig)/ (N_sig - tau_sig * B_sig)**2 + (Nc_sig + tau_sig **2 * Bc_sig) / (Nc_sig - tau_sig * Bc_sig)**2 - 2*(Nc_sig + tau_sig **2 * Bc_sig)/((N_sig - tau_sig * B_sig) * (Nc_sig - tau_sig * Bc_sig))
        tpr_unc = np.append(tpr_unc, tprarr*(unc_LHS)**(0.5))
        print(f"tprarr {tprarr} +- {tprarr*(unc_LHS)**(0.5)}, fprarr {fprarr}")



    cleanSig = np.delete(sigavse, np.argwhere(np.isnan(sigavse)))
    cleanBkg = np.delete(bkgavse, np.argwhere(np.isnan(bkgavse)))

    avseOriginal = np.concatenate((cleanSig,cleanBkg))
    avseOgLabels = np.concatenate((np.ones(len(cleanSig)), np.zeros(len(cleanBkg))))
    ogfpr, ogtpr, ogthresholds    = roc_curve(avseOgLabels, avseOriginal)
    
    bdtauc = auc(boundary_line, tpr)
    ogauc  = roc_auc_score(avseOgLabels, avseOriginal)


    plt.plot([0],[0],color="white",                                              label = " Classifier     AUC    ")
    plt.plot(ogfpr , ogtpr , color = "#13294B", linestyle = "--", linewidth = 4, label = f"   A/E      {np.round(ogauc, 3)}")
    plt.plot(fpr, tpr, color = "#EF426F" , linestyle = "-", linewidth = 4,       label = f"   BDT      {np.round(bdtauc, 3)}")
    plt.fill_between(fpr, tpr+tpr_unc, tpr-tpr_unc, color = "#EF426F" , alpha = 0.3, linestyle = "-", linewidth = 4,       label = f"   BDT      {np.round(bdtauc, 3)}")
    plt.hlines(y = 1, xmin = 0, xmax = 1, linewidth = 3, color = cmapNormal(0.5), linestyles = 'dashed', alpha = 0.7)
    plt.xlim((0,1))
    plt.ylim((0,1.1))
    plt.legend(loc="lower right")
    plt.xlabel("False Positivity Rate", fontsize = 40)
    plt.ylabel("True Positivity Rate", fontsize = 40)
    plt.title("BDT vs traditional A/E ROC performance", fontsize = 40) #, fontsize = 24, pad = 15, fontstyle='italic')
    plt.savefig(f"{plotPath}/ROC_sideband.png",dpi=300, transparent=True)
    plt.cla()
    plt.clf()
    plt.close()
    return tpr, fpr

def printMVC(pcaMat):
    plt.imshow(pcaMat, cmap=cmapDiv)
    plt.suptitle("Multi Variate PCA", fontsize = 30, fontweight = 15)
    plt.title("Read as [x y]", fontsize = 24, pad = 15, fontstyle='italic')
    plt.xticks(np.arange(len(fname)), fname, rotation=60)
    plt.yticks(np.arange(len(fname)), fname)
    plt.colorbar()
    plt.savefig(f"{plotPath}/mvc.png",dpi=300, bbox_inches = 'tight', pad_inches = 0.3, transparent=True)
    plt.cla()
    plt.clf()
    plt.close()

def printBVC(pcaVect, pcaNames):
    plt.figure(figsize=(12,14))
    barColors = cmapNormal_r(pcaVect+0.3)
    plt.barh(np.arange(len(pcaVect)), pcaVect, color=barColors)
    plt.suptitle("PCA - Log Scale", fontsize = 30, fontweight = 15)
    # plt.title("Log Scale", fontsize = 24, pad = 15, fontstyle='italic')
    plt.yticks(np.arange(len(pcaNames)), pcaNames) #, rotation=90
    plt.semilogx()
    plt.savefig(f"{plotPath}/bvc.png",dpi=300, bbox_inches = 'tight', pad_inches = 0.3, transparent=True)
    plt.cla()
    plt.clf()
    plt.close()

    small = (pcaVect<0.03)
    others = np.sum(pcaVect*small)

    pcaVectPie = np.append(np.delete(pcaVect, small), others)
    pcaNamesPie = np.append(np.delete(pcaNames, small), "Others (< 3%)")
    Piecolors = cmapNormal_r(pcaVectPie+0.3)

    plt.pie(pcaVectPie, labels=pcaNamesPie, autopct='%1.1f%%', colors=Piecolors)
    plt.suptitle("PCA", fontsize = 30, fontweight = 15)
    # plt.title("Raw Scale", fontsize = 24, pad = 15, fontstyle='italic')
    plt.savefig(f"{plotPath}/bvcPIE.png",dpi=300, bbox_inches = 'tight', pad_inches = 0.3, transparent=True)
    plt.cla()
    plt.clf()
    plt.close()

    # plt.pie(np.abs(np.min(np.log10(pcaVect)))-np.abs(np.log10(pcaVect)), labels=pcaNames, autopct='%1.1f%%')
    # plt.suptitle("Bivariate PCA - Log Scale", fontsize = 30, fontweight = 15)
    # # plt.title("Log Scale", fontsize = 24, pad = 15, fontstyle='italic')
    # plt.savefig(f"{plotPath}/bvcPIElog.png",dpi=300, bbox_inches = 'tight', pad_inches = 0.3, transparent=True)
    # plt.cla()
    # plt.clf()
    # plt.close()

    # cut = np.where(np.char.find(np.array(pcaNames, dtype=str), "AvsE")>0)[0]
    # pcaVect = np.delete(pcaVect, cut)
    # pcaNames = np.delete(pcaNames, cut)
    
    # plt.pie(np.abs(np.min(np.log10(pcaVect)))-np.abs(np.log10(pcaVect)), labels=pcaNames, autopct='%1.1f%%')
    # plt.suptitle("Bivariate PCA - AvsE cut - Log", fontsize = 30, fontweight = 15)
    # # plt.title("Log Scale", fontsize = 24, pad = 15, fontstyle='italic')
    # plt.savefig(f"{plotPath}/bvcPIElogCut.png",dpi=300, bbox_inches = 'tight', pad_inches = 0.3, transparent=True)
    # plt.cla()
    # plt.clf()
    # plt.close()

    # plt.pie(pcaVect, labels=pcaNames, autopct='%1.1f%%')
    # plt.suptitle("Bivariate PCA - AvsE cut - Raw", fontsize = 30, fontweight = 15)
    # # plt.title("Raw Scale", fontsize = 24, pad = 15, fontstyle='italic')
    # plt.savefig(f"{plotPath}/bvcPIErawCut.png",dpi=300, bbox_inches = 'tight', pad_inches = 0.3, transparent=True)
    # plt.cla()
    # plt.clf()
    # plt.close()


def printPCAResults(pcaResults, pcaNames):    
    [pcaComp, pcaEVR] = pcaResults
    [pcaNames, pltNames] = pcaNames

    print(f"PCA components - {pcaComp}")
    print(f"PCA explained variance ratio - {pcaEVR}")
    print(f"Names - {pcaNames}")

    plt.barh(np.arange(len(pcaEVR)), np.abs(pcaEVR))
    plt.suptitle("PCA - Log Scale", fontsize = 30, fontweight = 15)
    plt.yticks(np.arange(len(pltNames)), pltNames) #, rotation=90
    plt.semilogx()
    plt.xlim(0.001, 1)
    plt.savefig(f"bvcComp.png",dpi=300, bbox_inches = 'tight', pad_inches = 0.3, transparent=False)



    pcaCompPlot = np.zeros(pcaComp.shape)
    for s in range(pcaComp.shape[0]):
        pcaCompPlot[s, :] = np.abs(pcaComp[s,:])/np.sum(np.abs(pcaComp[s,:]))*pcaEVR[s]
    plt.figure()
    inc = np.zeros(len(pcaEVR))
    plt.barh(np.arange(len(pcaEVR)), np.abs(pcaCompPlot[:, 0]), label=f"{pcaNames[0]}")
    for i in range(1, len(pcaCompPlot[0, :])):
        inc = np.abs(inc+pcaCompPlot[:, i-1])
        plt.barh(np.arange(len(pcaEVR)), np.abs(pcaCompPlot[:, i]), left=inc, label=f"{pcaNames[i]}")

    plt.suptitle("PCA - Log Scale", fontsize = 30, fontweight = 15)
    plt.yticks(np.arange(len(pltNames)), pltNames) #, rotation=90
    plt.semilogx()
    plt.xlim(0.001, 1)
    plt.legend(ncol=2, bbox_to_anchor = (1.05, 0.99))
    plt.savefig(f"bvc.png",dpi=300, bbox_inches = 'tight', pad_inches = 0.3, transparent=False)

