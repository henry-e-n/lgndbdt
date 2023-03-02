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
def clearAll():
    plt.cla()
    plt.clf()
    plt.close()
    return

def TrainingMetric(evals_result):
    clearAll()
    shap.initjs()
    ax = lgb.plot_metric(evals_result, metric='binary_logloss') # plot of log loss, should be smooth indicating the BDT was appropriately learning over iterations
    plt.savefig(f"{plotPath}/TrainingMetric.png", dpi=100, transparent=True)
    return

def TreeVis(gbm):
    clearAll()
    lgb.plot_tree(gbm, dpi=1000, show_info="data_percentage", figsize=(12,8))
    plt.savefig(f"{plotPath}/PlotTree.png", dpi=1000, transparent=True)
    return

def BDTDistrib(y_pred, Y_test, side_pred = [], side_test=[]):
    clearAll()
    plt.rcParams['font.size'] = 25
    plt.rcParams["figure.figsize"] = (15,16)
    rg=np.arange(0.0,1.01,0.01) # 0.0075
    plt.hist(y_pred[Y_test==1], label="Signal", bins=rg, histtype="step", linewidth = 3, color = "#13294B")# color=cmapNormal(0.2),linewidth=3)
    plt.hist(y_pred[Y_test==0], label="Background",bins=rg, histtype="step", linewidth=3, color = "#EF426F") # , color=cmapNormal(0.8)
    plt.gca().ticklabel_format(axis="y",style="sci")
    if len(side_pred) != 0:
        sideband_signal = side_pred[side_test==1]
        sideband_bkg = side_pred[side_test==0]
        np.random.shuffle(sideband_signal)
        np.random.shuffle(sideband_bkg)
        tau_sig = 1/4
        tau_bkg = 1/4
        plt.hist(sideband_signal[:int(tau_sig*len(sideband_signal))], label="Sideband Signal", bins=rg, histtype="step", linewidth = 3, color = "#13294B", alpha = 0.6)# color=cmapNormal(0.2),linewidth=3)
        plt.hist(sideband_bkg[:int(tau_bkg*len(sideband_bkg))], label="Sideband Background",bins=rg, histtype="step", linewidth=3, color = "#EF426F", alpha = 0.6) # , color=cmapNormal(0.8)
    
    plt.legend(loc="upper center",frameon=False)
    plt.xlabel("BDT output")
    plt.ylabel("# of events / 0.01 BDT Output(a.u.)")
    plt.title("BDT Result Distribution", fontsize = 40)
    plt.savefig(f"{plotPath}/BDT_distribution.png",dpi=300, transparent=True)
    return

def BDTSummary(shap_values, sample):
    clearAll()
    plt.rcParams["figure.figsize"] = (15,8)
    summary_legacy(shap_values[1], sample, plot_type="dot", plot_size=(15,8), feature_names=fname,show=False, cmap=cmapNormal)
    plt.colorbar(fraction = 0.05)
    plt.title("BDT SHAP Feature Importance Summary")
    plt.savefig(f"{plotPath}/bdt_summary.png",dpi=300, bbox_inches = 'tight', pad_inches = 0.3, transparent=True)
    return

def plot_covariance(covMat, saveName, covName = fname):
    clearAll()
    plt.imshow(covMat, cmap=cmapDiv, vmin=-1, vmax=1)
    plt.xticks(np.arange(len(covName)), covName, rotation=60)
    plt.yticks(np.arange(len(covName)), covName)
    plt.title(f"{saveName}")
    plt.colorbar()
    plt.savefig(f"{plotPath}/{saveName.replace(' ', '')}.png",dpi=300, bbox_inches = 'tight', pad_inches = 0.3, transparent=True)
    return

def make_dist_plot(data, shap, selectDict, var1, var2, point=False):
    clearAll()    
    index1 = selectDict[var1]
    index2 = selectDict[var2]
    shapindex = index2
    plt.figure(figsize=(12,8))    
    xlow = np.mean(data[:, index1]) - 3*np.std(data[:, index1])
    xhi  = np.mean(data[:, index1]) + 3*np.std(data[:, index1])
    xlowfit = np.mean(data[:, index1]) - 1*np.std(data[:, index1])

    selector = (data[:,index1] > xlowfit) & (data[:,index2]>-10) & (data[:,index2]<1000)
    z = np.polyfit(data[selector,index1], data[selector,index2],deg=1)
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
    return

def plot_SHAP_force(explainer, shap_values):
    clearAll()
    shapFP = shap.force_plot(explainer.expected_value[1], shap_values, fname, matplotlib = True, show=False, plot_cmap = "PkYg", text_rotation=45)
    plt.savefig(f"{plotPath}/ForcePlot.png",dpi=200, bbox_inches = 'tight', pad_inches = 0.3, transparent=True)
    return


def plot_ROC(sigavse, bkgavse, Y_test, y_pred, sigRaw, bkgRaw, selectDict, inc_ext = True):
    clearAll()
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
    return

def getROC_sideband(peaks_known, peaks_pred, side_sig, side_bkg, sigavse, bkgavse):
    clearAll()
    print(f"SS peak: {len(peaks_known==1)}, SS sideband {len(side_sig)}")
    print(f"MS peak: {len(peaks_known==0)}, MS sideband {len(side_bkg)}")

    dx=0.005                               # Defines resolution of ROC curve
    boundary_line = np.arange(0, 1+dx, dx) # Defines x points at which to calculate TPR and FPR
    
    # Define lists for tpr, fpr, with sideband subtraction, and tpr uncertainty
    tpr = []
    fpr = []
    tpr_side = []
    fpr_side = []
    tpr_unc = []
    tpr_unc_side = []
    
    pred_1 = peaks_pred[peaks_known==1] # predicted values that are known to be SS
    pred_0 = peaks_pred[peaks_known==0] # predicted values that are known to be MS
    
    N_sig = len(pred_1)   # Number of SS in peak
    B_sig = len(side_sig) # Number of SS in sideband
    N_bkg = len(pred_0)   # Number of MS in peak
    B_bkg = len(side_bkg) # Number of MS in sideband
    tau_sig = 1/4         # energy width ratio between the signal and background windows
    tau_bkg = 1/4
    for i in range(len(boundary_line)):
        Nc_sig = np.sum(pred_1>boundary_line[i])   # SS beyond boundary
        Bc_sig = np.sum(side_sig>boundary_line[i]) # SS beyond boundary in sideband

        Nc_bkg = np.sum(pred_0>boundary_line[i])   # MS beyond boundary
        Bc_bkg = np.sum(side_bkg>boundary_line[i]) # MS beyond boundary in sideband

        tprarrSide = (Nc_sig-tau_sig*Bc_sig)/(N_sig-tau_sig*B_sig)
        fprarrSide = (Nc_bkg-tau_bkg*Bc_bkg)/(N_bkg-tau_bkg*B_bkg)
        tpr_side = np.append(tpr_side, tprarrSide)
        fpr_side = np.append(fpr_side, fprarrSide)
        unc_LHS_side = (N_sig + tau_sig **2 * B_sig)/ (N_sig - tau_sig * B_sig)**2 + (Nc_sig + tau_sig **2 * Bc_sig) / (Nc_sig - tau_sig * Bc_sig)**2 - 2*(Nc_sig + tau_sig **2 * Bc_sig)/((N_sig - tau_sig * B_sig) * (Nc_sig - tau_sig * Bc_sig))
        tpr_unc_side = np.append(tpr_unc_side, tprarrSide*(unc_LHS_side)**(0.5))
        
        # Without Sideband Subtraction
        tprArr = (Nc_sig)/(N_sig)   # tpr SS_beyond/SS_total
        fprArr = (Nc_bkg)/(N_bkg)   # fpr MS_beyond/MS_total
        tpr = np.append(tpr, tprArr)
        fpr = np.append(fpr, fprArr)
        unc_LHS = (N_sig)/ (N_sig)**2 + (Nc_sig) / (Nc_sig)**2 - 2*(Nc_sig)/((N_sig) * (Nc_sig))
        tpr_unc = np.append(tpr_unc, tprArr*(unc_LHS)**(0.5))
    
    ############################################################
    # Analysis of Traditional A/E Cut
    cleanSig = np.delete(sigavse, np.argwhere(np.isnan(sigavse)))
    cleanBkg = np.delete(bkgavse, np.argwhere(np.isnan(bkgavse)))
    avseOriginal = np.concatenate((cleanSig,cleanBkg))
    avseOgLabels = np.concatenate((np.ones(len(cleanSig)), np.zeros(len(cleanBkg))))
    ogfpr, ogtpr, ogthresholds    = roc_curve(avseOgLabels, avseOriginal)
    ############################################################
    # Area under the curve calculations
    bdtauc = auc(boundary_line, tpr)
    bdtauc_side = auc(boundary_line, tpr_side)
    ogauc  = auc(np.linspace(0,1,len(ogtpr)), ogtpr) # roc_auc_score(avseOgLabels, avseOriginal)
    ogauc_MCI  = MC_integration(ogtpr)
    bdtauc_MCI = MC_integration(tpr)
    ############################################################
    # horizontal 90% retention cuts
    hlineBDT = np.argmin(np.abs(tpr-0.90))
    hlineBDT_side = np.argmin(np.abs(tpr_side-0.9))
    hlineOG  = np.argmin(np.abs(ogtpr-0.90))
    plt.hlines(y = tpr[hlineBDT], xmin = 0, xmax = np.max((fpr[hlineBDT], ogfpr[hlineOG])), linewidth = 2, color = cmapNormal(0.5), linestyles = 'dashed', alpha = 0.7)
    plt.vlines(x = fpr[hlineBDT], ymin = 0, ymax = tpr[hlineBDT], linewidth = 2, color = cmapNormal(0.5), linestyles = 'dashed', alpha = 0.7)
    plt.vlines(x = fpr_side[hlineBDT_side], ymin = 0, ymax = tpr_side[hlineBDT_side], linewidth = 2, color = cmapNormal(0.5), linestyles = 'dashed', alpha = 0.7)
    plt.vlines(x = ogfpr[hlineOG]  , ymin = 0, ymax = ogtpr[hlineOG]  , linewidth = 2, color = cmapNormal(0.5), linestyles = 'dashed', alpha = 0.7)
    
    plt.vlines(x = 0, ymin = 0, ymax = 1, linewidth = 1, color = cmapNormal(0.5), linestyles = 'dashed', alpha = 0.5)
    plt.hlines(y = 1, xmin = 0, xmax = 1, linewidth = 1, color = cmapNormal(0.5), linestyles = 'dashed', alpha = 0.5)

    ##############################
    ##############################
    # Plotting
    plt.plot([0],[0],color="white",                                              label = "    Classifier       DEP      SEP      AUC     ")
    plt.plot(ogfpr , ogtpr , color = "#13294B", linestyle = "--", linewidth = 4, label = f"       A/E          90.0%   {np.round(100*ogfpr[hlineOG],1)}%    {np.round(ogauc, 2)}")
    plt.plot(fpr, tpr, color = "#EF426F" , linestyle = "-", linewidth = 1,       label = f"       BDT         90.0%   {np.round(100*fpr[hlineBDT],1)}%    {np.round(bdtauc, 2)}")
    
    plt.fill_between(fpr, tpr+tpr_unc, tpr-tpr_unc, color = "#EF426F" , alpha = 0.3, linestyle = "-", linewidth = 4)
    
    plt.plot(fpr_side, tpr_side, color = "#25781f" , linestyle = "-.", linewidth = 1,       label = f"BDT SideSub  90.0%   {np.round(100*fpr_side[hlineBDT_side],1)}%      {np.round(bdtauc_side, 2)}")
    plt.fill_between(fpr_side, tpr_side+tpr_unc_side, tpr_side-tpr_unc_side, color = "#25781f" , alpha = 0.3, linestyle = "-", linewidth = 4)
    
    plt.xlim((-0.02,1))
    plt.ylim((0,1.1))
    plt.legend(loc="lower right")
    plt.xlabel("False Positivity Rate", fontsize = 40)
    plt.ylabel("True Positivity Rate", fontsize = 40)
    return tpr, fpr



def MC_integration(BDT_ROC):
    #===============================================================
    # Probabilty density functions and probability distributions.
    # Name convention follows that of R: d???? is the density function, 
    # r???? is the distribution. 
    # dnorm will give you the normal distribution function (a Gaussian),
    # and rnorm will give you R random variables, normally distributed.
    # The distributions (r????) contain an additional argument bounds to 
    # account for cases where compact support is not provided by the 
    # function itself but needs to be enforced (e.g. runif). 
    #---------------------------------------------------------------
    def dunif(x):
        return (np.zeros(x.size)+1.0) 

    def runif(R,bounds):
        return (bounds[1]-bounds[0])*np.random.rand(R)+bounds[0]

    def dcauc(x):
        return 1.0/(np.pi*(1.0+x*x))
        
    def rcauc(R,bounds):
        return np.tan(np.pi*(np.random.rand(R)-0.5))

    def linFuncArr(yArray, x):
        return yArray[int(x*len(yArray))]

    def rnorm(R,bounds):
        ind1   = np.arange(R/2)*2
        ind2   = np.arange(R/2)*2+1
        u1     = np.random.rand(R/2)
        u2     = np.random.rand(R/2)
        x      = np.zeros(R)
        x[ind1]= np.sqrt(-2.0*np.log(u1))*np.cos(2.0*np.pi*u2)
        x[ind2]= np.sqrt(-2.0*np.log(u1))*np.sin(2.0*np.pi*u2)
        return x

    #===============================================================
    # initialization

    def init(s_target,s_proposal):
        yArray     = s_target/len(s_target)
        fDTAR      = linFuncArr # array of y values
        bounds_dst = np.array([0.0,1.0])

        if (s_proposal == 'uniform'):
            fDPRO   = dunif
            fRPRO   = runif
        elif (s_proposal == 'cauchy'):
            fDPRO   = dcauc
            fRPRO   = rcauc
        else: 
            raise Exception("[init]: invalid s_proposal=%s\n" % (s_proposal))

        # This is rather crude: we search for the maximum value of fDTAR on the
        # bounds_dst specified, and make sure fDPRO on this interval is larger.
        # This would not work very well for multi-dimensional problems...
        x          = np.arange(len(yArray))/len(yArray)
        # x          = (bounds_dst[1]-bounds_dst[0])*np.arange(1000)/999.0+bounds_dst[0]
        R          = x.size
        Qx         = fDPRO(x)
        Px         = yArray #[x]
        maxloc     = np.argmax(Px/Qx) # if > 1, need adaption.
        cval       = Px[maxloc]/Qx[maxloc] # in case our sampling was not sufficient
        print("[init]: cval = %13.5e" % (cval)) 

        return fDTAR, yArray, fDPRO,fRPRO,bounds_dst,cval

    #===============================================================
    # function xr = reject(fDTAR,fDPRO,fRPRO,R)
    # Returns an array of random variables sampled according to a
    # target distribution fDTAR. A proposal distribution fDPRO can
    # be provided.
    #
    # input: fDTAR     : function pointer to the target distribution density function.
    #                    The function must take arguments fTAR(x,bounds),
    #                    and must return the value of fTAR at x.
    #        fDPRO     : function pointer to the proposal distribution density function.
    #        fRPRO     : function pointer to the proposal distribution function. 
    #                    Note that this will return a set of random numbers sampled
    #                    according to fDPRO. Check dnorm and rnorm, for example.
    #        R         : number of samples to be generated
    #        bounds_dst: array of shape (2,N), where N is the number of
    #                    dimensions (elements) in a single x, and the two
    #                    fields per dimension give the lower and upper bound
    #                    in that dimension.
    # output: x_r      : random variables sampled according to P(x)
    #--------------------------------------------------------------
    def reject(fDTAR, yArray, fDPRO,fRPRO,R,bounds_dst,scale):

        mode  = 0
        s     = bounds_dst.shape
        if (len(s) > 1):
            N     = (bounds_dst.shape)[1]
            xr    = np.zeros((R,N))
        else:
            N     = 1
            xr    = np.zeros(R)
        Rsuc = 0
        Rtot = 0
        if (mode == 0): # slow version for demonstration
            while (Rsuc < R):
                # x                  = np.arange(len(fDTAR))#
                x                  = fRPRO(1,bounds_dst)
                u                  = np.random.rand(1)
                Qx                 = scale*fDPRO(x)
                Px                 = fDTAR(yArray, x)#[x]
                if (u <= Px/Qx):
                    xr[Rsuc]       = x
                    Rsuc           = Rsuc+1
                Rtot               = Rtot+1 
            print("[reject]: Rtot = %6i Rsuc = %6i Rsuc/Rtot = %13.5e" % (Rtot,Rsuc,float(Rsuc)/float(Rtot)))
        return xr, float(Rsuc)/float(Rtot)

    #===============================================================
    
    s_target   = BDT_ROC
    s_proposal = "uniform"
    R          = 10000

    fDTAR,yArray,fDPRO,fRPRO,bdst,scal = init(s_target,s_proposal) 
    xr, AUC                            = reject(fDTAR, yArray, fDPRO,fRPRO,R,bdst,scal)
    print(f"AUC {AUC}")
    # check(xr,fDTAR, yArray, fDPRO,bdst,scal)
    return AUC
    #===============================================================

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


def sourceLoc_distCheck(sigRAWTop, bkgRAWTop, sigRAWSide, bkgRAWSide, selectDict, parameter_name):
    plt.hist(sigRAWTop[:, selectDict["parameter_name"]], histtype="step", linewidth = 3, color = cmapNormal[0])
    plt.hist(bkgRAWTop[:, selectDict["parameter_name"]], histtype="step", linewidth = 3, color = cmapNormal[1])
    plt.hist(sigRAWSide[:, selectDict["parameter_name"]], histtype="step", linewidth = 3, color = cmapNormal[2])
    plt.hist(bkgRAWSide[:, selectDict["parameter_name"]], histtype="step", linewidth = 3, color = cmapNormal[3])
    plt.savefig(f"{plotPath}/{parameter_name}DistributionHistogram.png",dpi=300, transparent=True)
