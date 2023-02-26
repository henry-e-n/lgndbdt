from typing import Concatenate
# from extraction_utils.config import *

from lgndbdt.raw_to_bdt.get_config import get_config as config
filenames_config, BDT_config, path_config, visual_config = config()
[raw_files, dsp_files]       = filenames_config
[fname, distMatch, distStep] = BDT_config
[detName, targetPeak, source, savePath, psdPath, plotPath, fname] = path_config
[terminalCMAP, cmapNormal, cmapNormal_r, cmapDiv] = visual_config
    

    
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import itertools


def singVarCorr(shap_values, nComp):
    num = shap_values.shape[1]

    combos = []
    for subset in itertools.combinations(np.arange(num), nComp):
        combos.append(subset)
    combos = np.array(combos, dtype=object)
    
    pcaMatrix = np.zeros((num, num))
    for subset in combos:
        pca = PCA(n_components=2)
        pca.fit(shap_values[:, [subset[0], subset[1]]])
        [first, second] = pca.explained_variance_ratio_
        pcaMatrix[subset[1], subset[0]] = first
        pcaMatrix[subset[0], subset[1]] = second
    return pcaMatrix

def biVarCorr(shap_values, fname, remove=" ", standard=True, nComp = 2, singVar=False):
    events = shap_values.shape[0]
    num = shap_values.shape[1]

    combos = []
    for L in range(1,nComp+1):
        for subset in itertools.combinations(np.arange(num), L):
            combos.append(subset)
    if singVar != False:
        combos.append([singVar])
    combos = np.array(combos, dtype=object)

    numCombos = len(combos)
    addedSHAP = np.zeros((events, numCombos))
    namesArr = np.zeros(numCombos, dtype=object)
    for subset in range(numCombos):
        scombo = combos[subset]
        nameList = ""#np.array([], dtype=str)
        for i in range(len(scombo)):#nComp):
            shap0 = shap_values[:, scombo[i]]
            # shap1 = shap_values[:, scombo[1]]
            addedSHAP[:, subset] = addedSHAP[:, subset] + shap0 
            nameList = nameList + f"{fname[scombo[i]]} "
        namesArr[subset] = nameList #namesArr[subset].append(f"{fname[scombo[i]]}, ") # {fname[scombo[1]]}] "

    def runPCA(shapMat):
        pcaRes = np.zeros(numCombos)
        for subset in range(numCombos):
            pca = PCA(n_components=2)
            pca.fit(shapMat)
            pcaRes = pca.explained_variance_ratio_
        return pcaRes

    if remove != " ":
        for r in range(len(remove)):
            cut = np.where(np.char.find(np.array(namesArr, dtype=str), remove[r])>0)[0]
            addedSHAP = np.delete(addedSHAP, cut, axis = 1)
            namesArr = np.delete(namesArr, cut)
            print(f"Made Cut - new shape {addedSHAP.shape}")
    
    if standard:
        # print(addedSHAP)
        scaler = StandardScaler()
        scaler.fit(addedSHAP)
        # print(f"MEAN {scaler.mean_}")
        addedSHAP = scaler.transform(addedSHAP)
        # print(f"StandardScaler")
        # print(addedSHAP)
    
    print(addedSHAP)
    pcaRes = runPCA(addedSHAP)
    return pcaRes, namesArr
