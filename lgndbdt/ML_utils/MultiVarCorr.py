from extraction_utils.config import *
from sklearn.decomposition import PCA
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

def biVarCorr(shap_values, fname, remove=" ", nComp = 2):
    events = shap_values.shape[0]
    num = shap_values.shape[1]

    combos = []
    for subset in itertools.combinations(np.arange(num), nComp):
        combos.append(subset)
    combos = np.array(combos, dtype=object)

    numCombos = len(combos)
    addedSHAP = np.zeros((events, numCombos))
    namesArr = np.zeros(numCombos, dtype=object)
    for subset in range(numCombos):
        scombo = combos[subset]
        shap0 = shap_values[:, scombo[0]]
        shap1 = shap_values[:, scombo[1]]
        addedSHAP[:, subset] = shap0 + shap1
        namesArr[subset] = f"[{fname[scombo[0]]}, {fname[scombo[1]]}] "
    # print(addedSHAP.shape)

    def runPCA():
        pcaRes = np.zeros(numCombos)
        for subset in range(numCombos):
            pca = PCA()
            pca.fit(addedSHAP)
            pcaRes = pca.explained_variance_ratio_
        return pcaRes

    if remove != " ":
        cut = np.where(np.char.find(np.array(namesArr, dtype=str), remove)>0)[0]
        print(cut, addedSHAP.shape)
        addedSHAP = np.delete(addedSHAP, cut, axis = 1)
        namesArr = np.delete(namesArr, cut)
        print(cut, addedSHAP.shape)

    pcaRes = runPCA()
    return pcaRes, namesArr
