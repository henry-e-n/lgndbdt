from extraction_utils.config import *
from sklearn.decomposition import PCA
import itertools


def multiVarCorr(shap_values, nComp):
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
