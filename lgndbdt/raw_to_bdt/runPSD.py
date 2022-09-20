import importlib
import extraction_utils
importlib.reload(extraction_utils)
from extraction_utils.config import *

from extraction_utils.raw_to_calibration import *
from extraction_utils.calibration_to_peakdata import *
from extraction_utils.Extraction import *
from extraction_utils.h5utils import paramExtract




def mkdir(detectorName):
    cwd = os.getcwd()
    path2make = os.path.join(cwd, f"{detectorName}") 
    try:
        os.mkdir(path2make)
    except FileExistsError:
        print(f"{detectorName} Path already exists")
    return

def cleanData(paramArr):
    print(f"Initial Shape: {paramArr[0].shape}")
    nans = []
    for i in range(len(paramArr)):
        for w in range(len(paramArr[i])):
            whereNan = np.where(np.isnan(paramArr[i][w]))
            if len(whereNan[0])>0:
                nans.append(w)
    # print(nans)
    for n in range(len(paramArr)):
        paramArr[n] = np.delete(paramArr[n], nans, 0)
    print(f"Final Shape: {paramArr[0].shape}")

    return paramArr


def runPSD():
    import extraction_utils.config
    importlib.reload(extraction_utils.config)
    from extraction_utils.config import detName
    print(f"runPSD {detName}")

    mkdir(detName)
    calPar, fitResults, peakIndex = calibration() # Calibrate Energy
    paramArr, paramKeys = getWFD(fitResults, peakIndex) # Return Selection Peak Criteria

    paramArr = cleanData(paramArr)

    # run Extraction
    extraction(paramArr, paramKeys)

    return