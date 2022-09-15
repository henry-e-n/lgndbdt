from extraction_utils.raw_to_calibration import *
from extraction_utils.calibration_to_peakdata import *
from extraction_utils.Extraction import *
from extraction_utils.config import savePath

def mkdir(detectorName):
    cwd = os.getcwd()
    path2make = os.path.join(cwd, f"{detectorName}") 
    try:
        os.mkdir(path2make)
    except FileExistsError:
        print("File or Path already exists")
    return

mkdir(detName)
calPar, fitResults = calibration() # Calibrate Energy
paramArr, paramKeys = getWFD(fitResults, 2) # Return Selection Peak Criteria

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

paramArr = cleanData(paramArr)

# run Extraction
extraction(paramArr, paramKeys)