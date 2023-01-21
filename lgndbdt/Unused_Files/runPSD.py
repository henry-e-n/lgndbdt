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


def run_psd():
    import extraction_utils.config
    importlib.reload(extraction_utils.config)
    from extraction_utils.config import detName
    print(f"runPSD {detName}")

    mkdir(detName)
    calPar, fitResults, peakIndex = calibration() # Calibrate Energy
    paramArr, paramKeys = extract_waveforms(fitResults, peakIndex) # Return Selection Peak Criteria

    paramArr = cleanData(paramArr)

    # run Extraction
    psd_extraction(paramArr, paramKeys)

    return