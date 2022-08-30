import os
import sys

# module_path = [os.path.abspath(os.path.join("C:/Users/henac/GitHub/ENAP_Personal/LegendMachineLearning/AnalysisExtraction"))]
# for path in module_path:
#     if path not in sys.path:
#         sys.path.append(path)

from extraction_utils.config     import *

from extraction_utils.h5Extract  import *
from ML_utils.BDTPrep    import *
from ML_utils.BDTTrain   import *
from extraction_utils.CleanData  import *
from extraction_utils.RawToTrain import *


"""
See config file for appropriate dependencies:
"""
def parameter_extraction(saveFiles = 1, numWave = -1, runP0 = 1):
    args = [detName, saveFiles, 1000, runP0] # detname, savefiles, numWave, p0

    rawToTrain(args, "DEP")
    rawToTrain(args, "FEP")

    return
    
if __name__ == "__main__":
    parameter_extraction()
