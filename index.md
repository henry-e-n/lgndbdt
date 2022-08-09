## Welcome to lgndbdt Wiki


### Description

'lgndbdt' contains the python-based coding pipelines for :
  - Extracting pulse shape parameters from raw nuclear event waveform data (collected by Germanium Detectors). 
  - Training and visualizing an interpretable Boosted Decision Tree Machine Learning Model using the LightGBM and SHAP packages. 

```markdown
# There are two main functions that run the parameter extraction and BDT codes respectively.
# They can be accessed via: 

import lgndbdt
lgndbdt.raw_to_bdt.PSD_Extraction.parameter_extraction()
lgndbdt.raw_to_bdt.BDT.run_BDT()
```

