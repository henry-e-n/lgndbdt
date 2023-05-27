import tqdm
import numpy as np
from scipy.optimize import minimize, curve_fit

from utilities.global_config import terminalCMAP

###################################################################
###################################################################
def dp0fx(popt, wfArray):
    """
    Function: Pole Zero function

    Parameters:
        - popt: [tau1, tau2, f] 
            - tau1, tau2 - 2 exponential decay terms
            - f - relation fraction
        - wfArray: [len(Waveforms), ADC values len(cells)]
    
    Returns:
        - y_out: mean of standard deviation of fit waveforms
    """
    [tau1, tau2, f] = popt
    y_out = np.zeros(wfArray.shape[0])
    max_amp = 10
    
    for wf in range(wfArray.shape[0]):
        wf_in = wfArray[wf,:]
        wf_in = (wf_in-wf_in.min())/(wf_in.max()-wf_in.min())*max_amp
        wf_out = np.zeros(len(wf_in))
        # Defines the constant terms
        const1 = 1 / tau1
        const2 = 1 / tau2
        # Defines the exponential terms
        exp1 = np.exp(-1 / tau1)
        exp2 = np.exp(-1 / tau2)
        
        frac = f
        # Sets initial of output to same as input
        wf_out[0] = wf_in[0]
        e1 = e2 = wf_in[0]
        e3 = 0
         
        for i in range(1, len(wf_in), 1): # Iterates over rest of wf
            e1 += wf_in[i] - e2 + e2*const1
            e3 += wf_in[i] - e2 - e3*const2
            e2  = wf_in[i]
            wf_out[i] = e1 - frac*e3

        maxInd = np.argmax(wf_in) # Index of max
        pre_wf = wf_in[:maxInd] # waveform before max
        
        peak = np.max(wf_in)
        peakInd = wf_in.argmax()
        val97 = 0.97*peak
        diffArr = np.absolute(wf_in - val97)
        closestFromPeak = diffArr[peakInd:].argmin()
        ind97 = peakInd+closestFromPeak
        y_out[wf] = (np.std(wf_out[(ind97):]))
    
    return np.mean(y_out)


###################################################################
###################################################################

def dp0Vis(popt, wfArray):
    [tau1, tau2, f] = popt
    wfInAdj = np.zeros(wfArray.shape)
    wfCorr = np.zeros(wfArray.shape)
    max_amp = 10
    trash = []
    for wf in tqdm(range(wfArray.shape[0]),
                    desc="Applying P0 to waveforms......",
                    colour = terminalCMAP[1]):
        wf_in = wfArray[wf,:]
        if wf_in.min()<-5000:
            trash.append(wf)
        else:
            wf_in = (wf_in-wf_in.min())/(wf_in.max()-wf_in.min())*max_amp
            wfInAdj[wf, :] = wf_in
            wf_out = np.zeros(len(wf_in))
            # Defines the constant terms
            const1 = 1 / tau1
            const2 = 1 / tau2
            # Defines the exponential terms
            exp1 = np.exp(-1 / tau1)
            exp2 = np.exp(-1 / tau2)
            
            frac = f
            # Sets initial of output to same as input
            wf_out[0] = wf_in[0]
            e1 = e2 = wf_in[0]
            e3 = 0
            
            for i in range(1, len(wf_in), 1): # Iterates over rest of wf
                e1 += wf_in[i] - e2 + e2*const1
                e3 += wf_in[i] - e2 - e3*const2
                e2  = wf_in[i]
                wfCorr[wf, i] = e1 - frac*e3
    return wfInAdj, wfCorr, trash

###################################################################
###################################################################

def get_PZ(vals, popt, numWave = 100):
    """
    Function: Performs P0 correction on waveforms

    Parameters:
        - vals: Waveform ADC values [len(numWave), len(Cells)]
        - popt: Boolean 0, runs minimization on P0 function to determine slope correction, uses input popt
        - *numWave: number of waveforms
    
    Returns:
        - wfIn: input waveforms in corresponding data structure
        - wfCorr: Corrected waveforms
    """
    if popt == 0:
        if numWave >= 10:
            dp0Num = 10
        else:
            dp0Num = numWave
            
        for i in tqdm(range(1),
                        desc   = "Running PZ minimization.......",
                        colour = terminalCMAP[0]):  
            res = minimize(dp0fx,
                        [72*40, 2.1*40, 0.0105], 
                        args    = vals[:dp0Num,:], 
                        method  = 'Nelder-Mead', 
                        tol     = 1e-4, 
                        bounds  = ((60*40, 180*40), (1, 5*40),(0.01,0.022))) # 90*40 0.012
        
        popt = tuple(res.x)
    wfIn, wfCorr, trashPZ  = dp0Vis(popt, vals[:numWave,:])
    return wfIn, wfCorr, trashPZ