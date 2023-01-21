def AvsE2(times, values, dtimes, plots = [], numWF = 2500, currentEstimator = 100):
    """
    Arguments:
        - ntimes array - [#Waveforms, #TimeCells]
        - values - [#Waveforms, #TimeCells] - choose one waveform
        - dtimes - [#Waveforms, #TimeCells]
    
    Plots should either be:
        - False - no plots to be printed
        - E - energy array same length as times
    """
    Eest = np.zeros(numWF)
    maxA = np.zeros(numWF)

    for m in tqdm(range(numWF),
                  desc = "Getting Current Amplitude.....",
                  colour = terminalCMAP[0]):
        aMatrix = getA(times, values[m, :], dtimes)
        maxA[m] = np.max(aMatrix)
        Eest[m] = EfromA(aMatrix)
    
    if not plots==[]:
        plt.plot(plots[:len(maxA)], maxA,'.', color = cmapNormal(0.3))
        plt.xlabel("Energy (MeV)")
        plt.ylabel("Max Current (A)")
        plt.show()
    return maxA, Eest


    