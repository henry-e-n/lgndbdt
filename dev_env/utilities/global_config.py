from matplotlib.colors import LinearSegmentedColormap, ListedColormap



terminalCMAP = ['#4B9CD3', '#13294B', '#EF426F', '#00A5AD', '#FFD100', '#C4D600'] # {Carolina, DBlue, Pink, Cyan, Yellow, Piss}
cmapNormal   = LinearSegmentedColormap.from_list("Custom", ["#151515", '#13294B', '#4B9CD3', "#F4E8DD"], N=50)#, '#C8A2C8'
cmapNormal_r = cmapNormal.reversed("cmapNormal_r")
cmapDiv      = LinearSegmentedColormap.from_list("Custom", ['#13294B', "#F4E8DD", '#4B9CD3'], N=50) #["#EF426F", '#F4E8DD', '#00A5AD'], N=50)
