import sys
import numpy as np
import matplotlib.pyplot as pl

#import seaborn as sns
#palette = sns.color_palette()
#sns.set()

file = sys.argv[1]
data = np.loadtxt(file)
y1, y2, y3, t = data
pl.plot(t, y1, 'o')
pl.plot(t, y2, 'o')
pl.plot(t, y3, 'o')
pl.show()
