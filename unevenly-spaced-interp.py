# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 09:50:06 2021

@author: xaver
"""
# from stackoverflow, scipy.interp1d handling unvenly spaced data
import re
import numpy as np
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt

text = '--1--4-----5-3-22---55-'
parts = [c for c in re.split(r'(-|\d+)', text) if c]
data = np.array([(x, int(y)) for x, y in enumerate(parts) if y != '-'])
x, y = data.T
f = interpolate.interp1d(x, y, kind='cubic')

newx = np.linspace(x.min(), x.max())
newy = f(newx)

plt.plot(newx, newy)
plt.scatter(newx, newy)
plt.scatter(x, y, s=20)
plt.show()