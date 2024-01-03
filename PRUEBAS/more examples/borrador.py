# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 18:25:37 2024

@author: bruno
"""

from pylab import *
from numpy import *
import numpy as np

x = np.random.randint(0, 463.48468, size=(19468,))
print(x)

hopSize = 128
frameSize = 2048
sampleRate = 44100


# Visualize output pitch values
fig = plt.figure()
plot(range(19468), x, 'b')
n_ticks = 10
xtick_locs = [i * (19468 / 10.0) for i in range(n_ticks)]
xtick_lbls = [i * (19468 / 10.0) * hopSize / sampleRate for i in range(n_ticks)]
xtick_lbls = ["%.2f" % round(x,2) for x in xtick_lbls]
plt.xticks(xtick_locs, xtick_lbls)
ax = fig.add_subplot(111)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Pitch (Hz)')
suptitle("Predominant melody pitch")

#plt.savefig('./hola1')

# Visualize output pitch confidence
fig = plt.figure()
plot(range(19468), confidence, 'b')
n_ticks = 10
xtick_locs = [i * (19468 / 10.0) for i in range(n_ticks)]
xtick_lbls = [i * (19468 / 10.0) * hopSize / sampleRate for i in range(n_ticks)]
xtick_lbls = ["%.2f" % round(x,2) for x in xtick_lbls]
plt.xticks(xtick_locs, xtick_lbls)
ax = fig.add_subplot(111)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Confidence')
suptitle("Predominant melody pitch confidence")

#show()
#plt.savefig('./hola2')
