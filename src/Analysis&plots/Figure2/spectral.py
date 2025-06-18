import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from util import data_path

spectral = pd.read_csv(data_path('Figure2/Spectral/spectral.csv'), header=None, sep='\t', names=['wvl', 'yellow', 'black', 'red'])  
lockin = pd.read_csv(data_path('Figure2/Spectral/lockin.csv'), header=None, sep='\t', names=['wvl', 'yellow', 'black', 'red'])

plt.figure(figsize=(10, 10))
plt.plot(spectral['wvl'], spectral['yellow'], label='spectral', color='red')
plt.scatter(lockin['wvl'], lockin['yellow'], label='lockin', color='blue', facecolors='none', s=100)
plt.xlim(500, 700)
plt.ylim(0, 1)
plt.legend()
plt.savefig(data_path('Figure2/Spectral/Yellow.svg'),transparent=True)
plt.show()

plt.figure(figsize=(10, 10))
plt.plot(spectral['wvl'], spectral['black'], label='spectral', color='red')
plt.scatter(lockin['wvl'], lockin['black'], label='lockin', color='blue', facecolors='none', s=100)
plt.xlim(500, 700)
plt.ylim(0, 1)
plt.legend()
plt.savefig(data_path('Figure2/Spectral/Black.svg'),transparent=True)
plt.show()


fig = plt.figure(figsize=(10, 10))
# fig.patch.set_alpha(0.0)
plt.plot(spectral['wvl'], spectral['red'], label='spectral', color='red')
plt.scatter(lockin['wvl'], lockin['red'], label='lockin', color='blue', facecolors='none', s=100)
plt.xlim(500, 700)
plt.ylim(0, 1)
plt.legend()
plt.savefig(data_path('Figure2/Spectral/Red.svg'), transparent=True)
plt.show()
