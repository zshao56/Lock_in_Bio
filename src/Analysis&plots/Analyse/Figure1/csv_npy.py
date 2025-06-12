import numpy as np
import matplotlib.pyplot as plt
from util import data_path
import pandas as pd

light = pd.read_csv(data_path('Figure1/light_5000K.csv'),header=0, names=['wavelength', 'intensity'],sep='\t')
solar = pd.read_csv(data_path('Figure1/solar_5000K.csv'),header=0, names=['wavelength', 'intensity'],sep='\t')

l_wvl = light['wavelength']
l_intensity = light['intensity']
s_wvl = solar['wavelength']
s_intensity = solar['intensity']

l_5000K = np.zeros((400,2))
l_5000K[:,0] = l_intensity
l_5000K[:,1] = s_intensity

print(l_5000K.shape)

np.save(data_path('Figure1/381_780.npy'), l_5000K)
np.save(data_path('Figure1/381_780.npy'), l_5000K)



