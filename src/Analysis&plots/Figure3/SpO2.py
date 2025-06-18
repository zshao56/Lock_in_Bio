import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util import data_path

spo2_path = data_path('Figure3/SpO2/SpO2.csv')
spo2_df = pd.read_csv(spo2_path, header=None, names=['reference', 'regular', 'lockin'], sep='\t', skiprows=1)

reference = spo2_df['reference'].values
regular = spo2_df['regular'].values
lockin = spo2_df['lockin'].values

plt.figure(figsize=(10, 6))
plt.plot(reference, label='reference', color='black')
plt.plot(regular, label='regular', color='blue')
plt.plot(lockin, label='lockin', color='orange')
plt.legend()
plt.savefig(data_path('Figure3/SpO2/SpO2.svg'))
plt.show()