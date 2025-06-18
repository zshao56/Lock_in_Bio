import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from util import data_path

file_names = ['Gaming', 'reading', 'Sleeping', 'Talking']

for file_name in file_names:
    if file_name == 'Talking':
        files = data_path('Figure2/PPG/' + file_name + '.csv')
        output = data_path('Figure2/PPG/' + file_name + '.svg')
        file = pd.read_csv(files, header=None, sep = '\t')
        x = file[0] / 50
        y1 = file[1]
        y2 = file[2]
        plt.figure(figsize=(12, 8))
        plt.plot(x, y1, color='#EF0000', linewidth=2)
        plt.plot(x, y2, color='#336699', linewidth=2)
        plt.xlabel('time (s)')
        plt.ylabel('Amplitude')
        plt.xlim(0 , 4)
        plt.ylim(-30, 140)
        plt.xticks(np.arange(0, 5, 1))
        plt.yticks([])
        plt.savefig(data_path(output))  
        plt.show()
    else:
        files = data_path('Figure2/PPG/' + file_name + '.csv')
        output = data_path('Figure2/PPG/' + file_name + '.svg')
        file = pd.read_csv(files, header=None, sep = '\t')
        x = file[0] / 50
        y = file[1]
        plt.figure(figsize=(12, 8))
        plt.plot(x, y, color='#FEC211', linewidth=2)
        plt.xlabel('time (s)')
        plt.ylabel('Amplitude')
        plt.xlim(0 , 4)
        plt.ylim(-30, 140)
        plt.xticks(np.arange(0, 5, 1))
        plt.yticks([])
        plt.savefig(data_path(output))    
        plt.show()
