import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from util import data_path

# Read the data
Dia = pd.read_csv(data_path('Figure4/BP/Diastolic.csv'), sep='\t', header=None, names=['x', 'y'])

Sys = pd.read_csv(data_path('Figure4/BP/Systolic.csv'), sep='\t', header=None, names=['x', 'y'])

# Create figure with two subplots
plt.figure(figsize=(12, 10))

# Scatter plot
plt.subplot(1, 2, 1)
plt.scatter(Dia['x'], Dia['y'], alpha=0.5, label='Diastolic')
plt.scatter(Sys['x'], Sys['y'], alpha=0.5, label='Systolic')
plt.legend()
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Scatter Plot')

# Distribution plot
plt.subplot(1, 2, 2)
sns.histplot(Dia['x'], kde=True, color='red', label='Diastolic')
sns.histplot(Sys['x'], kde=True, color='blue', label='Systolic')
plt.xlabel('X values')
plt.title('Distribution of X values')

plt.tight_layout()
plt.show()