## plot csv
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/v4/Figure2/long_time.csv', sep='\t')



plt.plot(df['counts'], df['regular'], label='regular')
plt.plot(df['counts'], df['Lock-in'], label='Lock-in')
plt.legend()
plt.savefig('data/v4/Figure2/cycles.svg')
plt.show()



