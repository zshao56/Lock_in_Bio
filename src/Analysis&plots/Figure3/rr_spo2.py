import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util import data_path

Reference_file = data_path('Figure3/RR/RR_reference.csv')
Device_file = data_path('Figure3/RR/RR_lockin.csv')
camera_file = data_path('Figure3/RR/RR_regular.csv')
  
df1 = pd.read_csv(Reference_file, sep=',', header=None, skiprows=1) 
x1, y1 = 0.96, 100  # 第一个点
x2, y2 = 1.43, 90   # 第二个点
slope1 = (y2 - y1) / (x2 - x1)
intercept1 = y1 - slope1 * x1

df1['normalized_column'] = df1[5] * slope1 + intercept1
df1['normalized_column'] = df1['normalized_column'].apply(lambda x: 100 if x > 100 else x)



df2 = pd.read_csv(Device_file, sep=',', header=None, skiprows=1) 
x11, y11 = 1.155, 100  # 第一个点
x22, y22 = 1.862, 90   # 第二个点
slope2 = (y22 - y11) / (x22 - x11)
intercept2 = y11 - slope2 * x11

df2['normalized_column'] = df2[5] * slope2 + intercept2
df2['normalized_column'] = df2['normalized_column'].apply(lambda x: 100 if x > 100 else x)

df3 = pd.read_csv(camera_file, sep=',', header=None, skiprows=1) 
x111, y111 = 1.05, 100  # 第一个点
x222, y222 = 2.21, 90   # 第二个点
slope3 = (y222 - y111) / (x222 - x111)
intercept3 = y111 - slope3 * x111

df3['normalized_column'] = df3[5] * slope3 + intercept3
df3['normalized_column'] = df3['normalized_column'].apply(lambda x: 100 if x > 100 else x)



plt.figure(figsize=(10, 6))
plt.plot(df1['normalized_column'], label='Reference')
plt.plot(df2['normalized_column'], label='Lockin')
plt.plot(df3['normalized_column'], label='regular camera')
plt.legend()
plt.show()


output_file = data_path('Figure3/SpO2/spo2.csv')

result_df = pd.DataFrame({
    'Reference': df1['normalized_column'],  
    'Lockin': df2['normalized_column'],   
    'regular camera': df3['normalized_column']                              # 数据1的AC值
})
# 保存结果到CSV文件，默认使用逗号分隔
result_df.to_csv(output_file, index=False, sep='\t')


