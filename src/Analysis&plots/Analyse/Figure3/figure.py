import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util import data_path

# 读取CSV文件
input_file = data_path('Figure4/PPG_lockin.csv')  # 输入文件名
# 假设CSV文件没有表头，列分别是标数、数据1、数据2
df = pd.read_csv(input_file, sep='\t', header=None, names=['count', 'visible', 'NIR'])

# 提取列数据
labels = df['count']  # 第一列：标数    
vis = df['visible']   # 第二列：数据1
IR = df['NIR']   # 第三列：数据2

# 绘制图形
plt.figure(figsize=(10, 6)) 
plt.plot(vis[::5], label='Visible')
plt.plot(IR[::5], label='NIR')
# 添加图例
plt.legend()
# 显示图形
plt.show()