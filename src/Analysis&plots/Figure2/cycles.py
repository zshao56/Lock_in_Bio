import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util import data_path

# 读取CSV文件
lockin_file = data_path('Figure2/cycles/PPG_lockin.csv')  # 输入文件名
regular_file = data_path('Figure2/cycles/PPG_regular.csv')  # 输入文件名
# 假设CSV文件没有表头，列分别是标数、数据1、数据2
lock = pd.read_csv(lockin_file, sep='\t', header=None, names=['count', 'NIR', 'visible'], skiprows=1)
regular = pd.read_csv(regular_file, sep='\t', header=None, names=['count', 'NIR', 'visible'], skiprows=1)

# 提取列数据
labels = lock['count']  # 第一列：标数    
lockin_vis = lock['visible']   # 第二列：数据1
lockin_IR = lock['NIR']   # 第三列：数据2
labels = regular['count']  # 第一列：标数
regular_vis = regular['visible']   # 第二列：数据1
regular_IR = regular['NIR']   # 第三列：数据2

# 绘制图形
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))  # 2行1列布局

# 绘制红外信号
ax1.plot(regular_IR, label='regular', color='#f3adae')
ax1.plot(lockin_IR, label='lockin', color='#de1515')
ax1.axis('off')
ax1.set_title('IR Signal Comparison')  # 添加标题

# 绘制可见光信号
ax2.plot(regular_vis, label='regular', color='#a9a0c0')
ax2.plot(lockin_vis, label='lockin', color='#81277f')
ax2.axis('off')
ax2.set_title('Visible Signal Comparison')

plt.tight_layout()  # 自动调整子图间距
plt.savefig(data_path('Figure2/cycles/combined_signals.svg'))
plt.show()