import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from util import data_path


# 获取数据文件路径
files_path = data_path('Figure4/ECG/')
# 遍历所有CSV文件并绘制散点图
for file in os.listdir(files_path):
    if file.endswith('pos.csv'):
        full_path = os.path.join(files_path, file)
        print(f'正在处理文件: {full_path}')
        # 读取数据
        data = pd.read_csv(full_path, sep='\t', header=None, skiprows=1, names=['reference','measurements'])
        
        # 计算相关系数
        correlation = np.corrcoef(data['reference'], data['measurements'])[0,1]
        # 计算线性回归
        slope, intercept = np.polyfit(data['reference'], data['measurements'], 1)
        fit_line = slope * data['reference'] + intercept
        
        # 创建新的图表
        plt.figure(figsize=(10, 10))
        
        # 绘制散点图和拟合线
        plt.scatter(data['reference'], data['measurements'], alpha=0.3, s=100, edgecolor='none', color='gray')
        plt.plot(data['reference'], fit_line, color='red', linestyle='--')
        
        # 添加相关系数和拟合方程文本
        plt.text(0.05, 0.95, f'R = {correlation:.3f}\ny = {slope:.3f}x + {intercept:.3f}', 
                transform=plt.gca().transAxes, fontsize=12)
        
        # 设置图表属性
        plt.xlabel('Reference', fontsize=12)
        plt.ylabel('Measurements', fontsize=12)
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        
        
        # 调整布局并显示
        plt.tight_layout()
        plt.savefig(f'D:/OneDrive - UW-Madison/git/Lock_in_camera/data/v4/Figure4/ECG/{file}.svg')
        plt.show()

