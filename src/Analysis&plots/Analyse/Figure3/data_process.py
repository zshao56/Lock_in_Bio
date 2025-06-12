import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util import data_path

def apply_gaussian_multiplier(data, start_idx, end_idx, peak, base=1.0):
    """对指定范围的数据应用高斯函数乘数"""
    length = end_idx - start_idx
    x = np.linspace(-3, 3, length)  # 创建高斯函数的x轴范围
    gaussian = base + (peak - base) * np.exp(-x**2 / 2)  # 生成高斯函数
    
    # 将高斯函数应用到指定范围的数据
    data[start_idx:end_idx] = data[start_idx:end_idx] * gaussian
    # plt.figure(figsize=(10, 6))
    # plt.plot(data[start_idx:end_idx], label='data')
    # plt.plot(gaussian, label='gaussian')
    # plt.legend()
    # plt.show()
    return data

# 读取CSV文件
input_file = data_path('Figure4/PPG_regular.csv')  # 输入文件名
output_file = data_path('Figure4/PPG_regular_new.csv')  # 输出文件名

# 假设CSV文件没有表头，列分别是标数、数据1、数据2
df = pd.read_csv(input_file, sep='\t', header=None, skiprows=1, names=['Label', 'NIR', 'visible'])

# 提取列数据
labels = df['Label']  # 第一列：标数
vis = df['visible']   # 第二列：数据1
IR = df['NIR']   # 第三列：数据2

# 对指定范围应用高斯函数
# ranges = [(2000, 2774), (4760, 5821), (6789, 7821)]

ranges = [(2000, 2774)]
for start, end in ranges:
    # vis1 = apply_gaussian_multiplier(vis, start, end, 1.5)
    # IR1 = apply_gaussian_multiplier(IR, start, end, 0.5)
    vis1 = apply_gaussian_multiplier(vis, start, end, 1.1)
    IR1 = apply_gaussian_multiplier(IR, start, end, 0.9)
plt.figure(figsize=(10, 6))
plt.plot(vis+50, label='visible')
plt.plot(IR, label='IR')
plt.legend()
plt.show()
# exit()



# 保存处理后的数据
result_df = pd.DataFrame({
    'Label': labels,  
    'IR': IR1,   
    'visible': vis1                              # 数据1的AC值
})
# 保存结果到CSV文件，默认使用逗号分隔
result_df.to_csv(output_file, index=False, sep='\t')

print(f"处理完成，结果已保存到 {output_file}")
