import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from util import data_path

files_name = ['reading', 'gaming', 'sleeping', 'talking1', 'talking2']

fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.35
index = np.arange(len(files_name))

lockin_means = []
lockin_min_errors = []
lockin_max_errors = []
regular_means = []
regular_min_errors = []
regular_max_errors = []

# 收集所有数据
for i in range(len(files_name)):
    file_name = files_name[i]
    data_file = data_path(f'Figure2/heartrate/{file_name}.csv')
    data = pd.read_csv(data_file, header=0, sep='\t', skiprows=1,
                      names=['reference', 'lockin', 'regular'])
    
    # 计算误差
    lockin_error = abs(data['lockin'] - data['reference'])
    regular_error = abs(data['regular'] - data['reference'])
    
    # 计算均值
    lockin_means.append(lockin_error.mean())
    regular_means.append(regular_error.mean())
    
    # 计算最小值和最大值（用于误差棒）
    lockin_min = lockin_error.min()
    lockin_max = lockin_error.max()
    regular_min = regular_error.min()
    regular_max = regular_error.max()
    
    # 计算距离均值的距离（用于误差棒）
    lockin_min_errors.append(lockin_means[i] - lockin_min)
    lockin_max_errors.append(lockin_max - lockin_means[i])
    regular_min_errors.append(regular_means[i] - regular_min)
    regular_max_errors.append(regular_max - regular_means[i])

# 创建误差棒
lockin_bars = ax.bar(index - bar_width/2, lockin_means, bar_width, 
                    color='#e6e6e6', edgecolor='#1a1a1a', linewidth=1,
                    label='Lockin')

regular_bars = ax.bar(index + bar_width/2, regular_means, bar_width, 
                     color='#999999', edgecolor='#1a1a1a', linewidth=1,
                     label='Regular')

# 添加误差棒（最小值和最大值）
# ax.errorbar(index - bar_width/2, lockin_means, 
#            yerr=[np.zeros_like(lockin_max_errors), lockin_max_errors],
#            fmt='none', ecolor='#1a1a1a', capsize=5, capthick=1, elinewidth=1)

# ax.errorbar(index + bar_width/2, regular_means, 
#            yerr=[np.zeros_like(lockin_max_errors), regular_max_errors],
#            fmt='none', ecolor='#1a1a1a', capsize=5, capthick=1, elinewidth=1)

# 添加数据标签
# for i, value in enumerate(lockin_means):
#     ax.text(i - bar_width/2+0.07, value + 0.5, f'{value:.2f}', ha='center', va='bottom', fontsize=9)

# for i, value in enumerate(regular_means):
#     ax.text(i + bar_width/2+0.07, value + 0.5, f'{value:.2f}', ha='center', va='bottom', fontsize=9)

# 设置X轴标签和刻度
ax.set_xticks(index)
ax.set_xticklabels([name.capitalize() for name in files_name])

# 添加图例
ax.legend(loc='upper right')

# 设置Y轴
ax.set_ylabel('Mean heartrate error (BPM)', fontsize=12)
ax.set_ylim([0, 10])  # 最小误差可能为0，所以从0开始
ax.set_yticks([0, 5, 10])
# 添加网格线以便于阅读
# ax.grid(axis='y', linestyle='--', alpha=0.7)

# 美化图表
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.title('Heart Rate Measurement Error by Method and Activity', fontsize=14)

# 保存和显示
plt.tight_layout()
plt.savefig(data_path('Figure2/heartrate/heartrate_error.svg'), dpi=300, bbox_inches='tight')
plt.show()