import numpy as np

from util import data_path

# 加载两个npy文件
data1 = np.load(data_path('0909.npy'))  # shape: (45795, 300, 5)
data2 = np.load(data_path('0921.npy'))  # shape: (5, 300, 5)

# 合并数组，沿着第一个维度(axis=0)进行拼接
merged_data = np.concatenate((data1, data2), axis=0)

# 验证合并后的形状是否为(45800, 300, 5)
print(merged_data.shape)  # 应输出 (45800, 300, 5)

# 保存合并后的数组
np.save(data_path('0909_0921.npy'), merged_data)

print("合并完成！")
