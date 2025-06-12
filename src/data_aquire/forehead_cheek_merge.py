import os
import numpy as np

# 文件夹路径
input_folder = r"C:\Users\bliu259-admin\Documents\uw-ppg-project\data\network\rowdata"  # 替换为实际的 .npy 文件夹路径
output_file = r"C:\Users\bliu259-admin\Documents\uw-ppg-project\data\network\rowdata\merge.npy"  # 合并后的文件路径

# 初始化一个列表来存储所有数据
data_list = []

# 遍历文件夹中的 .npy 文件
for filename in os.listdir(input_folder):
    if filename.endswith(".npy"):
        file_path = os.path.join(input_folder, filename)
        print(f"Loading file: {file_path}")
        data = np.load(file_path)
        data_list.append(data)

# 合并数据（按行堆叠）
merged_data = np.vstack(data_list)  # 垂直合并，按行叠加

print(merged_data.shape)

# 保存合并后的文件
np.save(output_file, merged_data)

print(f"Data merged and saved to {output_file}")
