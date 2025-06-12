import numpy as np
import pandas as pd

# 定义文件路径
file1_path = r'C:\Users\bliu259-admin\Documents\uw-ppg-project\data\ecg_raw\ECG_1124.txt'
file2_path = r'C:\Users\bliu259-admin\Documents\uw-ppg-project\data\ppg_reference\1124\interpolated_results.txt'
output_path = r'C:\Users\bliu259-admin\Documents\uw-ppg-project\data\network\1124\rawdata\ECG_referencePPG.npy'

# 读取文件
file1 = pd.read_csv(file1_path, sep='\t', header=None, names=['time', 'data1'])
ref = pd.read_csv(file2_path, sep='\t', header=None, names=['time', 'data2'])

# 将时间列转换为时间格式
file1['time'] = pd.to_datetime(file1['time'], format='%H:%M:%S.%f')
ref['time'] = pd.to_datetime(ref['time'], format='%H:%M:%S.%f')


# 格式化为字符串，保留秒的小数点后两位
file1['time'] = file1['time'].dt.strftime('%H:%M:%S.%f').str[:-4]
ref['time'] = ref['time'].dt.strftime('%H:%M:%S.%f').str[:-4]


# 找到所有文件中共同的时间点
common_times = np.intersect1d(file1['time'],ref['time'])


# 从每个文件中提取公共的时间点
file1_common = file1[file1['time'].isin(common_times)].reset_index(drop=True)
ref_common = ref[ref['time'].isin(common_times)].reset_index(drop=True)


# 确保文件中的数据是非空的，并且各文件都同时存在有效数据
valid_data1 = file1_common['data1'].notna() & (file1_common['data1'] != 0)
valid_data_ref = ref_common['data2'].notna()& \
                   (ref_common['data2'] != 0)


# 查找所有文件都同时有数据的时间点
valid_data = valid_data1 & valid_data_ref

# 提取所有有效的300个连续时间段的数据
selected_data = []
for i in range(len(valid_data) - 299):
    if valid_data.iloc[i:i + 300].all():  # 确保这300个数据点都是有效的
        selected_file1 = file1_common.iloc[i:i + 300]
        selected_ref = ref_common.iloc[i:i + 300]


        # 合并 data1 和计算得到的均值
        combined_data = np.hstack((
            selected_file1['data1'].values.reshape(-1, 1), 
            selected_ref['data2'].values.reshape(-1, 1)
        ))


        # 添加到最终结果数组
        selected_data.append(combined_data)

# 转换为 NumPy 数组并保存
selected_data = np.array(selected_data)
print('shape_of_selected_data:', selected_data.shape)

np.save(output_path, selected_data)