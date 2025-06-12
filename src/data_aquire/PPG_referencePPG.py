import numpy as np
import pandas as pd

# 定义文件路径
file1_path = r'C:\Users\bliu259-admin\Documents\uw-ppg-project\data\ppg_reference\1124\interpolated_results.txt'
right_cheek_path = r'C:\Users\bliu259-admin\Documents\uw-ppg-project\data\ppg_denoised_lockin\1124\merge\right_cheek.txt'
forehead_path = r'C:\Users\bliu259-admin\Documents\uw-ppg-project\data\ppg_denoised_lockin\1124\merge\forehead.txt'
left_cheek_path = r'C:\Users\bliu259-admin\Documents\uw-ppg-project\data\ppg_denoised_lockin\1124\merge\left_cheek.txt'
output_path = r'C:\Users\bliu259-admin\Documents\uw-ppg-project\data\network\rowdata\1124_averaged_data.npy'

# 读取文件
file1 = pd.read_csv(file1_path, sep='\t', header=None, names=['time', 'data1'])
right_cheek = pd.read_csv(right_cheek_path, sep='\t', header=None, names=['time', 'data2', 'data3', 'data4', 'data5'])
forehead = pd.read_csv(forehead_path, sep='\t', header=None, names=['time', 'data2', 'data3', 'data4', 'data5'])
left_cheek = pd.read_csv(left_cheek_path, sep='\t', header=None, names=['time', 'data2', 'data3', 'data4', 'data5'])

# 将时间列转换为时间格式
file1['time'] = pd.to_datetime(file1['time'], format='%H:%M:%S.%f')
right_cheek['time'] = pd.to_datetime(right_cheek['time'], format='%H:%M:%S.%f')
forehead['time'] = pd.to_datetime(forehead['time'], format='%H:%M:%S.%f')
left_cheek['time'] = pd.to_datetime(left_cheek['time'], format='%H:%M:%S.%f')

# 格式化为字符串，保留秒的小数点后两位
file1['time'] = file1['time'].dt.strftime('%H:%M:%S.%f').str[:-4]
right_cheek['time'] = right_cheek['time'].dt.strftime('%H:%M:%S.%f').str[:-4]
forehead['time'] = forehead['time'].dt.strftime('%H:%M:%S.%f').str[:-4]
left_cheek['time'] = left_cheek['time'].dt.strftime('%H:%M:%S.%f').str[:-4]

# 找到所有文件中共同的时间点
common_times = np.intersect1d(file1['time'], right_cheek['time'])
common_times = np.intersect1d(common_times, forehead['time'])
common_times = np.intersect1d(common_times, left_cheek['time'])

# 从每个文件中提取公共的时间点
file1_common = file1[file1['time'].isin(common_times)].reset_index(drop=True)
right_cheek_common = right_cheek[right_cheek['time'].isin(common_times)].reset_index(drop=True)
forehead_common = forehead[forehead['time'].isin(common_times)].reset_index(drop=True)
left_cheek_common = left_cheek[left_cheek['time'].isin(common_times)].reset_index(drop=True)

# 确保文件中的数据是非空的，并且各文件都同时存在有效数据
valid_data1 = file1_common['data1'].notna() & (file1_common['data1'] != 0)
valid_data_right = right_cheek_common[['data2', 'data3', 'data4', 'data5']].notna().all(axis=1) & \
                   (right_cheek_common[['data2', 'data3', 'data4', 'data5']] != 0).all(axis=1)
valid_data_forehead = forehead_common[['data2', 'data3', 'data4', 'data5']].notna().all(axis=1) & \
                      (forehead_common[['data2', 'data3', 'data4', 'data5']] != 0).all(axis=1)
valid_data_left = left_cheek_common[['data2', 'data3', 'data4', 'data5']].notna().all(axis=1) & \
                  (left_cheek_common[['data2', 'data3', 'data4', 'data5']] != 0).all(axis=1)

# 查找所有文件都同时有数据的时间点
valid_data = valid_data1 & valid_data_right & valid_data_forehead & valid_data_left

# 提取所有有效的300个连续时间段的数据
selected_data = []
for i in range(len(valid_data) - 299):
    if valid_data.iloc[i:i + 300].all():  # 确保这300个数据点都是有效的
        selected_file1 = file1_common.iloc[i:i + 300]
        selected_right = right_cheek_common.iloc[i:i + 300]
        selected_forehead = forehead_common.iloc[i:i + 300]
        selected_left = left_cheek_common.iloc[i:i + 300]

        # 计算每个文件 data2 到 data5 的平均值
        avg_right = selected_right[['data2', 'data3', 'data4', 'data5']].mean(axis=1)
        avg_forehead = selected_forehead[['data2', 'data3', 'data4', 'data5']].mean(axis=1)
        avg_left = selected_left[['data2', 'data3', 'data4', 'data5']].mean(axis=1)

        # 计算所有文件的均值的平均值
        overall_avg = np.mean([avg_right, avg_forehead, avg_left], axis=0).reshape(-1, 1)

        # 合并 data1 和计算得到的均值
        combined_data = np.hstack((
            selected_file1['data1'].values.reshape(-1, 1), 
            avg_left.values.reshape(-1, 1), 
            avg_forehead.values.reshape(-1, 1), 
            avg_right.values.reshape(-1, 1), 
            overall_avg.reshape(-1, 1)
        ))


        # 添加到最终结果数组
        selected_data.append(combined_data)

# 转换为 NumPy 数组并保存
selected_data = np.array(selected_data)
print('shape_of_selected_data:', selected_data.shape)

np.save(output_path, selected_data)
