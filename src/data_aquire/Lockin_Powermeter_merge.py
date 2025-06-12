import numpy as np
import pandas as pd

# 读取两个txt文件
file1 = pd.read_csv(r'C:\Users\bliu259-admin\Documents\uw-ppg-project\data\ppg_reference\1118\interpolated_results.txt', sep='\t',
                    header=None, names=['time', 'data1'])
file2 = pd.read_csv(r'C:\Users\bliu259-admin\Documents\uw-ppg-project\data\ppg_denoised_lockin\1118\merge\right_cheek.txt',
                    sep='\t', header=None, names=['time', 'data2', 'data3', 'data4', 'data5'])

path = r'C:\Users\bliu259-admin\Documents\uw-ppg-project\data\network\rowdata\right_cheek.npy'
# 将时间列转换为时间格式，确保精确匹配
file1['time'] = pd.to_datetime(file1['time'], format='%H:%M:%S.%f')
file2['time'] = pd.to_datetime(file2['time'], format='%H:%M:%S.%f')


# 格式化为字符串，保留秒的小数点后两位
file1['time'] = file1['time'].dt.strftime('%H:%M:%S.%f').str[:-4]
file2['time'] = file2['time'].dt.strftime('%H:%M:%S.%f').str[:-4]

# 找到两个文件中都包含的时间点
common_times = np.intersect1d(file1['time'], file2['time'])

# 从每个文件中提取公共的时间点
file1_common = file1[file1['time'].isin(common_times)].reset_index(drop=True)
file2_common = file2[file2['time'].isin(common_times)].reset_index(drop=True)

# 确保文件中的数据是非空的，并且两个文件都同时存在数据
valid_data1 = file1_common['data1'].notna() & (file1_common['data1'] != 0)
valid_data2 = file2_common[['data2', 'data3', 'data4', 'data5']].notna().all(axis=1) & (
            file2_common[['data2', 'data3', 'data4', 'data5']] != 0).all(axis=1)

# 查找同时有数据的时间点
valid_data = valid_data1 & valid_data2

# 提取所有有效的300个连续时间段的数据
selected_data = []
for i in range(len(valid_data) - 299):
    if valid_data.iloc[i:i + 300].all():  # 确保这300个数据点都是有效的
        selected_file1 = file1_common.iloc[i:i + 300]
        selected_file2 = file2_common.iloc[i:i + 300]

        # 合并 data1 和 data2-5
        combined_data = np.hstack((selected_file1['data1'].values.reshape(-1, 1),
                                   selected_file2[['data2', 'data3', 'data4', 'data5']].values))
        
        # 添加到最终结果数组
        selected_data.append(combined_data)

print('shape_of_selected_data:', np.array(selected_data).shape)

# 将所有选中的300个点的组合保存为一个npy文件
np.save(path, np.array(selected_data))
