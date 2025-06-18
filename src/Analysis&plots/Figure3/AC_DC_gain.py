import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util import data_path

# 定义采样频率和时间间隔
fs = 60  # 采样频率为60Hz
window_sizes = [30, 60, 60]  # 每秒的数据点数
# 读取CSV文件
PPG_files = ['reference.csv', 'regular.csv', 'lockin.csv']
RR_files = ['RR_reference.csv', 'RR_regular.csv', 'RR_lockin.csv']
lens = len(PPG_files)

RRs = []
for i in range(lens):
    window_size = window_sizes[i]
    input_file = data_path('Figure3/PPG/' + PPG_files[i])  # 输入文件名
    output_file = data_path('Figure3/RR/' + RR_files[i])  # 输出文件名


    # 假设CSV文件没有表头，列分别是标数、数据1、数据2
    df = pd.read_csv(input_file, sep='\t', header=None, skiprows=1, names=['Label', 'visible', 'NIR'])

    # 提取列数据
    labels = df['Label']  # 第一列：标数
    vis = df['visible']   # 第二列：数据1
    IR = df['NIR']   # 第三列：数据2

    # plt.figure(figsize=(10, 6))
    # plt.plot(vis, label='visible')
    # plt.plot(IR, label='IR')
    # plt.legend()
    # plt.ylim([-50,500])
    # plt.show()
    # exit()


    # 初始化结果列表
    dc1_list, ac1_list = [], []
    dc2_list, ac2_list = [], []
    # 非重叠滑动窗口处理数据
    for i in range(0, len(vis), window_size):  # 每次跳过一个窗口大小
        # 提取当前窗口的数据
        window_data1 = vis[i:i + window_size]
        window_data2 = IR[i:i + window_size]
        # 如果窗口数据不足（例如最后一段数据），跳过
        if len(window_data1) < window_size or len(window_data2) < window_size:
            continue
        # 计算DC值（滑动平均）
        dc1 = np.mean(window_data1)
        dc2 = np.mean(window_data2)
        # 计算AC值（最大值与最小值的差）
        ac1 = np.max(window_data1) - np.min(window_data1)
        ac2 = np.max(window_data2) - np.min(window_data2)
        # 将结果添加到列表中
        dc1_list.append(dc1)
        ac1_list.append(ac1)
        dc2_list.append(dc2)
        ac2_list.append(ac2)
    # 将列表转换为numpy数组以进行数组运算
    ac1_array = np.array(ac1_list)
    dc1_array = np.array(dc1_list)
    ac2_array = np.array(ac2_list)
    dc2_array = np.array(dc2_list)

    # 计算比率比（RR），避免除以零的错误
    a1 = 1
    b1 = 80
    a2 = 1
    b2 = 80
    ac1_array = a1 * ac1_array
    ac2_array = a2 * ac2_array
    dc1_array = a1 * dc1_array + b1
    dc2_array = a2 * dc2_array + b2
    RR = np.divide(np.divide(ac1_array, dc1_array), np.divide(ac2_array, dc2_array), where=(dc1_array!=0) & (dc2_array!=0))

    plt.figure(figsize=(10, 6))
    # plt.plot(ac1_array, label='AC_Visible')
    # plt.plot(ac2_array, label='AC_NIR')
    # plt.plot(dc1_array, label='DC_Visible')
    # plt.plot(dc2_array, label='DC_NIR')
    plt.plot(RR, label='RR')
    # plt.ylim([0,1])
    plt.legend()
    # plt.ylim([0,5])
    # plt.savefig(data_path('Figure4/RR.png'))
    plt.show()

    # 创建结果DataFrame
    result_df = pd.DataFrame({
        'Label': labels[::window_size][:len(dc1_list)],  # 每秒取一个标数
        'DC1': dc1_list,                                 # 数据1的DC值
        'AC1': ac1_list,                                 # 数据1的AC值
        'DC2': dc2_list,                                 # 数据2的DC值
        'AC2': ac2_list,                                  # 数据2的AC值
        'RR': RR
    })
    # 保存结果到CSV文件，默认使用逗号分隔
    result_df.to_csv(output_file, index=False)
    RRs.append(RR)
    print(f"处理完成，结果已保存到 {output_file}")

