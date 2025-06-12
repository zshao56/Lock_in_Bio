import cv2
import numpy as np
import mediapipe as mp
from scipy.interpolate import interp1d
import os
from datetime import datetime, timedelta
from OnlineGetData import *

def interpolate_and_save_data(Amplitude, roi_mean, file_name, folder_path, original_framerate=51, target_interval=0.01):
    # 提取初始时间（从文件名的前6个字符）
    initial_time_str = file_name.split('_')[0]  # 如 "135707"
    initial_time = datetime.strptime(initial_time_str, '%H%M%S')

    # 获取原始帧数和时间序列
    num_frames = Amplitude.shape[0]
    original_time_series = np.arange(0, num_frames / original_framerate, 1 / original_framerate)

    # 生成新的时间序列，间隔0.01秒
    new_time_series = np.arange(0, original_time_series[-1], target_interval)
    new_time_series_str = [(initial_time + timedelta(seconds=t)).strftime('%H:%M:%S.%f')[:-3] for t in new_time_series]

    # 对每个ROI均值进行插值
    interpolated_roi_mean = np.zeros((4, len(new_time_series)))
    for i in range(4):
        interpolator = interp1d(original_time_series, roi_mean[i, :], kind='linear')
        interpolated_roi_mean[i, :] = interpolator(new_time_series)

    # 组合时间和插值后的ROI数据
    data_to_save = np.column_stack([new_time_series_str] + [interpolated_roi_mean[i, :] for i in range(4)])

    # 保存为 .txt 文件，文件名为初始时间
    output_file_name = initial_time_str + '.txt'
    output_path = os.path.join(folder_path, output_file_name)

    np.savetxt(output_path, data_to_save, fmt='%s', delimiter='\t', header='Time\tROI1\tROI2\tROI3\tROI4')
    print(f'Interpolated data saved to {output_file_name}')

def main():
    # 文件夹路径
    folder_path = r'C:\Users\bliu259-admin\Desktop\REAL_time_detection\real_time_detection\ml raw data for 0909\PPG raw data\example'

    # 获取文件夹中所有 .npy 文件
    file_list = [f for f in os.listdir(folder_path) if f.endswith('_Amplitude.npy')]

    # 遍历文件并进行处理
    for file_name in file_list:
        # 构建完整文件路径
        file_path = os.path.join(folder_path, file_name)

        # 加载数据
        print(f'Processing file: {file_name}')
        Amplitude = np.load(file_path)

        # 假设 detect() 函数已经定义好
        roi_mean, x, y, w, h, x1, x2, y1, y2, fail_flag = detect(Amplitude)
        heartrate, denoised_data, peak, peak_locs = denoise(roi_mean)

        plt.figure(figsize=(10, 8))
        plt.subplot(1, 2, 2)
        for i in range(4):
            plt.plot(roi_mean[i, :], label=f'ROI {i + 1}')
            plt.title('ROI Mean Curves')
            plt.xlabel('Time (frames)')
            plt.ylabel('Amplitude')
            # plt.legend()

        # 第三个图：denoised_data的曲线图，标记峰位置
        plt.subplot(1, 2, 3)
        if fail_flag is True:
            pass
        else:
            for i in range(4):
                plt.plot(denoised_data[i], label=f'Denoised ROI {i + 1}')
                plt.scatter(peak_locs[i], denoised_data[i][peak_locs[i]], color='red', marker='o')
                plt.title('Denoised Data with Peaks')
                plt.xlabel('Time (frames)')
                plt.ylabel('Amplitude')
                # plt.legend()
        plt.show()
        # 调用插值和保存数据的函数
        interpolate_and_save_data(Amplitude, denoised_data, file_name, folder_path)

if __name__ == '__main__':
    main()
