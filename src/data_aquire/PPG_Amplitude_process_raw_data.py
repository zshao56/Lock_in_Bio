import os
import numpy as np
import cv2
import mediapipe as mp
from scipy.interpolate import interp1d
from datetime import datetime, timedelta
import multiprocessing
import matplotlib.pyplot as plt
import scipy.signal as signal
import pywt

def interpolate_and_save_data(roi_mean, file_name, folder_path, original_framerate=51, target_interval=0.01):
    # 提取初始时间（从文件名的前9个字符）包括毫秒
    initial_time_str = file_name.split('_')[0]  # 如 "093648253"
    initial_time = datetime.strptime(initial_time_str[:6], '%H%M%S') + timedelta(milliseconds=int(initial_time_str[6:9]))
    
    # 获取原始帧数和时间序列
    num_frames = len(roi_mean[1, :])
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
    np.savetxt(output_path, data_to_save, fmt='%s', delimiter='\t')
    print(f'Interpolated data saved to {output_file_name}')

def detect(Amplitude):
    video_data = Amplitude
    num_frames, height, width = video_data.shape
    roi_mean = np.zeros([4, num_frames])
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.2)
    def normalize_to_range(frame, min_val=0, max_val=255):
        min_frame = np.min(frame)
        max_frame = np.max(frame)
        if max_frame == min_frame:
            return np.full_like(frame, min_val, dtype=np.uint8)
        normalized_frame = (frame - min_frame) / (max_frame - min_frame) * (max_val - min_val) + min_val
        return np.clip(normalized_frame, min_val, max_val).astype(np.uint8)
    last_x0, last_y0 = 0, 0
    last_mean = [220, 220, 220, 220]
    count = 0
    fail_flag = False
    for i in range(num_frames):
        frame = video_data[i]
        frame_real = video_data[i]
        frame = normalize_to_range(frame, min_val=0, max_val=255)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            ih, iw, _ = rgb_frame.shape
            # forehead_indices = [454, 323, 361]
            forehead_indices = [10, 338, 297]
            x0 = int(np.mean([landmarks.landmark[i].x for i in forehead_indices]) * iw)
            y0 = int(np.mean([landmarks.landmark[i].y for i in forehead_indices]) * ih)
            count += 1
            if abs(x0 - last_x0) > 10:
                last_x0, last_y0 = x0, y0
        x0, y0 = last_x0, last_y0
        last_x0, last_y0 = x0, y0
        a = 12
        b = 7
        roi_coords = [
            (x0 - 10 - a, y0 + b, x0 - 5 - a, y0 + 10 + b),
            (x0 - 5 - a, y0 + b, x0 - a, y0 + 10 + b),
            (x0 - a + 5, y0 + b, x0 + 10 - a, y0 + 10 + b),
            (x0 - a, y0 + b, x0 + 5 - a, y0 + 10 + b),
        ]
        for j, (x1, y1, x2, y2) in enumerate(roi_coords):
            if 0 <= x1 < width and 0 <= y1 < height and 0 <= x2 <= width and 0 <= y2 <= height:
                roi_region = frame_real[y1:y2, x1:x2]
                roi_mean[j, i] = np.mean(roi_region)
            else:
                roi_mean[j, i] = last_mean[j]
            last_mean[j] = roi_mean[j, i]
    detect_rate = count / num_frames
    if detect_rate < 0.2:
        print("Fail to detect your face, please face the camera...")
        fail_flag = True
    x0 = x0
    y0 = y0
    x = x0 - 50
    y = y0 - 50
    w = 100
    h = 100
    x1 = x0 - 10 - a
    x2 = x0 + 10 - a
    y1 = y0 - 10 + b
    y2 = y0 + 10 + b
    return roi_mean, x, y, w, h, x1, x2, y1, y2, fail_flag

def denoise(roi_mean):
    B1 = roi_mean[0, :]
    B2 = roi_mean[1, :]
    B3 = roi_mean[2, :]
    B4 = roi_mean[3, :]
    len_b1 = len(B1)
    end_ind = len_b1 - 10
    print('len:', len_b1)
    denoised_data = np.zeros([4, end_ind - 9])
    fs = 51
    winwithd = 0.01
    max_butter = 3.1
    A1 = B1[10:end_ind]
    A2 = B2[10:end_ind]
    A3 = B3[10:end_ind]
    A4 = B4[10:end_ind]
    regions = [A1, A2, A3, A4]
    for i, data in enumerate(regions):
        b, a = signal.butter(5, [1, max_butter], btype='bandpass', fs=fs)
        filtered_ppG = signal.filtfilt(b, a, data)
        c = pywt.wavedec(filtered_ppG, 'db4', level=5)
        thr = np.median(np.abs(c[-1])) / 0.6745 * np.sqrt(2 * np.log(len(filtered_ppG)))
        c_denoised = [pywt.threshold(ci, thr, mode='soft') for ci in c]
        wavelet_denoised = pywt.waverec(c_denoised, 'db4')
        window_size = round(winwithd * fs)
        denoised_ppg = np.convolve(wavelet_denoised, np.ones(window_size) / window_size,
                                   mode='same')
        denoised_data[i] = denoised_ppg
    return denoised_data

##main function
def main():

    file_list = [f for f in os.listdir(folder_path) if f.endswith('_Amplitude.npy')]
    with multiprocessing.Pool(processes=3) as pool:
        results = pool.map(task, file_list)
        print(results)

def task(file_name):

    out_folder_path = r'.\PPG raw data\txt_raw_data\1019'
    file_path = os.path.join(folder_path, file_name)
    text_name = file_name.replace("_Amplitude.npy", ".txt")
    if os.path.exists(os.path.join(out_folder_path, text_name)):
        print("Text file already exists for:", text_name)
        return (file_name, True)
    print(f'Processing file: {file_name}')
    Amplitude = np.load(file_path)
    roi_mean, x, y, w, h, x1, x2, y1, y2, fail_flag = detect(Amplitude)
    # denoised_data = denoise(roi_mean)

    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(Amplitude[100 - 2], cmap='hot', interpolation='nearest')
    plt.colorbar(label='Intensity')
    plt.title('Heatmap of video_data[1]')
    plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='black', facecolor='none'))
    plt.gca().add_patch(plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='white', facecolor='none'))
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.legend()
    plt.subplot(1, 2, 2)
    for i in range(4):
        plt.plot(roi_mean[i, :], label=f'ROI {i + 1}')
        plt.title(f'ROI Mean Curves - {file_name}')
        plt.xlabel('Time (frames)')
        plt.ylabel('Amplitude')
    # plt.subplot(1, 3, 3)
    # for i in range(4):
    #     plt.plot(denoised_data[i], label=f'Denoised ROI {i + 1}')
    #     plt.title('Denoised Data')
    #     plt.xlabel('Time (frames)')
    #     plt.ylabel('Amplitude')
    plot_filename = os.path.join(out_folder_path, file_name.replace('_Amplitude.npy', '.png'))
    plt.savefig(plot_filename, format='png', dpi=300)

    interpolate_and_save_data(roi_mean, file_name, out_folder_path)

    return (file_name, True)

folder_path = r'.\PPG raw data\1019'
if __name__ == '__main__':
    main()

