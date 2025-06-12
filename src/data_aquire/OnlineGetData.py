import os
import pywt
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import ipywidgets as widgets
import time
from datetime import datetime
import cv2
from scipy.io import savemat
# matplotlib ipympl
import mediapipe as mp
from harvesters.core import Harvester

from PIL import Image

#---------------- discovere the  camera
def disc_cam():
    h = Harvester()

    # if you have installed diaphus via the C4Utility Installer, the DIAPHUS_GENTL64_FILE
    # environment variable below is automatically set
    # Otherwise set it by uncommenting and completing the line below
    # %env DIAPHUS_GENTL64_FILE /<path>/<to>/diaphus.cti
    cti_file = os.getenv("DIAPHUS_GENTL64_FILE")

    h.add_file(cti_file)
    h.update()

    label = f"{len(h.device_info_list)} devices detected on the network"
    options = []
    for i, dev in enumerate(h.device_info_list):
        options.append(f"{dev.id_}  (SN: \"{dev.serial_number}\")")

    # just the widget below
    w = widgets.RadioButtons(options=options, description=label, disabled=False)
    print(w)  # print the Ip of the discovered Camera

    #--------------- Open the discovered camera
    camera = h.create(w.index)

    #--------------- configure triggering
    camera.remote_device.node_map.TriggerSelector.value = 'RecordingStart'
    camera.remote_device.node_map.TriggerMode.value = 'Off'
    print(
        f"RecordingStart default TriggerSource in DeviceMode=LockinCam: {camera.remote_device.node_map.TriggerSource.value}"
    )
    camera.remote_device.node_map.TriggerSelector.value = 'FrameStart'
    camera.remote_device.node_map.TriggerMode.value = 'On'
    print(
        f"FrameStart default TriggerSource in DeviceMode=LockinCam: {camera.remote_device.node_map.TriggerSource.value}"
    )
    camera.remote_device.node_map.TriggerSource.value = 'Software'

    def lia_config(sensitivity=0.5,
                   n_periods=20,
                   internal_ref=True,
                   ref_frequency=1000,
                   frames=300):
        camera.remote_device.node_map.DeviceOperationMode.value = 'LockInCam'
        camera.remote_device.node_map.LockInSensitivity.value = sensitivity
        camera.remote_device.node_map.LockInTargetTimeConstantNPeriods.value = n_periods
        camera.remote_device.node_map.LockInCoupling.value = 'AC'
        camera.remote_device.node_map.AcquisitionBurstFrameCount.value = frames
        # configure the RTIO outputs to the recording active signal and the reference demodulation frequency
        camera.remote_device.node_map.LineSelector.value = "RTIO2"
        camera.remote_device.node_map.LineSource.value = "LockInReference"
        camera.remote_device.node_map.LineSelector.value = "RTIO3"
        camera.remote_device.node_map.LineSource.value = "RecordingActive"

        if internal_ref:
            camera.remote_device.node_map.LockInReferenceSourceType.value = 'Internal'
            camera.remote_device.node_map.LockInTargetReferenceFrequency.value = ref_frequency
            camera.remote_device.node_map.LockInExpectedFrequencyDeviation.value = 1
        else:
            camera.remote_device.node_map.LockInReferenceSourceType.value = 'External'
            camera.remote_device.node_map.LockInReferenceFrequencyScaler.value = 'Off'
            camera.remote_device.node_map.LockInReferenceSourceSignal.value = 'FI3'
            camera.remote_device.node_map.LockInExpectedFrequencyDeviation.value = 1

    lia_config()  # initialize with the default settings

    camera.remote_device.node_map.SignalGeneratorFrequency.value = 1000
    camera.remote_device.node_map.SignalGeneratorAmplitude.value = 0.01
    camera.remote_device.node_map.SignalGeneratorOffset.value = 0.01
    camera.remote_device.node_map.SignalGeneratorMode.value = "On"
    camera.remote_device.node_map.LightControllerSelector.value = "LightController0"
    camera.remote_device.node_map.LightControllerSource.value = "SignalGenerator"

    camera.remote_device.node_map.Scan3dExtractionMethod.value = "rawIQ"
    return camera, h


def acquire(camera, auto_stop=True):
    camera.start()
    camera.remote_device.node_map.TriggerSoftware.execute()

    with camera.fetch(timeout=20) as buffer:
        n_frames = len(buffer.payload.components) // 2
        height = buffer.payload.components[0].height
        width = buffer.payload.components[0].width
        out_shape = (n_frames, height, width)
        # print(out_shape)

        def transform_raw(img):
            return (img % 2 ** 15) / 2 ** 2

        # harvesters returns data as 1D np arrays
        I_1d = np.array([
            transform_raw(img.data.astype(float))
            for img in buffer.payload.components[:n_frames]
        ])
        Q_1d = np.array([
            transform_raw(img.data.astype(float))
            for img in buffer.payload.components[n_frames:2 * n_frames]
        ])

        I = I_1d.reshape(out_shape)
        Q = Q_1d.reshape(out_shape)

    if auto_stop:
        camera.stop()
    return (I, Q, height, width, n_frames)




def get_amplitude(bg_mean_Q, bg_mean_I, I_data, Q_data, n_frames):
    # 创建空列表用于存储每一帧的强度信号
    frames_list = []

    for i in range(n_frames):
        frame_Q = Q_data[i]
        frame_I = I_data[i]

        frame_Q = frame_Q.astype(np.float32)
        frame_I = frame_I.astype(np.float32)

        # 计算强度信号
        frame = np.sqrt((frame_Q - bg_mean_Q) ** 2 + (frame_I - bg_mean_I) ** 2)

        # 可选：对信号进行进一步处理，如归一化、滤波等
        # 左旋90°
        frame = np.rot90(frame)
        # 将每一帧的强度信号存储到列表中
        frames_list.append(frame)

    # 将列表转换为 numpy 数组
    Amplitude = np.array(frames_list, dtype=np.float32)
    # 保存为 .npy 文件
    # np.save('Amplitude.npy', Amplitude)
    #
    # print("强度信号保存成功！")

    return Amplitude[1:]


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
            forehead_indices = [454, 323, 361]
            x0 = int(np.mean([landmarks.landmark[i].x for i in forehead_indices]) * iw)
            y0 = int(np.mean([landmarks.landmark[i].y for i in forehead_indices]) * ih)

            count += 1
            if abs(x0 - last_x0) > 10:
                last_x0, last_y0 = x0, y0
        x0, y0 = last_x0, last_y0
        last_x0, last_y0 = x0, y0

        a = 15
        b = 0
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



def load_noise():
    # check if the noise is got previously.
    filename_I = 'noise_I.npy'
    filename_Q = 'noise_Q.npy'
    if os.path.isfile(filename_I) and os.path.isfile(filename_Q):
        noise_I = np.load(filename_I)
        noise_Q = np.load(filename_Q)
        # print("noise_I and noise_Q are loaded. \n")
    else:
        print("Please load the noise at first...\n")

    # Calculate mean along axis 0
    bg_mean_Q = np.mean(noise_Q, axis=0)
    bg_mean_I = np.mean(noise_I, axis=0)

    return bg_mean_Q, bg_mean_I



import scipy.signal as signal
import pywt
def denoise(roi_mean):

    # print('top8_indices:', top8_indices)
    B1 = roi_mean[0, :]
    B2 = roi_mean[1, :]
    B3 = roi_mean[2, :]
    B4 = roi_mean[3, :]


    len_b1 = len(B1)
    end_ind = len_b1 - 10

    print('len:', len_b1)
    denoised_data = np.zeros([4, end_ind - 9])
    peak = [None] * 4
    peak_locs = [None] * 4


    fs = 51
    found = False

    winwithd_range = np.arange(0.01, 0.32, 0.1)
    max_butter_range = np.arange(3.1, 2.5, -0.1)
    mindis_range = np.arange(0.3, 0.9, 0.1)
    min_width_range = np.arange(0, 11, 2)


    for th in range(5, 31, 5):
        if found:
            break
        for winwithd in winwithd_range:
            if found:
                break
            for max_butter in max_butter_range:
                if found:
                    break
                for mindis in mindis_range:
                    if found:
                        break
                    for min_width in min_width_range:
                        if found:
                            break

                        len_b1 = len(B1)
                        end_ind = len_b1 - 10
                        A1 = B1[10:end_ind]
                        A2 = B2[10:end_ind]
                        A3 = B3[10:end_ind]
                        A4 = B4[10:end_ind]

                        regions = [A1, A2, A3, A4]
                        region_names = ['Region1', 'Region2', 'Region3', 'Region4']
                        all_valid_peak_locs = [[] for _ in range(4)]

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
                            # Peak detection
                            peaks, _ = signal.find_peaks(denoised_ppg, distance=int(round(mindis * fs)))
                            peak_widths = signal.peak_widths(denoised_ppg, peaks)[0]
                            peak_threshold = - 0.1 * np.max(denoised_ppg[peaks])
                            valid_peak_idx = (denoised_ppg[peaks] > peak_threshold) & (peak_widths >= min_width) & (
                                    peak_widths <= 100)
                            valid_peaks = denoised_ppg[peaks][valid_peak_idx]
                            valid_peak_locs = peaks[valid_peak_idx]


                            all_valid_peak_locs[i] = valid_peak_locs

                            denoised_data[i] = denoised_ppg
                            peak[i] = valid_peaks
                            peak_locs[i] = valid_peak_locs
                            best_variance = np.inf

                        for j1 in range(len(all_valid_peak_locs[0]) - 2):
                            for j2 in range(len(all_valid_peak_locs[1]) - 2):
                                for j3 in range(len(all_valid_peak_locs[2]) - 2):
                                    for j4 in range(len(all_valid_peak_locs[3]) - 2):
                                        group1 = [
                                            all_valid_peak_locs[0][j1],
                                            all_valid_peak_locs[1][j2],
                                            all_valid_peak_locs[2][j3],
                                            all_valid_peak_locs[3][j4]
                                        ]
                                        group2 = [
                                            all_valid_peak_locs[0][j1 + 1],
                                            all_valid_peak_locs[1][j2 + 1],
                                            all_valid_peak_locs[2][j3 + 1],
                                            all_valid_peak_locs[3][j4 + 1]
                                        ]
                                        group3 = [
                                            all_valid_peak_locs[0][j1 + 2],
                                            all_valid_peak_locs[1][j2 + 2],
                                            all_valid_peak_locs[2][j3 + 2],
                                            all_valid_peak_locs[3][j4 + 2]
                                        ]
                                        variance_group1 = np.var(group1)
                                        variance_group2 = np.var(group2)
                                        variance_group3 = np.var(group3)
                                        mean_group1 = np.mean(group1)
                                        mean_group2 = np.mean(group2)
                                        mean_group3 = np.mean(group3)
                                        gap_2 = mean_group3 - mean_group2
                                        gap_1 = mean_group2 - mean_group1
                                        total_variance = variance_group1 + variance_group2 + variance_group3

                                        if (variance_group1 < th and variance_group2 < th and
                                                variance_group3 < th and total_variance < best_variance and
                                                abs(gap_2 - gap_1) < 7):
                                            best_variance = total_variance
                                            best_params = [winwithd, max_butter, mindis, min_width]
                                            best_group1 = group1
                                            best_group2 = group2
                                            best_group3 = group3
                                            best_th = th
                                            found = True

    if found:
        print(best_group1)
        print(best_group2)
        print(best_group3)
        heartrate = 60 * fs / ((np.mean(best_group3) - np.mean(best_group1)) / 2)
        if heartrate > 145:
            heartrate = heartrate / 2
            print(heartrate)
        print('best_params:', best_params)
        print('best_th:', best_th)
    else:
        print('final_params:', [winwithd, max_butter, mindis, min_width])
        heartrate = 0
        params = np.nan
        print('No suitable configuration found with variance < 0.4')


    return heartrate, denoised_data, peak, peak_locs
