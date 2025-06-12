import os
import numpy as np
import ipywidgets as widgets
from harvesters.core import Harvester
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
import seaborn as sns
## turn on the camera
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

    def lia_config(sensitivity=1,
                   n_periods=10,
                   internal_ref=True,
                   ref_frequency=500,
                   frames=300):
        camera.remote_device.node_map.DeviceOperationMode.value = 'LockInCam'
        camera.remote_device.node_map.LockInSensitivity.value = sensitivity
        camera.remote_device.node_map.LockInTargetTimeConstantNPeriods.value = n_periods
        camera.remote_device.node_map.LockInCoupling.value = 'DC'
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


    camera.remote_device.node_map.Scan3dExtractionMethod.value = "rawIQ"
    return camera, h

## acquire I&Q from the camera (processing the lock-in data)
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

##load the background data to normalize the baseline
def load_noise():
    # check if the noise is got previously.
    filename_I = r'C:\Users\bliu259-admin\Documents\uw-ppg-project\data\noise data/2_I.npy'
    filename_Q = r'C:\Users\bliu259-admin\Documents\uw-ppg-project\data\noise data/2_Q.npy'
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

## get the amplitude from the I&Q
def get_amplitude(bg_mean_Q, bg_mean_I, I_data, Q_data, n_frames):
    frames_list = []
    for i in range(n_frames):
        frame_Q = Q_data[i]
        frame_I = I_data[i]
        frame_Q = frame_Q.astype(np.float32)
        frame_I = frame_I.astype(np.float32)
        frame = np.sqrt((frame_Q - bg_mean_Q) ** 2 + (frame_I - bg_mean_I) ** 2)
        frame = np.rot90(frame)
        frames_list.append(frame)
    Amplitude = np.array(frames_list, dtype=np.float32)
    return Amplitude[1:]

## main function
def main():
    global bg_mean_Q, bg_mean_I, roi_mean, heart_rate, Amplitude
    bg_mean_Q, bg_mean_I = load_noise()
    cam, h = disc_cam()
    print("Prepare the pre-data...\n")
    for n in range(18):
        now = datetime.now()
        formatted_time = now.strftime("%H%M%S") + f"{now.microsecond // 1000:02d}"  # 添加毫秒部分，确保为两位数
        I, Q, height, width, n_frames = acquire(cam, auto_stop=True)
        Amplitude = get_amplitude(bg_mean_Q, bg_mean_I, I, Q, n_frames)
        # coordinate = [208, 255]
        # frame_values = np.zeros(Amplitude.shape[0])
        # for i in range(Amplitude.shape[0]):
        #     frame_values[i] = np.mean(Amplitude[i, coordinate[0]+10, coordinate[1]+10])
        # fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        # first_frame = Amplitude[10]
        # im = axs[0].imshow(first_frame, cmap='viridis', aspect='auto')
        # axs[0].set_title('Amplitude Heatmap (First Frame)')
        # axs[0].set_xlabel('X-axis')
        # axs[0].set_ylabel('Y-axis')
        # fig.colorbar(im, ax=axs[0])
        # axs[1].plot(range(Amplitude.shape[0]), frame_values,'g')
        # axs[1].set_title(f'Value at Coordinate {coordinate}')
        # axs[1].set_xlabel('Frame Number')
        # axs[1].set_ylabel('IQ Value')
        # axs[1].set_ylim([0,100])
        # axs[1].grid()
        # plt.tight_layout()
        # plt.show()
        # jpg_path = os.path.join('./PPG raw data/1112', formatted_time + '.jpg')
        # plt.savefig(jpg_path)


        file_path = os.path.join(r'C:\Users\bliu259-admin\Documents\uw-ppg-project\data\ppg_video_lockin\1124', formatted_time + '_Amplitude.npy')
        np.save(file_path, Amplitude)
        print('now time is:', formatted_time, 'now is:', n + 1)

if __name__ == "__main__":
    main()