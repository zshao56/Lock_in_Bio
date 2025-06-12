import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from util import data_path

# Load the npy file
path = data_path('Figure1/uwlogo.npy')
light_T = ['3000K', '4000K', '5000K', '6000K', '7000K', '8000K', '9000K', '10000K']


for i in range(len(light_T)):
    # Load the light source data
    light_source_path = data_path('Figure1/lights/'+light_T[i]+'.csv')
    print(light_source_path)
    data = np.load(path)
    # Load the light source data
    light_source = pd.read_csv(light_source_path, header=None, sep='\t', skiprows=1, names=['wavelength', 'intensity'])

    # Calculate the reflected intensity for each pixel
    reflected_intensity = data * light_source.iloc[:, 1].values[np.newaxis, np.newaxis, :]

    wavelength_ranges = [(620, 680), (500, 570), (450, 500)]

    wavelengths = np.linspace(381, 780, reflected_intensity.shape[2])

    starts = []
    ends = []
    for start, end in wavelength_ranges:
        start_idx = np.argmin(np.abs(wavelengths - start))
        end_idx = np.argmin(np.abs(wavelengths - end))
        starts.append(start_idx)
        ends.append(end_idx)

    # Compute RGB by averaging reflected intensity over each wavelength range
    height, width, _ = reflected_intensity.shape
    rgb = np.zeros((height, width, 3))  # Array for R, G, B channels
    for j, (start_idx, end_idx) in enumerate(zip(starts, ends)):
        # Average the intensity over the wavelength range for each channel
        rgb[:, :, j] = np.mean(reflected_intensity[:, :, start_idx:end_idx + 1], axis=2)

    # Normalize RGB values to [0, 1] range
    rgb = np.clip(rgb, 0, None)  # Ensure no negative values
    rgb = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb))

    # Plot the RGB image
    # 创建掩码，将RGB值全为0的像素设为透明
    mask = np.any(rgb != 0, axis=2)
    # 创建RGBA图像，将mask作为alpha通道
    rgba = np.zeros((rgb.shape[0], rgb.shape[1], 4))
    rgba[..., :3] = rgb
    rgba[..., 3] = mask

    plt.imshow(rgba)
    plt.axis('off')  # 隐藏坐标轴
    output_dir = data_path('Figure1/RGB')

    # 打印当前处理的光源温度和对应的输出文件路径
    plt.savefig(os.path.join(output_dir, light_T[i]+'_RGB.svg'), bbox_inches='tight', pad_inches=0)
    plt.close()