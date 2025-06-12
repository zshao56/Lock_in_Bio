import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from util import data_path
import pandas as pd

light = pd.read_csv(data_path('Figure1/light_5000K.csv'),header=0, names=['wavelength', 'intensity'],sep='\t')
wvl = light['wavelength']
intensity = light['intensity']
d50_intensity_normalized = intensity / np.max(intensity)

# 普朗克黑体辐射公式（单位：W·sr⁻¹·m⁻²·nm⁻¹）
def planck_spectrum(wavelength_nm, T):
    h = 6.62607015e-34  # 普朗克常数 (J·s)
    c = 299792458       # 光速 (m/s)
    k = 1.380649e-23    # 玻尔兹曼常数 (J/K)
    
    wavelength_m = wavelength_nm * 1e-9  # 纳米转米
    term = (h * c) / (wavelength_m * k * T)
    intensity = (2 * h * c**2) / (wavelength_m**5) * (1 / (np.exp(term) - 1))
    return intensity

# 参数设置
T = 5000  # 色温 (K)
wavelengths_nm = np.linspace(300, 2500, 1000)  # 波长范围 300-2500 nm

# 计算光谱强度（归一化到最大值为1）
intensity = planck_spectrum(wavelengths_nm, T)
intensity_normalized = intensity / np.max(intensity)

# 绘制光谱图
plt.figure(figsize=(10, 6))
plt.plot(wavelengths_nm, intensity_normalized, color='black', linewidth=1.5, label=f'5000K sunlight')
plt.plot(wvl, d50_intensity_normalized, color='blue', linewidth=1.5, label=f'CIE D50 (5000K)')
# 标出可见光范围 (380-780nm)
plt.axvspan(380, 780, color='gray', alpha=0.2, label='Visible Light')
plt.axvline(5500, color='red', linestyle='--', alpha=0.5)  # 5500K理论峰值波长（维恩位移定律）

# 颜色渐变背景（模拟光谱颜色）
cmap = LinearSegmentedColormap.from_list('visible_spectrum', 
    ['#9400d3', '#4b0082', '#0000ff', '#00ff00', '#ffff00', '#ff7f00', '#ff0000'])
plt.imshow(np.array([np.linspace(0, 1, 100)]), 
           extent=[380, 780, 0, 1.1], aspect='auto', cmap=cmap, alpha=0.15)

# 图表标注
plt.title(f"Blackbody Radiation Spectrum at T = {T}K", fontsize=14)
plt.xlabel("Wavelength (nm)", fontsize=12)
plt.ylabel("Normalized Intensity", fontsize=12)
plt.xlim(380, 780)
plt.ylim(0, 1.05)
plt.grid(alpha=0.3)
plt.legend()
plt.show()