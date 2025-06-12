import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from util import data_path
import pandas as pd

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
wavelengths_nm = np.linspace(381, 780, 400)  # 波长范围 300-2500 nm

# 计算光谱强度（归一化到最大值为1）
intensity = planck_spectrum(wavelengths_nm, T)
intensity_normalized = intensity / np.max(intensity)

wvl = wavelengths_nm.tolist()
intensity = intensity_normalized.tolist()

output_path = data_path('Figure1/solar_5000K.csv')
np.savetxt(output_path, np.column_stack((wvl,intensity)), header='wavelength\tintensity', delimiter='\t')