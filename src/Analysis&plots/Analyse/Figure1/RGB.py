import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from util import data_path
# 读取光谱文件
def read_spectrum(file_path):
    data = np.loadtxt(file_path, delimiter='\t', skiprows=1)
    return data[:, 0], data[:, 1]

# 光谱转XYZ
def spectrum_to_xyz(wavelengths, intensity):
    # 加载并插值CIE数据
    cie_data = np.array([
    [360, 0.0001299, 0.000003917, 0.0006061],
    [365, 0.0002321, 0.000006965, 0.001086],
    [370, 0.0004149, 0.00001239, 0.001946],
    [375, 0.0007416, 0.00002202, 0.003486],
    [380, 0.001368, 0.000039, 0.006450],
    [385, 0.002236, 0.000064, 0.010550],
    [390, 0.004243, 0.000120, 0.020050],
    [395, 0.007650, 0.000217, 0.036210],
    [400, 0.014310, 0.000396, 0.067850],
    [405, 0.023190, 0.000640, 0.110200],
    [410, 0.043510, 0.001210, 0.207400],
    [415, 0.077630, 0.002180, 0.371300],
    [420, 0.134380, 0.004000, 0.645600],
    [425, 0.214770, 0.007300, 1.039050],
    [430, 0.283900, 0.011600, 1.385600],
    [435, 0.328500, 0.016840, 1.622960],
    [440, 0.348280, 0.023000, 1.747060],
    [445, 0.348060, 0.029800, 1.782600],
    [450, 0.336200, 0.038000, 1.772110],
    [455, 0.318700, 0.048000, 1.744100],
    [460, 0.290800, 0.060000, 1.669200],
    [465, 0.251100, 0.073900, 1.528100],
    [470, 0.195360, 0.090980, 1.287640],
    [475, 0.142100, 0.112600, 1.041900],
    [480, 0.095640, 0.139020, 0.812950],
    [485, 0.057950, 0.169300, 0.616200],
    [490, 0.032010, 0.208020, 0.465180],
    [495, 0.014700, 0.258600, 0.353300],
    [500, 0.004900, 0.323000, 0.272000],
    [505, 0.002400, 0.407300, 0.212300],
    [510, 0.009300, 0.503000, 0.158200],
    [515, 0.029100, 0.608200, 0.111700],
    [520, 0.063270, 0.710000, 0.078250],
    [525, 0.109600, 0.793200, 0.057250],
    [530, 0.165500, 0.862000, 0.042160],
    [535, 0.225750, 0.914850, 0.029840],
    [540, 0.290400, 0.954000, 0.020300],
    [545, 0.359700, 0.980300, 0.013400],
    [550, 0.433450, 0.994950, 0.008750],
    [555, 0.512050, 1.000000, 0.005750],
    [560, 0.594500, 0.995000, 0.003900],
    [565, 0.678400, 0.978600, 0.002750],
    [570, 0.762100, 0.952000, 0.002100],
    [575, 0.842500, 0.915400, 0.001800],
    [580, 0.916300, 0.870000, 0.001650],
    [585, 0.978600, 0.816300, 0.001400],
    [590, 1.026300, 0.757000, 0.001100],
    [595, 1.056700, 0.694900, 0.001000],
    [600, 1.062200, 0.631000, 0.000800],
    [605, 1.045600, 0.566800, 0.000600],
    [610, 1.002600, 0.503000, 0.000340],
    [615, 0.938400, 0.441200, 0.000240],
    [620, 0.854450, 0.381000, 0.000190],
    [625, 0.751400, 0.321000, 0.000100],
    [630, 0.642400, 0.265000, 0.000050],
    [635, 0.541900, 0.217000, 0.000030],
    [640, 0.447900, 0.175000, 0.000020],
    [645, 0.360800, 0.138200, 0.000010],
    [650, 0.283500, 0.107000, 0.000000],
    [655, 0.218700, 0.081600, 0.000000],
    [660, 0.164900, 0.061000, 0.000000],
    [665, 0.121200, 0.044580, 0.000000],
    [670, 0.087400, 0.032000, 0.000000],
    [675, 0.063600, 0.023200, 0.000000],
    [680, 0.046770, 0.017000, 0.000000],
    [685, 0.032900, 0.011920, 0.000000],
    [690, 0.022700, 0.008210, 0.000000],
    [695, 0.015840, 0.005723, 0.000000],
    [700, 0.011359, 0.004102, 0.000000],
    [705, 0.008111, 0.002929, 0.000000],
    [710, 0.005790, 0.002091, 0.000000],
    [715, 0.004109, 0.001484, 0.000000],
    [720, 0.002899, 0.001047, 0.000000],
    [725, 0.002049, 0.000740, 0.000000],
    [730, 0.001440, 0.000520, 0.000000],
    [735, 0.001000, 0.000361, 0.000000],
    [740, 0.000690, 0.000249, 0.000000],
    [745, 0.000476, 0.000172, 0.000000],
    [750, 0.000332, 0.000120, 0.000000],
    [755, 0.000235, 0.000085, 0.000000],
    [760, 0.000166, 0.000060, 0.000000],
    [765, 0.000117, 0.000042, 0.000000],
    [770, 0.000083, 0.000030, 0.000000],
    [775, 0.000059, 0.000021, 0.000000],
    [780, 0.000042, 0.000015, 0.000000]
    ])
    cie_wl = cie_data[:, 0]
    x_interp = interp1d(cie_wl, cie_data[:, 1], bounds_error=False, fill_value=0)
    y_interp = interp1d(cie_wl, cie_data[:, 2], bounds_error=False, fill_value=0)
    z_interp = interp1d(cie_wl, cie_data[:, 3], bounds_error=False, fill_value=0)
    
    x = x_interp(wavelengths)
    y = y_interp(wavelengths)
    z = z_interp(wavelengths)
    delta = wavelengths[1] - wavelengths[0]
    X = np.trapz(intensity * x, dx=delta)
    Y = np.trapz(intensity * y, dx=delta)
    Z = np.trapz(intensity * z, dx=delta)
    return X, Y, Z

# XYZ转RGB
def xyz_to_srgb(X, Y, Z):
    matrix = np.array([[3.2406, -1.5372, -0.4986],
                       [-0.9689, 1.8758, 0.0415],
                       [0.0557, -0.2040, 1.0570]])
    XYZ = np.array([X, Y, Z]) / Y
    RGB = np.dot(matrix, XYZ)
    RGB = np.clip(RGB, 0, 1)
    RGB = np.where(RGB <= 0.0031308, 12.92 * RGB, 1.055 * RGB**(1/2.4) - 0.055)
    return RGB

# 主程序
if __name__ == "__main__":
    # 读取数据
    input_path_1 = data_path('Figure1/solar_5000K.csv')
    input_path_2 = data_path('Figure1/light_5000K.csv')
    wavelengths1, intensity1 = read_spectrum(input_path_1)
    wavelengths2, intensity2 = read_spectrum(input_path_2)
    
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[3, 1, 2], hspace=0.5)

    # ================= 光谱图 =================
    ax_spectrum = fig.add_subplot(gs[0, :])
    ax_spectrum.plot(wavelengths1, intensity1, 
                    label='Solar 5000K', color='orange', lw=2)
    ax_spectrum.plot(wavelengths2, intensity2, 
                    label='Light 5000K', color='deepskyblue', lw=2, linestyle='--')
    ax_spectrum.set_xlabel('Wavelength (nm)', fontsize=12)
    ax_spectrum.set_ylabel('Normalized Intensity', fontsize=12)
    ax_spectrum.set_title('Spectral Comparison', fontsize=14, pad=15)
    ax_spectrum.legend(loc='upper right')
    ax_spectrum.grid(alpha=0.3)

    # ================= 计算RGB颜色 =================
    X1, Y1, Z1 = spectrum_to_xyz(wavelengths1, intensity1)
    X2, Y2, Z2 = spectrum_to_xyz(wavelengths2, intensity2)
    rgb1 = np.clip(xyz_to_srgb(X1, Y1, Z1), 0, 1)
    rgb2 = np.clip(xyz_to_srgb(X2, Y2, Z2), 0, 1)

    # ================= RGB颜色块 =================
    # Solar 5000K颜色块
    ax_color1 = fig.add_subplot(gs[1, 0])
    ax_color1.imshow([[rgb1]], aspect='auto')
    ax_color1.axis('off')
    text_color1 = 'white' if np.mean(rgb1) < 0.4 else 'black'
    ax_color1.text(0.5, 0.5, 
                  f"Solar RGB:\n({rgb1[0]:.3f}, {rgb1[1]:.3f}, {rgb1[2]:.3f})",
                  ha='center', va='center', 
                  color=text_color1, fontsize=11,
                  transform=ax_color1.transAxes)

    # Light 5000K颜色块
    ax_color2 = fig.add_subplot(gs[1, 1])
    ax_color2.imshow([[rgb2]], aspect='auto')
    ax_color2.axis('off')
    text_color2 = 'white' if np.mean(rgb2) < 0.4 else 'black'
    ax_color2.text(0.5, 0.5, 
                  f"Light RGB:\n({rgb2[0]:.3f}, {rgb2[1]:.3f}, {rgb2[2]:.3f})",
                  ha='center', va='center', 
                  color=text_color2, fontsize=11,
                  transform=ax_color2.transAxes)

    # ================= RGB通道柱状图 =================
    ax_bar = fig.add_subplot(gs[2, :])
    
    # 柱状图参数设置
    bar_width = 0.35
    x_index = np.arange(3)  # R/G/B三个通道
    colors = ['#FF4444', '#44FF44', '#4444FF']  # 红/绿/蓝
    
    # 绘制双组柱状图
    bars1 = ax_bar.bar(x_index - bar_width/2, rgb1, bar_width,
                      color=colors, edgecolor='k', label='Solar 5000K')
    bars2 = ax_bar.bar(x_index + bar_width/2, rgb2, bar_width,
                      color=colors, edgecolor='k', alpha=0.6, label='Light 5000K')

    # 添加数值标签
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    add_value_labels(bars1)
    add_value_labels(bars2)

    # 图表装饰
    ax_bar.set_title('RGB Channel Comparison', fontsize=12, pad=10)
    ax_bar.set_xticks(x_index)
    ax_bar.set_xticklabels(['Red', 'Green', 'Blue'])
    ax_bar.set_ylabel('Normalized Value')
    ax_bar.set_ylim(0, 1.2)
    ax_bar.legend(loc='upper right')
    ax_bar.grid(axis='y', alpha=0.3)

    # ================= 全局调整 =================
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # 留出顶部标题空间
    fig.suptitle('Spectral Analysis with RGB Visualization', 
                fontsize=16, y=0.97)
    plt.savefig(data_path('Figure1/5000K_RGB.svg'))
    plt.show()