import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from util import data_path

# 定义文件路径和颜色映射
paths = ['Figure3/frames/video_red.npy', 'Figure3/frames/video_infrared.npy']
purple = ["black", "#82277E", "white"]
red = ["black", "#DF1715", "white"]
cmap = [
    LinearSegmentedColormap.from_list("black_red_white", red),
    LinearSegmentedColormap.from_list("black_purple_white", purple)
]

# 遍历每个路径
for i in range(len(paths)):
    # 加载 NPY 文件
    brightness_data = np.load(data_path(paths[i]))
    
    # 提取指定帧
    frames_to_plot = [0, 1, 2, 5, 6, 7, -3, -2, -1]  # 第一帧、第5帧、第10帧、第15帧和最后一帧
    frames = [brightness_data[i] for i in frames_to_plot]
    
    # 设置 seaborn 风格
    sns.set(style="white")
    
    # 绘制热力图
    for idx, frame in zip(frames_to_plot, frames):
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            frame,
            cmap=cmap[i],
            cbar=False,
            square=True,
            vmin=30,  # 亮度最小值
            vmax=150  # 亮度最大值
        )
        plt.axis("off")  # 关闭坐标轴
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)  # 去除边距
        
        # 保存图像
        output_path = data_path(f"Figure2/frames/heatmap_{i}_{idx if idx != -1 else brightness_data.shape[0] - 1}.png")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.show()  # 关闭图像以释放内存