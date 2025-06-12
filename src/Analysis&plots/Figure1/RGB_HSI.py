import numpy as np
from PIL import Image
import cairosvg
import io
import matplotlib.pyplot as plt
from util import data_path

def rgb_to_hsi(svg_path):
    # 将SVG转换为PNG格式的字节流
    # 将DPI提高到3000以获得更高分辨率
    png_data = cairosvg.svg2png(url=svg_path, dpi=50000, scale=8.0, background_color=None)  # 增加DPI和缩放比例，设置白色背景
    
    # 从字节流创建PIL图像
    img = Image.open(io.BytesIO(png_data))
    
    # 转换为RGBA格式
    img = img.convert('RGBA')
    
    # 创建一个新的图像，保留透明度
    new_img = Image.new('RGBA', img.size, (0, 0, 0, 0))
    new_img.paste(img, (0, 0), img)
    
    # 显示转换后的PNG图像
    plt.figure(figsize=(8, 8))
    plt.imshow(new_img)
    plt.axis('off')
    plt.savefig(data_path('Figure1/uwlogo_RGB.png'), bbox_inches='tight', pad_inches=0, transparent=True)
    plt.show()
    
    # 更新img为新的图像
    img = new_img
    
    # 放大图像尺寸到原来的2倍
    new_size = tuple(2*x for x in img.size)
    img = img.resize(new_size, Image.LANCZOS)
    
    # 转换为numpy数组
    rgba = np.array(img)
    
    # 创建输出数组
    x, y = rgba.shape[:2]
    hsi = np.zeros((x, y, 400))
    
    # 生成波长数组 (381-780nm)
    wavelengths = np.linspace(381, 780, 400)
    
    # 对每个像素进行转换
    for i in range(x):
        for j in range(y):
            if rgba[i,j,3] > 0:  # 检查alpha通道
                r, g, b = rgba[i,j,:3]
                
                # 使用简单的高斯模型模拟光谱响应
                # 为RGB每个通道创建高斯响应
                r_response = np.exp(-(wavelengths - 650)**2 / (2 * 50**2)) * r/255
                g_response = np.exp(-(wavelengths - 550)**2 / (2 * 50**2)) * g/255
                b_response = np.exp(-(wavelengths - 450)**2 / (2 * 50**2)) * b/255
                
                # 组合响应
                hsi[i,j,:] = r_response + g_response + b_response
            
            # 透明像素保持为0
    
    return hsi

svg_path = data_path('Figure1/uwlogo.svg')
npy_path = data_path('Figure1/uwlogo.npy')
hsi = rgb_to_hsi(svg_path)
print(hsi.shape)

# 保存数据
np.save(npy_path, hsi)
