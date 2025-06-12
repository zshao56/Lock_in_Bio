import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util import data_path



def extract_video_data(video_path):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    # 获取视频的帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 存储数据的列表
    times = []
    gray_intensities = []
    r_intensities = []
    g_intensities = []
    b_intensities = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 计算当前帧的时间（秒）
        time = frame_count / fps
        
        # 转换为灰度图像并计算平均亮度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_intensity = np.mean(gray)
        
        # 分离BGR通道并计算各通道平均亮度
        b, g, r = cv2.split(frame)
        r_intensity = np.mean(r)
        g_intensity = np.mean(g)
        b_intensity = np.mean(b)
        
        # 存储数据
        times.append(time)
        gray_intensities.append(gray_intensity)
        r_intensities.append(r_intensity)
        g_intensities.append(g_intensity)
        b_intensities.append(b_intensity)
        
        frame_count += 1
        # 获取视频总帧数

        
        # 打印进度条
        if frame_count % 30 == 0:  # 每30帧更新一次进度条
            progress = (frame_count / total_frames) * 100
            print(f'\r处理进度: {progress:.1f}%', end='')
    # 释放视频对象
    cap.release()
    
    # 创建DataFrame并保存为CSV
    df = pd.DataFrame({
        'Time(s)': times,
        'Gray_Intensity': gray_intensities,
        'R_Intensity': r_intensities,
        'G_Intensity': g_intensities,
        'B_Intensity': b_intensities
    })
    
    return df

if __name__ == "__main__":
    # 替换为你的视频文件路径
    video_path = r"C:\Users\85406\Desktop\demo_1.mp4"
    
    # 提取数据
    df = extract_video_data(video_path)


    # 保存为CSV文件
    output_path = r"C:\Users\85406\Desktop\demo_1.csv"
    df.to_csv(output_path, index=False)
    print(f"数据已保存到 {output_path}")
    plt.figure(figsize=(10, 6))
    # 绘制灰度图像的变化
    plt.plot(df['Time(s)'], df['Gray_Intensity'], label='Gray Intensity')
    # 绘制R、G、B通道的变化
    plt.plot(df['Time(s)'], df['R_Intensity'], label='R Intensity')
    plt.plot(df['Time(s)'], df['G_Intensity'], label='G Intensity')
    plt.plot(df['Time(s)'], df['B_Intensity'], label='B Intensity')
    plt.xlabel('Time (s)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid(True)
    plt.show()