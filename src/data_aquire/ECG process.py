import os
import numpy as np
import cv2
from pdf2image import convert_from_path
from PIL import Image
from scipy.interpolate import interp1d
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
# Function to rename PDF files

def rename_pdf_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            try:
                # 提取文件名中最后一部分
                base_name = filename.split('-')[-1]
                old_number = base_name.replace('.pdf', '')[-6:]  # 提取原6位数字

                # 将前两位数字减1
                first_two = int(old_number[:2]) - 1  # 取前两位，转换为整数后减1
                remaining = old_number[2:]  # 剩余部分保持不变
                new_number = f"{first_two:02d}{remaining}"  # 组合成新号码，确保前两位有两位数

                # 构建新文件名
                new_filename = new_number + ".pdf"

                # 检查文件是否已存在，避免覆盖
                old_file = os.path.join(directory, filename)
                new_file = os.path.join(directory, new_filename)
                counter = 1
                while os.path.exists(new_file):
                    # 如果文件名冲突，添加计数后缀
                    new_number_with_suffix = f"{new_number}_{counter}"
                    new_file = os.path.join(directory, new_number_with_suffix + ".pdf")
                    counter += 1

                # 执行重命名
                os.rename(old_file, new_file)
                print(f'Renamed: {filename} -> {new_filename}')
            except Exception as e:
                print(f"Error renaming {filename}: {e}")

# Convert PDF to images and crop specific regions
def pdf_to_image(pdf_file, dpi, crop_box=None):
    images = convert_from_path(pdf_file, dpi=dpi)
    cropped_images = []
    for img in images:
        cropped_images.append(img.crop(crop_box) if crop_box else img)
    return cropped_images
# Split image vertically into multiple parts
def split_image_vertically(image, num_splits=4):
    width, height = image.size
    split_height = height // num_splits
    split_images = [image.crop((0, i * split_height,
                                width, (i + 1) * split_height)) for i in range(num_splits)]
    return split_images
# Remove red color from image
def remove_red_color(image):
    image = np.array(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2
    image[red_mask > 0] = [255, 255, 255]
    return image
# Extract curve and save coordinates
def extract_curve_and_save_coords(image_path, output_txt_path, offset=0):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_height, img_width = image.shape[:2]
    img_no_red = remove_red_color(image)

    gray = cv2.cvtColor(img_no_red, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 3)
    kernel = np.ones((7, 7), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    edges = cv2.Canny(cleaned, 20, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    sorted_contour = sorted(largest_contour, key=lambda point: point[0][0])

    with open(output_txt_path, 'w') as f:
        for point in sorted_contour:
            x, y = point[0]
            y_corrected = img_height - y  # 反转 Y 坐标
            x_offset = x + offset  # 添加偏移
            f.write(f"{x_offset}\t{y_corrected}\n")

# Interpolate coordinates and concatenate four parts
def interpolate_and_concatenate(txt_files, output_txt_file,
                                start_time, x_scale=9450, x_range=8, interval=0.01):
    all_x_coords, all_y_coords = [], []
    for idx, txt_file in enumerate(txt_files):
        x_offset = idx * x_scale
        x_coords, y_coords = [], []
        with open(txt_file, 'r') as file:
            for line in file:
                x, y = map(float, line.split())
                x_coords.append(x + x_offset)
                y_coords.append(y)
        all_x_coords.extend(x_coords)
        all_y_coords.extend(y_coords)
    all_x_coords_normalized = (np.array(all_x_coords) / x_scale) * x_range
    mask = ~np.isnan(all_x_coords_normalized) & ~np.isnan(all_y_coords)
    all_x_coords_normalized = np.array(all_x_coords_normalized)[mask]
    all_y_coords = np.array(all_y_coords)[mask]
    interp_func = interp1d(all_x_coords_normalized, all_y_coords,
                           kind='linear', fill_value="extrapolate")
    x_new = np.arange(min(all_x_coords_normalized),
                      max(all_x_coords_normalized), interval)
    y_new = interp_func(x_new)
    adjusted_x_values = [start_time + timedelta(seconds=x - 0.5) for x in x_new]

    with open(output_txt_file, 'w') as file:
        for x, y in zip(adjusted_x_values, y_new):
            if x >= start_time:
                file.write(f"{x.strftime('%H:%M:%S')}."
                           f"{int(x.microsecond / 10000):02d}\t{y:.2f}\n")
# Extract time from filename
def extract_time_from_filename(pdf_filename):
    base_name = os.path.splitext(pdf_filename)[0]
    start_time_str = base_name[:6]
    start_time = datetime.strptime(start_time_str, '%H%M%S')
    return start_time
# Simple main function to demonstrate usage



def main():
    # Specify the folder where PDF files are stored
    pdf_folder = r'C:\Users\bliu259-admin\Documents\uw-ppg-project\data\ecg_pdf\1124'
    crop_box = (380, 2340, 9830, 8003)  # Adjust crop box as needed
    rename_pdf_files(pdf_folder)

    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            base_name = os.path.splitext(pdf_file)[0]
            start_time = extract_time_from_filename(pdf_file)
            cropped_images = pdf_to_image(pdf_path, dpi=1200, crop_box=crop_box)
            txt_files = []

            for i, cropped_image in enumerate(cropped_images):
                split_images = split_image_vertically(cropped_image)
                generated_images = []
                for j, split_img in enumerate(split_images):
                    image_path = f'{base_name}_page_{i}_part_{j}.png'
                    output_txt = f'{base_name}_coords_page_{i}_part_{j}.txt'

                    # 将分割图像保存为文件
                    split_img.save(image_path)
                    generated_images.append(image_path)
                    if j == 0:
                        # 裁剪前 1/16 部分
                        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                        img_height, img_width = image.shape[:2]
                        remove_width = int(img_width * 1 / 16)

                        # 裁剪图像
                        split_img_cropped = image[:, remove_width:]
                        cv2.imwrite(image_path, split_img_cropped)  # 保存裁剪后的图像
                        extract_curve_and_save_coords(image_path, output_txt, remove_width)  # 提取坐标
                    else:
                        extract_curve_and_save_coords(image_path, output_txt)  # 对其余部分提取坐标

                    txt_files.append(output_txt)
            # Interpolate and concatenate results
            output_txt_file = f'C:\\Users\\bliu259-admin\\Documents\\uw-ppg-project\\data\\ECG processed data\\1124\\{base_name}.txt'
            interpolate_and_concatenate(txt_files, output_txt_file, start_time)
            # Optionally, clean up intermediate files
            for txt_file in txt_files:
                os.remove(txt_file)
            for image_path in generated_images:
                os.remove(image_path)

if __name__ == '__main__':
    main()
