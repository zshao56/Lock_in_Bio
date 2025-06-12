import os
import pandas as pd

def process_region(region_name, input_root, output_root):
    """
    处理指定区域的txt文件，并整合保存为一个文件。
    
    Args:
        region_name (str): 区域名称（例如 'right_cheek', 'left_cheek', 'forehead'）。
        input_root (str): 输入文件夹根路径。
        output_root (str): 输出文件夹根路径。
    """
    # 设置输入和输出路径
    region_dir = os.path.join(input_root, region_name)
    output_file = os.path.join(output_root, f"{region_name}.txt")
    print(f"\nProcessing region: {region_name}")
    print(f"Input directory: {region_dir}")
    print(f"Output file: {output_file}")

    all_data = []

    # 遍历区域目录中的txt文件
    for file_name in os.listdir(region_dir):
        if file_name.endswith(".txt"):
            file_path = os.path.join(region_dir, file_name)
            print(f"\nReading file: {file_path}")
            print(f"File size: {os.path.getsize(file_path)} bytes")

            try:
                # 读取数据，跳过第一行（标题行）
                df = pd.read_csv(file_path, sep='\t', skiprows=1, header=None,
                                 names=["Time", "ROI1", "ROI2", "ROI3", "ROI4"])
                print(f"File loaded successfully with shape: {df.shape}")
                print(df.head())

                # 解析时间列
                df['full_time'] = pd.to_datetime(df['Time'], format='%H:%M:%S.%f')
                all_data.append(df)
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")
                continue

    # 检查是否读取了任何数据
    if not all_data:
        print(f"No valid data found in the {region_name} directory.")
    else:
        combined_df = pd.concat(all_data)
        sorted_df = combined_df.sort_values(by='full_time')

        # 只保留原始时间格式和ROI数据
        final_df = sorted_df[['Time', 'ROI1', 'ROI2', 'ROI3', 'ROI4']]

        # 保存到输出文件
        final_df.to_csv(output_file, sep="\t", index=False, header=False)
        print(f"数据已成功整合并排序，保存为 {output_file}")
        print(f"Total rows in combined data for {region_name}: {len(final_df)}")

        # 打印输出文件的前几行，用于验证
        print("\n输出文件的前几行:")
        with open(output_file, 'r') as f:
            for i, line in enumerate(f):
                if i < 5:  # 只打印前5行
                    print(line.strip())
                else:
                    break

# 定义输入和输出路径
input_root_dir = r"C:\Users\bliu259-admin\Documents\uw-ppg-project\data\ppg_denoised_lockin\1124"
output_root_dir = r"C:\Users\bliu259-admin\Documents\uw-ppg-project\data\ppg_denoised_lockin\1124\merge"

# 确保输出路径存在
if not os.path.exists(output_root_dir):
    os.makedirs(output_root_dir)

# 定义需要处理的区域
regions = ['right_cheek', 'left_cheek', 'forehead']

# 遍历每个区域，进行处理
for region in regions:
    process_region(region, input_root_dir, output_root_dir)
