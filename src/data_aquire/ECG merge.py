import os
import pandas as pd

from util import data_path

# 设置文件夹路径
root_dir = data_path("ECG processed data/1124")
root_dir1 = data_path("ecg_raw")

all_data = []

if __name__ == "__main__":
    # 直接遍历根目录中的每个txt文件
    for file_name in os.listdir(root_dir):
        if file_name.endswith(".txt"):
            file_path = os.path.join(root_dir, file_name)
            print(f"\nReading file: {file_path}")
            print(f"File size: {os.path.getsize(file_path)} bytes")

            try:
                # 读取数据
                df = pd.read_csv(file_path, sep='\t', header=None, names=["time", "data"])
                print(f"File loaded successfully with shape: {df.shape}")
                print(df.head())

                # 解析时间
                df['full_time'] = pd.to_datetime(df['time'], format='%H:%M:%S.%f')
                all_data.append(df[['full_time', 'data']])
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")
                continue

    # 检查是否成功读取了任何数据
    if not all_data:
        print("No valid data found in the provided directories.")
    else:
        combined_df = pd.concat(all_data)
        sorted_df = combined_df.sort_values(by='full_time')

        # 格式化时间列，只保留时分秒，精确到秒后两位小数
        sorted_df['formatted_time'] = sorted_df['full_time'].dt.strftime('%H:%M:%S.%f').apply(lambda x: x[:-4])

        # 只保留格式化后的时间和数据列
        final_df = sorted_df[['formatted_time', 'data']]

        output_file = os.path.join(root_dir1, "ECG_1124.txt")
        final_df.to_csv(output_file, sep="\t", index=False, header=False)
        print(f"数据已成功整合并排序，保存为 {output_file}")
        print(f"Total rows in combined data: {len(final_df)}")

        # 打印输出文件的前几行，用于验证
        print("\n输出文件的前几行:")
        with open(output_file, 'r') as f:
            for i, line in enumerate(f):
                if i < 5:  # 只打印前5行
                    print(line.strip())
                else:
                    break