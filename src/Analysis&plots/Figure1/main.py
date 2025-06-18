# 依次运行三个Python文件
import subprocess

data_path = 'src/Figure1/'

python_files = [
    data_path+'lockin.py',
    data_path+'regular.py',
    data_path+'RGB.py' 
]

for file in python_files:
    try:
        subprocess.run(['python', file], check=True)
        print(f'Successfully executed {file}')
    except subprocess.CalledProcessError as e:
        print(f'Error executing {file}: {e}')


