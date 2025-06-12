import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

data_path = r'C:\Users\bliu259-admin\Documents\uw-ppg-project\data\network\1124'

# 1. 加载数据 (17050, 300, 2)
data = np.load(os.path.join(data_path, 'rawdata', 'ECG_referencePPG.npy'))

# 2. 分离特征（PPG信号）和标签（ECG信号）
X = data[:, :, 1]  # PPG信号 (形状为17050×300)
y = data[:, :, 0]  # ECG信号 (形状为17050×300)
print(X.shape, y.shape)

# 3. 扩展 X 的维度以符合模型需求 (将其转换为形状17050×300×1)
X = X[:, :, np.newaxis]
print("After dimension expansion:", X.shape)

# 4. 按9:1比例划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# 5. 对特征（PPG信号）进行归一化
def normalize_features(X):
    """Normalize each sample independently for X."""
    X_normalized = np.zeros_like(X)
    for i in range(X.shape[0]):  # 遍历样本
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_normalized[i, :, 0] = scaler.fit_transform(X[i, :, 0].reshape(-1, 1)).flatten()
    return X_normalized

X_train_normalized = normalize_features(X_train)
X_test_normalized = normalize_features(X_test)

# 6. 对标签（ECG信号）进行归一化
def normalize_labels(y):
    """Normalize each sample independently for y."""
    y_normalized = np.zeros_like(y)
    for i in range(y.shape[0]):  # 遍历样本
        scaler = MinMaxScaler(feature_range=(-1, 1))
        y_normalized[i, :] = scaler.fit_transform(y[i, :].reshape(-1, 1)).flatten()
    return y_normalized

y_train_normalized = normalize_labels(y_train)
y_test_normalized = normalize_labels(y_test)

print(X_train_normalized.shape, y_train_normalized.shape, X_test_normalized.shape, y_test_normalized.shape)

# 7. 保存训练集和测试集
train_data_path = os.path.join(data_path, 'training_data')
test_data_path = os.path.join(data_path, 'test_data')
os.makedirs(train_data_path, exist_ok=True)

np.save(os.path.join(train_data_path, 'x_ECG_ref.npy'), X_train_normalized)
np.save(os.path.join(train_data_path, 'y_ECG_ref.npy'), y_train_normalized)
np.save(os.path.join(test_data_path, 'x_ECG_ref.npy'), X_test_normalized)
np.save(os.path.join(test_data_path, 'y_ECG_ref.npy'), y_test_normalized)

print("数据处理和保存完成！")

# 8. 加载训练集
X_train = np.load(os.path.join(train_data_path, 'x_ECG_ref.npy'))
y_train = np.load(os.path.join(train_data_path, 'y_ECG_ref.npy'))

# 9. 随机选择一些样本
num_samples = 5  # 随机选择5个样本进行绘制
random_indices = np.random.choice(X_train.shape[0], num_samples, replace=False)

# 10. 绘制样本
fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 3))

for i, idx in enumerate(random_indices):
    # 绘制PPG信号
    axes[i, 0].plot(X_train[idx, :, 0], label='PPG Signal')
    axes[i, 0].set_title(f'Sample {i+1} - PPG Signal')
    axes[i, 0].legend()

    # 绘制ECG信号
    axes[i, 1].plot(y_train[idx], label='ECG Signal', color='r')
    axes[i, 1].set_title(f'Sample {i+1} - ECG Signal')
    axes[i, 1].legend()

plt.tight_layout()
plt.show()
