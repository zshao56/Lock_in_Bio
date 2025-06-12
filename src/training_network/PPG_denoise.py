import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from util import data_path
import os
datapath = data_path('train data/1118_averages')

# 定义自定义损失函数 WeightedMSELoss
class WeightedMSELoss(nn.Module):
    def __init__(self, threshold=0.1, weight=10.0):
        super(WeightedMSELoss, self).__init__()
        self.threshold = threshold
        self.weight = weight

    def forward(self, outputs, targets):
        # 计算加权的 MSE 损失，确保 outputs 和 targets 都是 (batch_size, 300)
        mse_loss = (outputs - targets) ** 2

        # 计算差异
        diff = torch.abs(outputs - targets)

        # 创建权重张量
        weights = torch.ones_like(diff)
        weights[diff > self.threshold] = self.weight

        # 计算加权 MSE 损失
        weighted_mse_loss = weights * mse_loss
        return torch.mean(weighted_mse_loss)



# 定义Dataset类
class SequenceDataset(Dataset):
    def __init__(self, data, targets):
        # 输入数据为 (15345, 300, 4)，目标为 (15345, 300)
        self.data = torch.tensor(data, dtype=torch.float32)  # 输入 PPG 信号
        self.targets = torch.tensor(targets, dtype=torch.float32)  # 目标 ECG 信号

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


# 定义Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(TransformerModel, self).__init__()
        self.d_model = 64
        self.linear_in = nn.Linear(input_dim, self.d_model)

        self.nhead = 4
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead,
                                                        dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(self.d_model, 1)  # 输出为单通道（1个ECG信号）

    def forward(self, src):
        # src shape: (batch_size, 300, 4)
        batch_size, seq_len, num_channels = src.shape

        # 将每个时间步的 4 个通道通过线性层嵌入到 d_model 维度
        src = self.linear_in(src)  # (batch_size, 300, d_model)

        # 转换为 Transformer 所需的输入形状
        src = src.transpose(0, 1)  # (300, batch_size, d_model)
        src = self.transformer_encoder(src)  # (300, batch_size, d_model)

        # 再转换回 (batch_size, 300, d_model)
        src = src.transpose(0, 1)

        # 解码成单通道输出
        output = self.decoder(src)  # (batch_size, 300, 1)

        # 删除最后一维 (batch_size, 300)
        output = output.squeeze(-1)
        output = torch.tanh(output)
        return output


# 定义训练函数
def train(model, train_loader, criterion, optimizer, epoch, train_losses):
    model.train()
    running_loss = 0.0
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # train_data.append({
        #     'inputs': inputs.detach().cpu().numpy(),
        #     'outputs': outputs.detach().cpu().numpy(),
        #     'targets': targets.detach().cpu().numpy()
        # })
    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    print(f'Epoch {epoch + 1}, Loss: {epoch_loss}')


# 定义测试函数
def test(model, test_loader, criterion, test_losses, epoch):
    model.eval()
    test_loss = 0.0
    all_inputs = []
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            test_loss += loss.item()

            all_inputs.append(inputs.detach().cpu().numpy())
            all_outputs.append(outputs.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())

    epoch_loss = test_loss / len(test_loader)
    test_losses.append(epoch_loss)
    print(f'Test Loss: {epoch_loss}')

    if epoch % 50 == 0:
        all_inputs = np.concatenate(all_inputs, axis=0)
        all_outputs = np.concatenate(all_outputs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        num_samples = min(9, len(all_inputs))  # 至多绘制9个样本
        rows = 3
        cols = 3

        # 创建 3x3 的子图布局
        fig, axes = plt.subplots(rows, cols, figsize=(15, 10))

        # 循环绘制每个样本
        for i in range(num_samples):
            row = i // cols
            col = i % cols
            ax = axes[row, col]

            ax.plot(all_inputs[i][:, 0], label='Input 1', linestyle='--')
            ax.plot(all_inputs[i][:, 1], label='Input 2', linestyle='--')
            ax.plot(all_inputs[i][:, 2], label='Input 3', linestyle='--')
            ax.plot(all_inputs[i][:, 3], label='Input 4', linestyle='--')
            ax.plot(all_outputs[i], label='Output', color='blue')
            ax.plot(all_targets[i], label='Target', linestyle=':', color='green')

            ax.set_title(f'Sample {i + 1}')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Amplitude')
            ax.legend()
            test_data.append({
                'inputs': inputs.cpu().numpy(),
                'outputs': outputs.cpu().numpy(),
                'targets': targets.cpu().numpy()
            })



        # 如果样本数不足9个，则隐藏多余的子图
        for i in range(num_samples, rows * cols):
            fig.delaxes(axes.flatten()[i])

        plt.tight_layout()
        plt.savefig(os.path.join(datapath, f'results/test_2_{epoch}_{i}.jpg'))



# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 导入数据
x_train = np.load(os.path.join(datapath, 'training_data/x.npy'))[:, :300, :]  # 形状为 (224304, 200, 4)
y_train = np.load(os.path.join(datapath, 'training_data/y.npy'))[:, :300]  # 形状为 (224304, 200)
x_test = np.load(os.path.join(datapath, 'test_data/x.npy'))[:, :300, :]  # 形状为 (num_test_samples, 200, 4)
y_test = np.load(os.path.join(datapath, 'test_data/y.npy'))[:, :300]  # 形状为 (num_test_samples, 200)

train_dataset = SequenceDataset(x_train, y_train)
test_dataset = SequenceDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 设置模型参数
input_dim = x_train.shape[2]  # 4 channels (PPG channels)
output_dim = y_train.shape[1]
hidden_dim = 512  # Transformer hidden dimension
num_layers = 2  # Number of transformer layers

# 实例化模型并移动到 GPU
model = TransformerModel(input_dim, hidden_dim, output_dim, num_layers).to(device)
model.load_state_dict(torch.load(os.path.join(datapath, 'results/transformer_model_initial.pth')))
print('Model loaded successfully.')

# 损失函数和优化器
criterion = WeightedMSELoss(threshold=0.1, weight=20.0)
optimizer = optim.Adam(model.parameters(), lr=0.002)

# 训练模型
num_epochs = 500
train_losses = []
test_losses = []

train_data = []
test_data = []
# 训练和测试函数调用保持不变
for epoch in range(num_epochs):
    train(model, train_loader, criterion, optimizer, epoch, train_losses)
    test(model, test_loader, criterion, test_losses, epoch)

# 训练结束后保存模型
torch.save(model.state_dict(), os.path.join(datapath, 'results/transformer_model_2.pth'))

# np.save('results/train_data_c1.npy', train_data)
# np.save('test_data.npy', test_data)
np.save(os.path.join(datapath, 'loss_data/train_losses_2.npy'), train_losses)
np.save(os.path.join(datapath, 'loss_data/test_losses_2.npy'), test_losses)
print('Model saved successfully.')

# 可视化损失曲线
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.savefig(os.path.join(datapath, 'loss_data/losses_2.jpg'))

# 加载模型并测试
model = TransformerModel(input_dim, hidden_dim, output_dim, num_layers).to(device)
model.load_state_dict(torch.load(os.path.join(datapath, 'results/transformer_model_2.pth')))
model.eval()  # 切换为评估模式
print('Model loaded successfully.')

# 测试加载后的模型
test(model, test_loader, criterion, test_losses, 0)
