import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from torch.optim.lr_scheduler import ExponentialLR

# Dataset
class SequenceDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.tensor(data, dtype=torch.float32)  # (N, 3, 300)
        self.targets = torch.tensor(targets, dtype=torch.float32)  # (N, 300)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# ConvBiLSTM Model
class ConvBiLSTM(nn.Module):
    def __init__(self, input_channels, hidden_dim, seq_len, num_layers, num_filters, kernel_size, num_conv_layers=5):
        super(ConvBiLSTM, self).__init__()
        self.conv_layers = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=input_channels if i == 0 else num_filters,
                        out_channels=num_filters,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2
                    ),
                    nn.BatchNorm1d(num_filters),
                    nn.ReLU()
                ) for i in range(num_conv_layers)
            ]
        )
        self.bilstm = nn.LSTM(input_size=num_filters, hidden_size=hidden_dim, num_layers=num_layers,
                              batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        x = self.conv_layers(x)         # (N, 3, 300) → (N, 32, 300)
        x = x.transpose(1, 2)           # (N, 300, 32)
        x, _ = self.bilstm(x)           # (N, 300, 1024)
        x = self.fc(x)                  # (N, 300, 1)
        return x.squeeze(-1)            # (N, 300)

# Weighted MSE Loss
class WeightedMSELoss(nn.Module):
    def __init__(self, threshold=0.3, weight=20.0):
        super(WeightedMSELoss, self).__init__()
        self.threshold = threshold
        self.weight = weight

    def forward(self, outputs, targets):
        diff = torch.abs(outputs - targets)
        weights = torch.ones_like(diff)
        weights[diff > self.threshold] = self.weight
        loss = weights * (outputs - targets) ** 2
        return torch.mean(loss)

# Setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load data
datapath = "./"
x_train = np.load(os.path.join(datapath, 'training_data/x.npy'))  # (N, 300, 3)
y_train = np.load(os.path.join(datapath, 'training_data/y.npy'))[:, :300]  # (N, 300)
x_test = np.load(os.path.join(datapath, 'test_data/x.npy'))
y_test = np.load(os.path.join(datapath, 'test_data/y.npy'))[:, :300]

x_train = x_train[:, :300, :].transpose(0, 2, 1)  # (N, 3, 300)
x_test = x_test[:, :300, :].transpose(0, 2, 1)

train_dataset = SequenceDataset(x_train, y_train)
test_dataset = SequenceDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Model
input_channels = 3
hidden_dim = 512
seq_len = 300
num_layers = 3
num_filters = 32
kernel_size = 3
num_epochs = 200

model = ConvBiLSTM(input_channels, hidden_dim, seq_len, num_layers, num_filters, kernel_size).to(device)
criterion = WeightedMSELoss(threshold=0.3, weight=20.0)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.8, 0.9))
scheduler = ExponentialLR(optimizer, gamma=0.95)

# Logging
def get_next_case_dir(base_path, prefix='case_'):
    existing_cases = [d for d in os.listdir(base_path) if d.startswith(prefix)]
    case_numbers = [int(d[len(prefix):]) for d in existing_cases if d[len(prefix):].isdigit()]
    next_case_number = max(case_numbers, default=0) + 1
    return os.path.join(base_path, f'{prefix}{next_case_number}')

case_dir = get_next_case_dir(os.path.join(datapath, 'logs'))
writer = SummaryWriter(log_dir=case_dir)
print(f"Logs will be saved in: {case_dir}")

# Train/Test
def train_and_test(model, train_loader, test_loader, criterion, optimizer, epoch, train_losses, test_losses, writer):
    model.train()
    running_train_loss = 0.0
    running_test_loss = 0.0

    train_iter = iter(train_loader)
    test_iter = iter(test_loader)
    num_batches = max(len(train_loader), len(test_loader))

    for i in tqdm(range(num_batches), desc=f"Epoch {epoch + 1}"):
        try:
            train_inputs, train_targets = next(train_iter)
            train_inputs, train_targets = train_inputs.to(device), train_targets.to(device)
            optimizer.zero_grad()
            train_outputs = model(train_inputs)
            train_loss = criterion(train_outputs, train_targets)
            train_loss.backward()
            optimizer.step()
            running_train_loss += train_loss.item()
            writer.add_scalar('Step/Train_loss', train_loss.item(), epoch * num_batches + i)
        except StopIteration:
            pass

        model.eval()
        try:
            test_batch = next(test_iter)
        except StopIteration:
            test_iter = iter(test_loader)
            test_batch = next(test_iter)

        with torch.no_grad():
            try:
                test_inputs, test_targets = test_batch
                test_inputs, test_targets = test_inputs.to(device), test_targets.to(device)
                test_outputs = model(test_inputs)
                test_loss = criterion(test_outputs, test_targets)
                running_test_loss += test_loss.item()
                writer.add_scalar('Step/Test_loss', test_loss.item(), epoch * num_batches + i)
            except StopIteration:
                pass
        model.train()

    train_epoch_loss = running_train_loss / len(train_loader)
    test_epoch_loss = running_test_loss / len(test_loader)
    train_losses.append(train_epoch_loss)
    test_losses.append(test_epoch_loss)

    writer.add_scalar('Epoch/Train_loss', train_epoch_loss, epoch)
    writer.add_scalar('Epoch/Test_loss', test_epoch_loss, epoch)
    print(f"Epoch {epoch + 1} Train Loss: {train_epoch_loss:.6f}, Test Loss: {test_epoch_loss:.6f}")

# Training loop
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    train_and_test(model, train_loader, test_loader, criterion, optimizer, epoch, train_losses, test_losses, writer)

    # Interpolate beta values
    beta1 = 0.8 + (0.9 - 0.8) * ((epoch + 1) / num_epochs)
    beta2 = 0.9 + (0.999 - 0.9) * ((epoch + 1) / num_epochs)
    for param_group in optimizer.param_groups:
        param_group['betas'] = (beta1, beta2)

    scheduler.step()

# Save
os.makedirs(os.path.join(datapath, 'convresults'), exist_ok=True)
os.makedirs(os.path.join(datapath, 'convloss_data'), exist_ok=True)
torch.save(model.state_dict(), os.path.join(datapath, 'convresults/convbilstm_model.pth'))
np.save(os.path.join(datapath, 'convloss_data/train_losses.npy'), train_losses)
np.save(os.path.join(datapath, 'convloss_data/test_losses.npy'), test_losses)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.savefig(os.path.join(datapath, 'convloss_data/losses.jpg'))
plt.close()

writer.close()
print("Training complete and model saved.")
