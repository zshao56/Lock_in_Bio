import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# Dataset class
class PPGDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x = torch.tensor(x_data, dtype=torch.float32)  # (N, 200, 3)
        self.y = torch.tensor(y_data, dtype=torch.float32)  # (N, 200, 2)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # Use only the last time step for y
        return self.x[idx], self.y[idx, -1, :]  # (3, 200), (2,)

# Transformer model
class BPTransformer(nn.Module):
    def __init__(self, input_dim=3, d_model=64, num_heads=4, num_layers=3, dim_feedforward=128, dropout=0.1):
        super(BPTransformer, self).__init__()
        self.input_fc = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, 
                                                   dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_fc = nn.Linear(d_model, 2)

    def forward(self, x):
        # x: (N, 200, 3)
        x = self.input_fc(x)              # (N, 200, 64)
        x = self.transformer(x)           # (N, 200, 64)
        out = x[:, -1, :]                 # take the last time step: (N, 64)
        out = self.output_fc(out)         # (N, 2)
        return out

# Weighted MSE Loss
class WeightedMSELoss(nn.Module):
    def __init__(self, threshold=0.1, weight=2.0):
        super(WeightedMSELoss, self).__init__()
        self.threshold = threshold
        self.weight = weight

    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        weights = torch.where(diff > self.threshold, self.weight, 1.0)
        loss = weights * (pred - target) ** 2
        return torch.mean(loss)

# Training function
def train_model(model, train_loader, test_loader, optimizer, criterion, epochs, device):
    train_losses, test_losses = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                output = model(x_batch)
                loss = criterion(output, y_batch)
                val_loss += loss.item()
            avg_val_loss = val_loss / len(test_loader)
            test_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs}  Train Loss: {avg_train_loss:.4f}  Test Loss: {avg_val_loss:.4f}")

    return train_losses, test_losses

# Main
if __name__ == "__main__":
    # Load your data
    # Shape: x = (N, 200, 3), y = (N, 200, 2)
    x_train = np.load("training_data/x.npy")
    y_train = np.load("training_data/y.npy")
    x_test = np.load("test_data/x.npy")
    y_test = np.load("test_data/y.npy")

    train_dataset = PPGDataset(x_train, y_train)
    test_dataset = PPGDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BPTransformer().to(device)
    criterion = WeightedMSELoss(threshold=0.1, weight=2.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 100
    train_losses, test_losses = train_model(model, train_loader, test_loader, optimizer, criterion, epochs, device)

    # Save model
    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), "results/bp_transformer.pth")

    # Save loss curves
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Test Loss")
    plt.legend()
    plt.savefig("results/loss_curve.png")
    plt.close()

    print("Training finished.")
