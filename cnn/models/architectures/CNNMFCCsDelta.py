import torch.nn as nn
import torch.nn.functional as F


class CNNMFCCsDelta(nn.Module):
    def __init__(self, num_classes=9, dropout_rate=0.1):
        super(CNNMFCCsDelta, self).__init__()

        # Input shape: [B, 3, 40, 172]
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample by 2
        self.dropout = nn.Dropout(dropout_rate)

        # After 2x2 pooling applied 4 times, spatial dims reduce:
        # Original: (40, 172) → (20, 86) → (10, 43) → (5, 21) → (2, 10)
        self.fc1 = nn.Linear(128 * 2 * 10, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Input: [B, 3, 40, 172]
        x = F.relu(self.conv1(x))  # → [B, 16, 40, 172]
        x = self.pool(x)           # → [B, 16, 20, 86]

        x = F.relu(self.conv2(x))  # → [B, 32, 20, 86]
        x = self.pool(x)           # → [B, 32, 10, 43]

        x = F.relu(self.conv3(x))  # → [B, 64, 10, 43]
        x = self.pool(x)           # → [B, 64, 5, 21]

        x = F.relu(self.conv4(x))  # → [B, 128, 5, 21]
        x = self.pool(x)           # → [B, 128, 2, 10]

        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # Flatten → [B, 128*2*10]
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
