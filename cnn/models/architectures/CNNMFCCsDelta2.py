import torch.nn as nn
import torch.nn.functional as F


class CNNMFCCsDelta2(nn.Module):
    def __init__(self, num_classes=9, dropout_rate=0.3):
        super(CNNMFCCsDelta2, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 2×2 downsampling
        self.dropout = nn.Dropout2d(dropout_rate)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Output shape: [B, 128, 1, 1]
        self.fc = nn.Linear(128, num_classes)  # Only one fully connected layer

    def forward(self, x):
        # Input: [B, 3, 40, 172]
        x = self.pool(self.conv1(x))  # [B, 16, 20, 86]
        x = self.pool(self.conv2(x))  # [B, 32, 10, 43]
        x = self.pool(self.conv3(x))  # [B, 64, 5, 21]
        x = self.pool(self.conv4(x))  # [B, 128, 2, 10]

        x = self.dropout(x)
        x = self.global_pool(x)       # [B, 128, 1, 1]
        x = x.view(x.size(0), -1)     # [B, 128]
        x = self.fc(x)                # [B, num_classes]

        return x