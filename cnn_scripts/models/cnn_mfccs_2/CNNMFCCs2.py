import torch.nn as nn
import torch.nn.functional as F


class CNNMFCCs2(nn.Module):
    def __init__(self, input_dim=40, num_classes=9, dropout_rate=0.3):
        super(CNNMFCCs2, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=input_dim,
                               out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        self.conv4 = nn.Conv1d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(256)

        # Global pooling to reduce to fixed-size
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.dropout = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: [B, 40, 172] → Conv1d expects (B, C_in, T)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, kernel_size=2)  # halve temporal dim

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, kernel_size=2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool1d(x, kernel_size=2)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.global_pool(x)  # [B, 256, 1]
        x = x.squeeze(-1)        # [B, 256]

        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)          # [B, num_classes]

        return x
