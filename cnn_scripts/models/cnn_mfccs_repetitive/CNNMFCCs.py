import torch.nn as nn
import torch.nn.functional as F


class CNNMFCCs(nn.Module):
    def __init__(self, input_dim=40, num_classes=9, dropout_rate=0.1):
        super(CNNMFCCs, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=input_dim,
                               out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1)

        self.dropout = nn.Dropout(dropout_rate)

        # 40 time steps preserved through all convs
        self.fc1 = nn.Linear(128 * 172, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Input shape: [B, 40, 172
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # flatten all except batch
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
