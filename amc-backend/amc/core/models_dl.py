import torch
import torch.nn as nn

class CNN1D(nn.Module):
    """1D CNN architecture."""
    def __init__(self, in_channels: int, sequence_length: int, num_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Output shape: (batch_size, 256, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.conv(x).squeeze(-1)
        return self.fc(features)

class CNNLSTM(nn.Module):
    """CNN + LSTM architecture."""
    def __init__(self, in_channels: int, sequence_length: int, num_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4), # sequence_length / 4
            nn.Dropout1d(p=0.4)  # Spatial Dropout
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),  # Increased dropout
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, in_channels, seq_len)
        features = self.conv(x)  # shape: (batch, 64, reduced_len)
        features = features.transpose(1, 2)  # shape: (batch, reduced_len, 64)
        lstm_out, (hn, cn) = self.lstm(features)  # hn shape: (2, batch, 64)
        
        # Concatenate the final forward hidden state and backward hidden state
        # hn[0] is the forward final state, hn[1] is the backward final state
        hn_cat = torch.cat((hn[0], hn[1]), dim=1)  # shape: (batch, 128)
        
        return self.fc(hn_cat)

class CNN2D(nn.Module):
    """2D CNN architecture operating on spectrograms (image-like representations)."""
    def __init__(self, in_channels: int, height: int, width: int, num_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Output shape: (batch_size, 128, 1, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # Input shape: (batch, in_channels, height, width)
        features = self.conv(x).squeeze(-1).squeeze(-1)
        return self.fc(features)
