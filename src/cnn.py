"""1D ResNet used for the Check-in 2 baseline (same architecture as notebooks/03_cnn_baseline.ipynb)."""
import torch
import torch.nn as nn


class ResBlock1d(nn.Module):
    def __init__(self, c_in, c_out, stride=1, kernel=7, dropout=0.2):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(c_in, c_out, kernel, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(
                nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False),
                nn.BatchNorm1d(c_out),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.drop(self.relu(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class ECGResNet1d(nn.Module):
    """Input (batch, 12, 1000) raw mV at 100 Hz -> 5 logits. Per-lead z-score stats live in buffers."""

    def __init__(self, n_leads=12, n_classes=5, dropout=0.2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(1, n_leads, 1))
        self.register_buffer("std", torch.ones(1, n_leads, 1))
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.blocks = nn.Sequential(
            ResBlock1d(64, 64, stride=1, dropout=dropout),
            ResBlock1d(64, 128, stride=2, dropout=dropout),
            ResBlock1d(128, 128, stride=1, dropout=dropout),
            ResBlock1d(128, 256, stride=2, dropout=dropout),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def set_normalization(self, X_train):
        """Per-lead mean/std from this model's own training records only."""
        X = torch.as_tensor(X_train, dtype=torch.float64)
        self.mean.copy_(X.mean(dim=(0, 2), keepdim=True).float())
        self.std.copy_(X.std(dim=(0, 2), keepdim=True, unbiased=False).float())

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.head(self.blocks(self.stem(x)))
