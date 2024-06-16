# video_model.py
import torch.nn as nn
import torch.nn.functional as F

class YourModel(nn.Module):
    def __init__(self, num_classes=101):
        super(YourModel, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * 56 * 56 * 28, num_classes)  # Adjust dimensions based on input size

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 56 * 56 * 28)
        x = self.fc1(x)
        return x
