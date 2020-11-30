import torch.nn as nn
import torch


class SelfmadeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SelfmadeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 16, kernel_size=3, stride=1),
        )
        self.classifier = nn.Sequential(
            # nn.Linear(16 * 4 * 4, 96),
            nn.Linear(16 * 4 * 4, 128),
            nn.ReLU(),
            # nn.Linear(96, num_classes)
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
