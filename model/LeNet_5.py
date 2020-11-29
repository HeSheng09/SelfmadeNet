import torch.nn as nn
import torch


class LeNet_5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet_5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=(5, 5), stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(6, 16, kernel_size=(5, 5)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 120, kernel_size=(5, 5)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(120 * 1 * 1, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
