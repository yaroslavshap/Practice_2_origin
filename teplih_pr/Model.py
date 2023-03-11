import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1)
        self.head = nn.Linear(in_features=65536, out_features=10)

    def forward(self, data):
        out = self.pool(self.conv1(data))
        out = self.pool(self.conv2(out))
        out = self.pool(self.conv3(out))
        out = self.pool(self.conv4(out))
        out = self.pool(self.conv5(out))
        b, c, h, w = out.shape

        head = self.head(out.reshape(b, c*h*w))

        return head