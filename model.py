# model.py
import torch
import torch.nn as nn
import config  # 我们不再需要它来计算大小了，但最好留着

class SimpleSirenCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleSirenCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.flatten = nn.Flatten()

        # 2. 将 nn.Linear 替换为 nn.LazyLinear
        #    它只需要 "out_features" (输出大小)
        self.fc1 = nn.LazyLinear(64)
        # ------------------------

        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)

        # 当代码第一次运行到这里时，self.fc1 会自动
        # 检查 x 的形状 (e.g., [32, 5120])
        # 并将自己设置为 nn.Linear(5120, 64)
        x = self.fc1(x)

        x = self.relu4(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

