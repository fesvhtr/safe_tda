import torch
import torch.nn as nn

class NSFWPatchMLPClassifierM(nn.Module):
    def __init__(self, input_dim=300):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 64),
            nn.ReLU(),

            nn.Linear(64, 1)  # 输出一个标量，用于 BCEWithLogitsLoss
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # 输出 shape: [batch_size]


class NSFWPatchMLPClassifierL(nn.Module):
    def __init__(self, input_dim=700): # input_dim 仍然作为参数传入
        super().__init__()

        self.net = nn.Sequential(
            # 第一个块
            nn.Linear(input_dim, 1024),      # 直接写入神经元数量
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.4),                 # 直接写入 Dropout 率

            # 第二个块
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),

            # 第三个块
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),

            # 输出层
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
