import torch
import torch.nn as nn

class NSFWPatchMLPClassifier(nn.Module):
    def __init__(self, input_dim=300):
        super(NSFWPatchMLPClassifier, self).__init__()

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
