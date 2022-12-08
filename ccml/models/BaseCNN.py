import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    def __init__(self, in_c:int = 1, classes:int = 6):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_c, 32, (7, 7), stride=2, padding=3),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, (5, 5), stride=2, padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.linear = nn.Linear(128, classes)
        
    def forward(self, x):
        x = self.cnn(x)
        x = self.avg_pool(x).view(x.size(0), x.size(1))
        x = self.linear(x)
        
        x = F.softmax(x, dim=-1)
        return x

if __name__=='__main__':
    model = CNNModel(1, 6)
    x = torch.randn(size=(8, 1, 128, 128))
    out = model(x)
    print()