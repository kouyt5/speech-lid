import torch
import torch.nn as nn
import torchmetrics


class MnistModel(nn.Module):
    
    def __init__(self, droprate:int = 0.1, hidden_dim:int = 128):
        super().__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(28*28, hidden_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(droprate),
            
            nn.Linear(hidden_dim, hidden_dim // 2, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(droprate)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(hidden_dim // 2, 10)
        )
        
        self.loss = nn.CrossEntropyLoss()
        self.acc = torchmetrics.Accuracy()
    
    def forward(self, x):
        
        x = x.view(x.size(0), -1)
        
        x = self.linear1(x)
        x = self.linear2(x)
        return x
    

        
