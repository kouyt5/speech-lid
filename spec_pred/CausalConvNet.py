import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils import weight_norm


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size=32, output_size=1, num_layers=1, dropout=0.25):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            nonlinearity='relu',    # 'tanh' or 'relu'
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        output, hidden = self.rnn(x, hidden)
        pred = self.linear(output[:, -1, :])
        return pred, hidden

    def init_hidden(self):
        return torch.randn(1, 24, self.hidden_size)


class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=1, dropout=0.25):
        super(SimpleGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            # dropout=dropout,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        output, hidden = self.gru(x, hidden)
        pred = self.linear(output[:, -1, :])
        return pred, hidden

    def init_hidden(self):
        return torch.randn(1, 24, self.hidden_size)


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=1, dropout=0.25):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        pred = self.linear(output[:, -1, :])
        return pred

    def init_hidden(self):
        return torch.randn(1, 24, self.hidden_size)
    
class AvgLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=1, dropout=0.25):
        super(AvgLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        pred = self.linear(torch.mean(output[:,:,:], dim=1))
        return pred

    def init_hidden(self):
        return torch.randn(1, 24, self.hidden_size)
    
class MLP(nn.Module):
    def __init__(self,input_size, output_size,num_layers,hidden_size) -> None:
        super(MLP, self).__init__()
        layers = []
        for i in range(num_layers):
            in_size = input_size
            out_size = hidden_size
            if i != 0:
                in_size = hidden_size
            if i == num_layers - 1:
                out_size = output_size
            layers += [
                nn.Sequential(
                    nn.Linear(in_size, out_size),
                    nn.ReLU()
                )
            ]
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x.transpose(1,2)).squeeze(-1)

class CnnLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, kernel_size, num_layers, dropout=0.1) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size, padding=kernel_size//2, stride=2),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.ReLU(),
            nn.Dropout1d(dropout),
            nn.Conv1d(hidden_size, 2*hidden_size, kernel_size, padding=kernel_size//2, stride=2),
            nn.BatchNorm1d(num_features=2*hidden_size),
            nn.ReLU(),
            nn.Dropout1d(dropout),
            nn.Conv1d(2*hidden_size, 2*hidden_size, kernel_size, padding=kernel_size//2, stride=1),
            nn.BatchNorm1d(num_features=2*hidden_size),
            nn.ReLU(),
            nn.Dropout1d(dropout),
        )
        self.lstm = SimpleLSTM(2*hidden_size, hidden_size, output_size, num_layers, dropout)
        
    def forward(self, x):
        x = self.cnn(x.transpose(1,2))  # B * hidden_dim * T
        x = self.lstm(x.transpose(1,2))
        return x

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        output = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        pred = self.linear(output[:, -1, :])
        return pred


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class STCN(nn.Module):
    def __init__(self, input_size, in_channels, output_size, num_channels, kernel_size, dropout):
        super(STCN, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(1, 1), stride=1, padding=0),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1, 1), stride=1, padding=0),
            torch.nn.BatchNorm2d(1),
            torch.nn.ReLU()
        )
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
    
if __name__ == "__main__":
    input_size = 128
    output_size = 128
    num_channels = [512,512,512,512]
    kernel_size = 4
    dropout = 0.1
    model = TCN(input_size, output_size, num_channels, kernel_size, dropout=dropout)
    model_lstm = SimpleLSTM(input_size, 512, 128, 2)
    model_mlp = MLP(100, 1, num_layers=4, hidden_size=256)
    model_cnn_lstm = CnnLSTM(input_size, input_size*2, output_size, kernel_size=5, num_layers=2, dropout=0.1)
    x = torch.randn((8, 128, 100)).transpose(1,2)
    out = model(x)  # -> (8,128)
    out2 = model_lstm(x)
    out3 = model_mlp(x)
    out4 = model_cnn_lstm(x)
    print(out.shape)