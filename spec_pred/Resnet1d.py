"""
resnet for 1-d signal data, pytorch version
 
Shenda Hong, Oct 2019
"""

import numpy as np
import torchmetrics
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            groups=self.groups,
        )

    def forward(self, x):

        net = x

        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)

        net = self.conv(net)

        return net


class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding
    """

    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):

        net = x

        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)

        net = self.max_pool(net)

        return net


class BasicBlock(nn.Module):
    """
    ResNet Basic Block
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        groups,
        downsample,
        use_bn,
        use_do,
        is_first_block=False,
        dr=0.5,
    ):
        super(BasicBlock, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.downsample = downsample
        if self.downsample:
            self.stride = stride
        else:
            self.stride = 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do

        # the first conv
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=dr)
        self.conv1 = MyConv1dPadSame(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            groups=self.groups,
        )

        # the second conv
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=dr)
        self.conv2 = MyConv1dPadSame(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            groups=self.groups,
        )

        self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x):

        identity = x

        # the first conv
        out = x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)

        # the second conv
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)

        # if downsample, also downsample identity
        if self.downsample:
            identity = self.max_pool(identity)

        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1, -2)
            ch1 = (self.out_channels - self.in_channels) // 2
            ch2 = self.out_channels - self.in_channels - ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1, -2)

        # shortcut
        out += identity

        return out


class ResNet1D(nn.Module):
    """

    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)

    Output:
        out: (n_samples)

    Pararmetes:
        in_channels: dim of input, the same as n_channel
        base_filters: number of filters in the first several Conv layer, it will double at every 4 layers
        kernel_size: width of kernel
        stride: stride of kernel moving
        groups: set larget to 1 as ResNeXt
        n_block: number of blocks
        n_classes: number of classes

    """

    def __init__(
        self,
        in_channels,
        base_filters,
        kernel_size,
        stride,
        groups,
        n_block,
        n_classes,
        downsample_gap=2,
        increasefilter_gap=4,
        use_bn=True,
        use_do=True,
        verbose=False,
        dr=0.5,
        rnn=False,
        use_snr_info=False,
    ):
        super(ResNet1D, self).__init__()

        self.verbose = verbose
        self.n_block = n_block
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do

        self.downsample_gap = downsample_gap  # 2 for base model
        self.increasefilter_gap = increasefilter_gap  # 4 for base model

        self.rnn = rnn  # 使用LSTM
        self.use_snr_info = use_snr_info
        if self.rnn:
            self.context_rnn = BatchLSTM(
                in_ch=base_filters,
                out_ch=base_filters // 2,
                batch_first=True,
                bidirection=True,
                num_layers=1,
            )
        # first block
        self.first_block_conv = MyConv1dPadSame(
            in_channels=in_channels,
            out_channels=base_filters,
            kernel_size=self.kernel_size,
            stride=1,
        )
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        out_channels = base_filters

        # residual blocks
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            # is_first_block
            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False
            # downsample at every self.downsample_gap blocks
            if i_block % self.downsample_gap == 1:
                downsample = True
            else:
                downsample = False
            # in_channels and out_channels
            if is_first_block:
                in_channels = base_filters
                out_channels = in_channels
            else:
                # increase filters at every self.increasefilter_gap blocks
                in_channels = int(
                    base_filters * 2 ** ((i_block - 1) // self.increasefilter_gap)
                )
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels

            tmp_block = BasicBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                groups=self.groups,
                downsample=downsample,
                use_bn=self.use_bn,
                use_do=self.use_do,
                is_first_block=is_first_block,
                dr=dr,
            )
            self.basicblock_list.append(tmp_block)

        # final prediction
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        # self.do = nn.Dropout(p=0.5)
        if self.use_snr_info:
            self.dense = nn.Linear(out_channels + 20, n_classes)
        else:
            self.dense = nn.Linear(out_channels, n_classes)

        # final prediction2
        self.dense2 = nn.Linear(out_channels, 20)

    def forward(self, x):

        out = x

        # first conv
        if self.verbose:
            print("input shape", out.shape)
        out = self.first_block_conv(out)
        if self.verbose:
            print("after first conv", out.shape)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)

        # residual blocks, every block has two conv
        count = 0
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            if self.verbose:
                print(
                    "i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}".format(
                        i_block, net.in_channels, net.out_channels, net.downsample
                    )
                )
            out = net(out)
            if self.verbose:
                print(out.shape)
            if self.rnn and count < 5:
                count += 1
                length = [out.size(2) for i in range(out.size(0))]  # B*E*T
                c, _ = self.context_rnn(out.transpose(1, 2), length)
                out = out + c.transpose(1, 2)

        # final prediction
        if self.use_bn:
            out = self.final_bn(out)
        out = self.final_relu(out)
        out = out.mean(-1)
        if self.verbose:
            print("final pooling", out.shape)
        # out = self.do(out)
        out_snr = self.dense2(out)
        if self.use_snr_info:
            out = torch.cat((out_snr, out), dim=-1)
        out = self.dense(out)
        if self.verbose:
            print("dense", out.shape)
        # out = self.softmax(out)
        if self.verbose:
            print("softmax", out.shape)

        return out, out_snr


class BatchLSTM(nn.Module):
    def __init__(
        self,
        in_ch=128,
        out_ch=128,
        batch_first=True,
        bidirection=True,
        num_layers=1,
        dropout=0.0,
    ):
        super().__init__()
        self.batch_first = batch_first
        self.rnn = nn.LSTM(
            in_ch,
            out_ch,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=bidirection,
            dropout=dropout,
        )

    def forward(self, x, length):
        x = nn.utils.rnn.pack_padded_sequence(
            x, enforce_sorted=False, lengths=length, batch_first=True
        )
        x, h = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        return x, _


class CCMLResNet1D(nn.Module):
    def __init__(
        self,
        in_channels,
        base_filters,
        kernel_size,
        stride,
        groups,
        n_block,
        n_classes,
        downsample_gap=2,
        increasefilter_gap=4,
        use_bn=True,
        use_do=True,
        verbose=False,
        dr=0.5,
        rnn=False,
        use_snr_info=False,
    ):
        super(CCMLResNet1D, self).__init__()
        self.model = ResNet1D(
            in_channels,
            base_filters,
            kernel_size,
            stride,
            groups,
            n_block,
            n_classes,
            downsample_gap,
            increasefilter_gap,
            use_bn,
            use_do,
            verbose,
            dr,
            rnn,
            use_snr_info,
        )
        self.loss = nn.CrossEntropyLoss()
        self.acc = torchmetrics.Accuracy()

    def forward(self, x):
        return self.model(x)

class Transformer(nn.Module):
    # https://github.com/nklingen/Transformer-Time-Series-Forecasting
    def __init__(self,in_size=512, out_size=512,num_layers=7,dropout=0.1):
        super(Transformer, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=in_size, nhead=8, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        self.decoder = nn.Linear(in_size,out_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        src = src.transpose(0, 1)  # -> (seq, batch, feature)
        mask = self._generate_square_subsequent_mask(len(src)).to(src.device)
        output = self.transformer_encoder(src,mask)
        output = torch.mean(output, dim=0)
        output = self.decoder(output)
        return output

if __name__ == "__main__":
    # model = ResNet1D(
    #     512, 1024, kernel_size=5, stride=2, groups=1, n_block=6, n_classes=512, rnn=True,
    #     downsample_gap=4, increasefilter_gap=5
    # )
    model  = Transformer(512,512, num_layers=5, dropout=0.1)
    x = torch.randn((8, 512, 40), dtype=torch.float32).transpose(1,2)
    x = torch.randn((8, 40, 512), dtype=torch.float32)
    
    out = model(x)
    print(out.shape)
    import transformers
    
    
    
    # model = BatchLSTM(128,128, batch_first=True, bidirection=True)
    # x = torch.randn((8, 128, 128), dtype=torch.float32)
    # out = model(x)
    # print(out)
