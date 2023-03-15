import logging
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchvision
from torch.nn.utils.rnn import pad_sequence
from lid.s3prl_updream.wav2vec.wav2vec2_expert import UpstreamExpert
from lid.s3prl_updream.interfaces import Featurizer
from lid.wavlm.example import WavLMModel
from lid.model.resnet import ResNet18
from lid.model.xvector import XVEC

# dataprocess + pretrain + lastmodel
class PretrainLidModel(torch.nn.Module):
    def __init__(
        self,
        pt_path: str = None,
        dropout: float = 0.0,
        linear_dim: int = 768,
        num_class: int = 3,
        use_pre_train: bool = True,
        mask: bool = True,
        mask_channel_prob: float = 0.0,
        mask_prob: float = 0.0,
        last_model_name: str = "xvector",  # 最后分类层名字, xvector, i-vector etc.
        pre_train_name: str = "wavlm",
    ) -> None:
        super().__init__()
        self.data_processor = DataProcessor()
        if pre_train_name == "wavlm":
            self.pre_train_model = WavLMMPretrainModel(
                pt_path=pt_path,
                use_pre_train=use_pre_train,
                mask=mask,
                mask_channel_prob=mask_channel_prob,
                mask_prob=mask_prob,
            )
        else:
            self.pre_train_model = Wav2vecPretrainModel(
                pt_path=pt_path,
                use_pre_train=use_pre_train,
                mask=mask,
                mask_channel_prob=mask_channel_prob,
                mask_prob=mask_prob,
            )
        logging.info(f"last model name: {last_model_name}")
        if last_model_name == "xvector":
            self.lang_discriminator = XVectorModel(linear_dim, num_class)
        elif last_model_name == "linear":
            self.lang_discriminator = LinearModel(linear_dim, num_class)

    def forward(self, x, sr=16000):
        x = self.data_processor(x, sr)
        x, percent = self.pre_train_model(x)  # feature, feature length percent((B, L, F), [0.1, ..., 0.4])
        x = x.transpose(1,2).unsqueeze(1)  # -> (B, 1, F, L)
        x = self.lang_discriminator(x, percent)  # -> (B, C)
        return x  # not pass to a Softmax Layer
    
    def freeze_feature_extractor(self):
        model_freezes = []
        if hasattr(self.pre_train_model.featurizer, "model"):
            model_freezes.append(self.pre_train_model.featurizer.model.feature_extractor)
            model_freezes.append(self.pre_train_model.featurizer.model.post_extract_proj)
        # model_freezes.append(self.featurizer.upstream.model.mask_emb)
        for model in model_freezes:
            for params in model.parameters():
                params.requires_grad = False

    def unfreeze_feature_extractor(self):
        model_freezes = []
        if hasattr(self.pre_train_model.featurizer, "model"):
            model_freezes.append(self.pre_train_model.featurizer.model.feature_extractor)
            model_freezes.append(self.pre_train_model.featurizer.model.post_extract_proj)
        # model_freezes.append(self.featurizer.upstream.model.mask_emb)
        for model in model_freezes:
            for params in model.parameters():
                params.requires_grad = True

    def freeze_tranformer_encoder(self):
        model_freezes = []
        if hasattr(self.pre_train_model.featurizer, "model"):
            model_freezes.append(self.pre_train_model.featurizer.model.encoder)
        for model in model_freezes:
            for params in model.parameters():
                params.requires_grad = False

    def unfreeze_tranformer_encoder(self):
        model_freezes = []
        if hasattr(self.pre_train_model.featurizer, "model"):
            model_freezes.append(self.pre_train_model.featurizer.model.encoder)
        for model in model_freezes:
            for params in model.parameters():
                params.requires_grad = True

    def reset_param(self):
        logging.info("reset parameters...")
        for layer in self.pre_train_model.modules():
            if hasattr(layer, "reset_parameters"):
                logging.debug(f"reset {layer._get_name()}")
                layer.reset_parameters()
            else:
                logging.debug(f"reset ignore {layer._get_name()}")
                
class LidModel(torch.nn.Module):
    def __init__(
        self,
        dropout: float = 0.0,
        linear_dim: int = 80,
        num_class: int = 3,
        mask: bool = True,
        last_model_name:str = "xvector" # 最后分类层名字, xvector, i-vector etc.
    ) -> None:
        super().__init__()
        self.data_processor = DataProcessor()
        
        if last_model_name == "xvector":
            self.lang_discriminator = XVectorModel(linear_dim, num_class)
        elif last_model_name == "linear":
            self.lang_discriminator = LinearModel(linear_dim, num_class)
        elif last_model_name == "resnet":
            self.lang_discriminator = LidResnet(linear_dim, num_class)
        elif last_model_name == "resnet2":
            self.lang_discriminator = LidResnetWeSpeaker(linear_dim, num_class)
        elif last_model_name == "xvector2":
            self.lang_discriminator = LidXvectorWeSpeaker(linear_dim, num_class)

    def forward(self, x, sr=16000):
        # (Batch, T, n_mels)
        x = x.transpose(1,2).unsqueeze(1)  # -> (B, 1, F, L)
        x = self.lang_discriminator(x)  # -> (B, C)
        return x  # not pass to a Softmax Layer
    
    def freeze_feature_extractor(self):
        pass

    def unfreeze_feature_extractor(self):
        pass

    def freeze_tranformer_encoder(self):
        pass

    def unfreeze_tranformer_encoder(self):
        pass

    def reset_param(self):
        pass
    
class DataProcessor(torch.nn.Module):
    """
    数据预处理, 将音频原始波形转换为模型输入的数据
    """

    def __init__(self, target_rate: int = 16000) -> None:
        super().__init__()
        self.target_rate = target_rate
        self.resampler22k = torchaudio.transforms.Resample(
            orig_freq=22050, new_freq=target_rate
        )
        self.resampler441k = torchaudio.transforms.Resample(
            orig_freq=44100, new_freq=target_rate
        )

    def forward(self, x, sample_rate: int = 16000):
        """
        输入进行resample, 如果未指明,默认为16000
        Args:
            x (List[torch.Tensor]): 输入的多个音频
            sample_rate (int): 采样率 16000 44100 etc.

        Return:
            (List[torch.Tensor]): 和输入格式相同
        """
        if sample_rate == 16000:
            return x
        longest_len = max(x, key=lambda y: y.size(-1)).shape[-1]
        wav_percent = [x[i].shape[-1] / longest_len for i in range(len(x))]
        x = pad_sequence(x, batch_first=True)
        if sample_rate == 22050:
            x = self.resampler22k(x)
        elif sample_rate == 44100:
            x = self.resampler441k(x)
        else:
            return x
        wav_len = [int(percent * x.shape[-1]) for percent in wav_percent]
        x = self.unpad_sequence(x, wav_len)
        return x

    def unpad_sequence(self, x: torch.Tensor, wav_len: List):
        """

        Args:
            x (torch.Tensor): 重采样后的数据
            wav_len (List): 重采样后长度
        """
        return [x[i, : wav_len[i]] for i in range(len(wav_len))]


class WavLMMPretrainModel(torch.nn.Module):
    """
    wavlm模型和额外线形层, 输出语音识别结果
    """

    def __init__(
        self,
        pt_path: str = None,
        use_pre_train: bool = True,
        mask: bool = True,
        mask_channel_prob: float = 0.0,
        mask_prob: float = 0.0,
    ) -> None:
        super().__init__()
        logging.info(f"mask channel prob: {mask_channel_prob}, mask_prob {mask_prob}")
        if not mask:
            mask_channel_prob = 0.0
            mask_prob = 0
        self.featurizer = WavLMModel(
            pt_path, use_pre_train, mask_channel_prob, mask_prob
        )

    def forward(self, batch):
        feature = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)
        pad_mask = torch.ones_like(feature, device=feature.device)
        for i in range(len(batch)):
            pad_mask[i, : batch[i].size(0)] = 0
        feature = self.featurizer(feature, pad_mask)

        max_len = max(batch, key=lambda x: x.shape[0]).shape[0]
        percents = [item.shape[0] / max_len for item in batch]
        return feature, percents
    
class Wav2vecPretrainModel(torch.nn.Module):
    """
    wav2vec模型和额外线形层, 输出语音识别结果
    """

    def __init__(
        self,
        pt_path: str = None,
        use_pre_train: bool = True,
        mask: bool = True,
        mask_channel_prob: float = 0.0,
        mask_prob: float = 0.0,
    ) -> None:
        super().__init__()
        logging.info(f"mask channel prob: {mask_channel_prob}, mask_prob {mask_prob}")
        if not mask:
            mask_channel_prob = 0.0
            mask_prob = 0
        self.featurizer = Featurizer(
            upstream=UpstreamExpert(ckpt=pt_path, drop_layer=True, mask=mask),
            feature_selection="last_hidden_state",  # last_hidden_state, hidden_state_{0-24}
            upstream_device="cpu",
            layer_selection=None,  # 选择后的第几层特征 0-24
        )

    def forward(self, batch):
        feature = self.featurizer.upstream(batch)
        feature = self.featurizer(batch, feature)
        max_len = max(batch, key=lambda x: x.shape[0]).shape[0]
        percents = [item.shape[0] / max_len for item in batch]
        return feature, percents
    
"""
@author: cvqluu
repo: https://github.com/cvqluu/TDNN
"""
class TDNN(nn.Module):

    def __init__(
            self,
            input_dim=23,
            output_dim=512,
            context_size=5,
            stride=1,
            dilation=1,
            batch_norm=False,
            dropout_p=0.2
    ):
        '''
        TDNN as defined by https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf
        Affine transformation not applied globally to all frames but smaller windows with local context
        batch_norm: True to include batch normalisation after the non linearity

        Context size and dilation determine the frames selected
        (although context size is not really defined in the traditional sense)
        For example:
            context size 5 and dilation 1 is equivalent to [-2,-1,0,1,2]
            context size 3 and dilation 2 is equivalent to [-2, 0, 2]
            context size 1 and dilation 1 is equivalent to [0]
        '''
        super(TDNN, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm

        self.kernel = nn.Linear(input_dim * context_size, output_dim)
        self.nonlinearity = nn.ReLU()
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(output_dim)
        if self.dropout_p:
            self.drop = nn.Dropout(p=self.dropout_p)

    def forward(self, x):
        '''
        input: size (batch, seq_len, input_features)
        outpu: size (batch, new_seq_len, output_features)
        '''

        _, _, d = x.shape
        assert (d == self.input_dim), 'Input dimension was wrong. Expected ({}), got ({})'.format(self.input_dim, d)
        x = x.unsqueeze(1)

        # Unfold input into smaller temporal contexts
        x = F.unfold(
            x,
            (self.context_size, self.input_dim),
            stride=(1, self.input_dim),
            dilation=(self.dilation, 1)
        )

        # N, output_dim*context_size, new_t = x.shape
        x = x.transpose(1, 2)
        x = self.kernel(x.float())
        x = self.nonlinearity(x)

        if self.dropout_p:
            x = self.drop(x)

        if self.batch_norm:
            x = x.transpose(1, 2)
            x = self.bn(x)
            x = x.transpose(1, 2)

        return x


class X_vector(nn.Module):
    def __init__(self, input_dim=40, num_classes=8):
        super(X_vector, self).__init__()
        self.tdnn1 = TDNN(input_dim=input_dim, output_dim=512, context_size=5, dilation=1, dropout_p=0.5)
        self.tdnn2 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=1, dropout_p=0.5)
        self.tdnn3 = TDNN(input_dim=512, output_dim=512, context_size=2, dilation=2, dropout_p=0.5)
        self.tdnn4 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1, dropout_p=0.5)
        self.tdnn5 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=3, dropout_p=0.5)
        #### Frame levelPooling
        self.segment6 = nn.Linear(1024, 512)
        self.segment7 = nn.Linear(512, 512)
        self.output = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        tdnn1_out = self.tdnn1(inputs)
        tdnn2_out = self.tdnn2(tdnn1_out)
        tdnn3_out = self.tdnn3(tdnn2_out)
        tdnn4_out = self.tdnn4(tdnn3_out)
        tdnn5_out = self.tdnn5(tdnn4_out)
        ### Stat Pool

        mean = torch.mean(tdnn5_out, 1)
        std = torch.var(tdnn5_out, 1)
        stat_pooling = torch.cat((mean, std), 1)
        segment6_out = self.segment6(stat_pooling)
        x_vec = self.segment7(segment6_out)
        predictions = self.output(x_vec)
        return predictions, x_vec


class XVectorModel(nn.Module):
    def __init__(self, input_dim=23, num_classes=8):
        super(XVectorModel, self).__init__()
        self.model = X_vector(input_dim=input_dim, num_classes=num_classes)

    def forward(self, inputs, _=None):
        inputs = inputs.squeeze(1).transpose(1, 2)
        return self.model(inputs)[0]


class LinearModel(nn.Module):
    def __init__(self, input_dim=768, num_classes=3):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(2*input_dim, num_classes)
        
    def forward(self, x, _=None):
        # x (B, 1, F, L)
        x = x.squeeze(1).transpose(1, 2)  # -> (B, L, F)
        mean = torch.mean(x, dim=1)
        std = torch.var(x, dim=1)
        x = torch.cat((mean, std), dim=1)  # (B, 2*F)
        x = self.linear(x)
        return x
    
class LidResnet(nn.Module):
    def __init__(self, input_dim=768, num_classes=3) -> None:
        super(LidResnet, self).__init__()
        self.flatten = nn.Conv2d(1, 3, kernel_size=1)
        self.resnet = torchvision.models.resnet18(num_classes=num_classes)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.resnet(x)
        return x

class LidResnetWeSpeaker(nn.Module):
    def __init__(self, input_dim=768, num_classes=3) -> None:
        super(LidResnetWeSpeaker, self).__init__()
        self.resnet = ResNet18(feat_dim=80, embed_dim=256, pooling_func='MQMHASTP')
        self.last_linear = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = x.squeeze(1).transpose(1, 2)  # -> (B, L, F)
        x = self.resnet(x)  # (B, F')
        x = self.last_linear(x[-1])  # (B, num_classes)
        return x
    
class LidXvectorWeSpeaker(nn.Module):
    def __init__(self, input_dim=768, num_classes=3) -> None:
        super(LidXvectorWeSpeaker, self).__init__()
        self.resnet = XVEC(feat_dim=80, embed_dim=256, pooling_func='TSTP')
        self.last_linear = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = x.squeeze(1).transpose(1, 2)  # -> (B, L, F)
        x = self.resnet(x)  # (B, F')
        x = self.last_linear(x[-1])  # (B, num_classes)
        return x
    
if __name__ == '__main__':
    # model = X_vector(input_dim=40, num_classes=3)
    model_one = XVectorModel(input_dim=768, num_classes=3)
    data = torch.randn((8, 1, 768, 512), dtype=torch.float32)
    out = model_one(data)
    
    x = [
        torch.randn(
            16000,
        ),
        torch.randn(
            17000,
        ),
    ]
    model = PretrainLidModel(pt_path="/hdd/1/chenc/lid/speech-lid/lid/wavlm/ckpts/WavLM-Base-plus.pt")
    out = model(x)
    print()
