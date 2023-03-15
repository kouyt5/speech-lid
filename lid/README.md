# 面向复杂环境的语种识别

**模型：**
- [x] 基于Conformer的多语种语音识别模型
- [x] 基于无监督预训练模型（WavLM）的多语种语音识别模型
- [x] 基于词法（语音识别的后验）和语法特征（n-gram）的语种识别
- [x] ResNet 语种识别模型（分类模型）
- [x] X-Vector 语种识别模型（分类模型）
- [x] 面向噪声环境的语种识别（语音增强模型未开源，语种模块通过https进行调用）


**指标**

- [x] Cavg 开箱即用的torchmetrics集成（不需要trais分数文件，通过与各个类别之间进行遍历隐式的构建trais文件）
- [x] EER 指标的torchmetrics集成

**数据集：**

- [ ] common-voice
- [ ] 科大讯飞2020开发者比赛中的多语种语音识别数据集

## 预处理

首先需要构建数据集，在 raw_datasets.py 文件中可直接使用从common-voice下载的数据集，总的来说训练所需要的json文件格式如下：

```
{"path":"xxxx", "text": "xxxx", "lang": "cn",....}
{"path":"xxxx", "text": "xxxx", "lang": "en",....}
```

基于语音识别的语种识别还需要词典、语言模型等，词典txt文件示例如下：

```
a
b
c
```

语言模型训练建议使用kenlm，这里有一个docker封装好的库可直接生成语言模型文件：https://github.com/kouyt5/kenlm-docker

## 环境安装

在requirements文件夹下:

```
chmod a+x install.sh && ./install.sh
```
如还报错，根据提示安装对应环境

## 模型训练

有监督分类模型&无监督分类模型训练：

```shell
python main_cross.py --config-name xf_asr_lid model.last_model_name=resnet2
```

有监督语音识别模型训练：

```shell
python main_supervised --config-name xf_asr_supervised
```

无监督语音识别训练：

```shell
python main.py --config-name xf_asr_wavlm
```

配置文件在`config` 目录下，需要根据数据集实际位置和预训练模型路径进行设置，其中模型相关地址如下：

+ wavlm base: https://huggingface.co/s3prl/converted_ckpts/resolve/main/wavlm_base_plus.pt
+ wavlm large: https://huggingface.co/s3prl/converted_ckpts/resolve/main/wavlm_base_plus.pt
+ xlsr 300m: https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt

具体可参考仓库: https://github.com/s3prl/s3prl

所有预训练模型放在 wavlm/ckpts 文件夹下


## 模型测试

基于分类模型的测试
```shell
python test_cross.py --pt_path /path/to/pt_file --snr 100
```
`--snr` 参数表示信噪比，如果大于50，则不加噪声

基于有监督语音识别模型的测试：

```shell
python test_supervised --pt_path /path/to/pt_file
```

基于无监督语音识别的测试：

```shell
python test.py --pt_path xxxx --base_pt_path xxxx
```
`--base_pt_path` 参数表示预训练模型基于的原始模型路径，用于模型参数初始化

其中，部分参数需要手动的test.py文件中更改，例如语言模型路径、词典等。


相关测试命令可参考以下文件: `test_cross_all.sh`、`test_wavlm_all.sh`等