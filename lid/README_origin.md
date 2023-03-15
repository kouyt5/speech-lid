# Research on language identification And its implements

基于pytorch的复杂环境场景下的语种识别

## 研究点

- [ ] 语音增强算法
  - [ ] 基于SK的卷积端到端增强算法
  - [ ] Facebook 基于时域信号建模的增强模型
- [x] 语种识别算法
  - [ ] 基于LSTM的端到端语种识别
  - [x] 融合语音识别（自研orQuartzNet?）的鲁棒语种识别
  - [x] 基于无监督的语种识别(有没有可能没有任何效果？)
- [x] 算法训练
  - [ ] 多机分布式训练
  - [x] **简易框架**，不采用pytorch_lightning等，方便后续优化
  - [ ] torch-ort加速？
- [ ] 部署(工程实现)
  - [ ] 基于标准ONNX的部署（基于C？），TensorRT加速
  - [ ] 实现batch调度[rabbit-rpc-batch](https://github.com/kouyt5/rabbit-rpc-batch),提高系统的性能

## 算法工程设计

下面是系统实现的指导原则。
+ 尽量使用原生pytorch，方便模型等导入导出，以及修改。

### 架构设计

参考：
+ wenet（简易架构，没有过多的封装，但是训练功能简单，很多训练中的参数如错误率无法查看，只能查看到中间loss，没有提供多机训练适配。另外数据处理复杂，基于kaidi风格，需要适配shell脚本。但总体来说是性能比较好，设计简单的框架）
+ pytorch_lightning templete(自己搭建基于pytorch lightning框架的语音识别，但是该框架封装复杂，虽然提供了很多功能等集成，例如tensorboard、多卡训练等，并且训练过程可以很完美的在控制台打印。缺点也是他的优点，封装太复杂，实现一些比较奇怪的功能时可能非常麻烦，例如在训练过程中改变数据集，遇到错误后很难找到相关的文档)

#### 算法架构

+ 训练器(trainer(传入模型、优化器等等(参考wenet，是否是最优实践，考虑到分布式？)))
  + 训练和测试common的部分是什么？
+ 模型(CNN、LSTM)
  + 模型等具体参数写死在代码还是配置文件？
+ 数据采集处理（audio_processer(特征FBank...)）
+ 指标（wer、eer、acc等）
+ 日志打印收集器（tensorboard、wandb、python log，是否使用hydra？）
