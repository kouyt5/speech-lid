# CCML 基于pytorch的深度学习框架

和 pytorch_lightning 功能类似的框架，相比于 pytorch_lightning 功能较少，且不完善，只是便于修改和一些实现比较麻烦的操作，目前基于这个框架完成了语种识别模型的设计（lid），除了ccml框架核心代码外，每个文件夹为一个独立算法功能，lm是基于神经网络的语言模型，rml是辐射源识别，mnist 机器学习果蝇实验，wav2vec-exp 基于wav2vec的语音识别。

框架使用：

可参考lid模块下main.py文件中的使用方法，其中核心为Trainer类，示例如下：

```python
import xxxx  # 参考lid 下的 main.py

@hydra.main(config_path="conf", config_name="xf_asr_lid")
def main(cfg: DictConfig) -> None:
    """
    代码启动主函数
    """
    train_conf = cfg["trainer"]
    model_conf = cfg["model"]
    module_conf = cfg["module"]
    data_conf = cfg["data"]
    comet_conf = cfg["logger"]["comet"]

    seed_everything(0)  # 随机数种子
    profile_callback = ProfileCallback()  # 性能监测
    ckpt_callback = CkptCallback(file_name_metric=["epoch", "loss"], save_topk=3)
    lr_callback = LrCallback()  # 学习率记录回调

    model_agent = LidModule(**model_conf)  # 模型训练封装
    train_dataset = MergedDataset(**data_conf)
    val_dataset = MergedDataset(**data_conf)
    dataloader_params = dict(data_conf["dataloader_params"])  # batch_size etc.
    comet_logger = CometLogger(**comet_conf)
    trainer = Trainer(
        callbacks=[ckpt_callback, lr_callback, profile_callback], loggers=[comet_logger], **train_conf
    )  # 训练器定义
    trainer.fit(
            model_agent,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            dataloader_params=dataloader_params,
        )

if __name__ == "__main__":
    main()
```