import hydra
from omegaconf import DictConfig, OmegaConf
import logging

from lid.loggers.wandb_logger import WandbLogger

# 为什么使用hydra？
# hydra提供简易的配置文件读取接口、实验日志按照时间归档管理，并且对关键代码侵入性
# 较小，因此在这里使用，虽然增加了代码的复杂性，但是比起自己实现一套配置文件读取以及
# python的args parser，代码更加结构化
@hydra.main(config_path='conf', config_name='lid_base')
def main(cfg: DictConfig) -> None:
    """
    代码启动主函数
    """
    logging.info("begin init...")
    conf = OmegaConf.to_yaml(cfg)
    # 训练器参数 amp_level等，分布式参数 local_rank world_size
    train_conf = conf['train']
    # 模型参数
    model_conf = conf['model']
    # 优化器参数
    optim_conf = conf['optim']
    # 数据加载参数
    data_conf = conf['data']
    # loss
    loss_conf = conf['loss']
    # 解码器参数
    decoder_conf = conf['decoder']
    # 日志参数(tensorboard, wandb)
    wandb_conf = conf['loggers']['wandb']
    wandb_logger = WandbLogger(project=wandb_conf['project'],wandb_id=wandb_conf['wandb_id'],
                               entity=wandb_conf['entity'], name=wandb_conf['name'], group=wandb_conf['group'])
    
    
    

if __name__=='__main__':
    main()


