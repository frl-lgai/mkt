from dataclasses import dataclass, field
from typing import List, Dict, Optional
from omegaconf import OmegaConf, MISSING

OmegaConf.clear_resolver("include")

OmegaConf.register_new_resolver("include", 
    lambda yaml_file: OmegaConf.load(yaml_file)
)

def load(yaml_file, config_class=None):
    cfg = OmegaConf.load(yaml_file)

    if config_class:
        cfg_schema = OmegaConf.structured(config_class)
        OmegaConf.merge(cfg_schema, cfg)

    OmegaConf.resolve(cfg)

    #print(OmegaConf.to_yaml(cfg))

    return cfg

def to_dict(config):
    return OmegaConf.to_container(config)

def to_yaml(config):
    return OmegaConf.to_yaml(config)


@dataclass
class HFTrainingArgumentsConfig:
    output_dir: str = "/w/exp/mkt/model_8.8B_BI_MT_02"
    
    do_train: bool = True
    do_eval: bool = True
    do_predict: bool = False

    num_train_epochs: int = 10
    
    learning_rate: float = 2e-5
    lr_scheduler_type: str ="cosine"
    weight_decay: float = 0.0
    warmup_ratio: float = 0.1
    
    per_device_train_batch_size: int = 10
    per_device_eval_batch_size: int = 2

    dataloader_drop_last: bool = True
    
    evaluation_strategy: str = "steps"
    eval_steps: int = 10
    eval_accumulation_steps: int = 1
    
    save_strategy: str = "steps"
    save_steps: int = 50
    
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "loss"
    greater_is_better: bool = False
    
    logging_strategy: str = "steps"
    logging_first_step: bool = True
    logging_steps: int = 10
    
    report_to: str = "wandb"  
    deepspeed: str = "../configs/stage3.json"


@dataclass
class WandbConfig:
    group: str = "finetune-lingvo"
    project: str = "mkt"
    entity: str = "frl-lgai"


@dataclass
class FinetuneConfig:
    seed: int = 42
    model_name_or_path: str = "/w/exaone_2022/model_1.7B_BI_MT_02"
    data_dir: str = "/w/data/mkt"
    max_length: int = 1024
    num_proc: int = 1 # number of processes in data processing
    eos_token: str = "[EOS]" # eos token for prompting

    trainer: HFTrainingArgumentsConfig = MISSING
    wandb: WandbConfig = MISSING

@dataclass
class RewardConfig:
    seed: int = 42
    model_name_or_path: str = "/w/exaone_2022/model_1.7B_BI_MT_02"
    data_dir: str = "/w/data/mkt"
    max_length: int = 1024
    num_proc: int = 1 # number of processes in data processing

    trainer: HFTrainingArgumentsConfig = MISSING
    wandb: WandbConfig = MISSING