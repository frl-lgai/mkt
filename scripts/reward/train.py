from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from transformers import AutoConfig, AutoTokenizer
from transformers import Trainer, TrainingArguments, default_data_collator
from tokenizer import PreTrainedTokenizerBase

import deepspeed

#from transformers import AutoConfig, AutoTokenizer, DataCollatorWithPadding
#from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from mkt import data, config
from mkt.models import AutoModelForRewardModel
from mkt.utils import *


def padding_for_comparisons(batch, tokenizer, padding='longest'):
    pad = lambda batch: tokenizer.pad(batch, padding=padding, return_tensors='pt')

    # [{'input_ids': [1,3,4]}, {...}] -> {'input_ids': tensor[[1,3,4], [...]]}
    batch = default_data_collator(batch) 
    batch_neg = {'input_ids': batch['input_ids_neg'], 'attention_mask': batch['attention_mask_neg'],}

    return {**pad(batch), **pad(batch_neg)}


def main(cfg):

    seed_everything(cfg.seed)
    wandb_init_distributed(cfg, __file__)

    train_args = TrainingArguments(**cfg.trainer)

    model_config = AutoConfig.from_pretrained(cfg.model_dir)
    model_config.gradient_checkpointing = True
    model_config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_dir)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForRewardModel.from_pretrained(cfg.model_dir, config=model_config)
    #model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(len(tokenizer))

    train_dataset = data.load_feedback(cfg.data_dir, split='train', tokenizer=tokenizer, num_proc=cfg.num_proc)
    train_dataset = train_dataset.with_format('torch')

    eval_dataset = data.load_feedback(cfg.data_dir, split='valid', tokenizer=tokenizer, num_proc=cfg.num_proc)
    eval_dataset = eval_dataset.with_format('torch')

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=lambda x: padding_for_comparisons(x, tokenizer),
        args=train_args
    )

    # Train!
    train_result = trainer.train()
    trainer.save_model()

    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset['train'])

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == '__main__':

    import argparse
   
    parser = argparse.ArgumentParser(description="training_reward_models")
    parser.add_argument("--local_rank", type=int, default=0, help="local_rank for distributed training on gpus")
    parser.add_argument("--config", type=str, default="configs/reward_gpt2.yaml")    
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    
    cfg = config.load(args.config, config.FinetuneConfig)
    cfg.local_rank = args.local_rank
    
    main(cfg)