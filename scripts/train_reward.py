from typing import Optional, List, Dict, Any
from dataclasses import dataclass

import numpy as np

from transformers import AutoConfig, AutoTokenizer
from transformers import Trainer, TrainingArguments, default_data_collator
from transformers import EvalPrediction

import deepspeed

from mkt import data, config
from mkt.models.reward import AutoModelForRewardModel
from mkt.utils import *


def padding_for_comparisons(batch, tokenizer, padding='longest'):
    pad = lambda batch: tokenizer.pad(batch, padding=padding, return_tensors='pt')

    batch_neg = pad([{
        'input_ids':      e['input_ids_neg'],
        'attention_mask': e['attention_mask_neg'],
    } for e in batch])
    
    batch = pad([{
        'input_ids':      e['input_ids'],
        'attention_mask': e['attention_mask'],
    } for e in batch])

    batch['input_ids_neg'] = batch_neg['input_ids']
    batch['attention_mask_neg'] = batch_neg['attention_mask']

    return batch


def compute_accuracy(eval_preds: EvalPrediction):
    logits_pos, logits_neg = eval_preds.predictions
    return {
        "accuracy": np.mean(logits_pos > logits_neg),
        "logit_pos": np.mean(logits_pos),
        "logit_neg": np.mean(logits_neg),
    }


def main(cfg):

    seed_everything(cfg.seed)
    wandb_init_distributed(cfg, __file__)

    train_args = TrainingArguments(**cfg.trainer)

    model_config = AutoConfig.from_pretrained(cfg.model_name_or_path)
    model_config.gradient_checkpointing = True
    model_config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForRewardModel.from_pretrained(cfg.model_name_or_path, config=model_config)
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(len(tokenizer))

    dataset = data.load_feedback(cfg.data_dir, tokenizer=tokenizer, num_proc=cfg.num_proc)
    dataset = dataset.with_format('torch')

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset['train'],
        eval_dataset=dataset['valid'],
        data_collator=lambda x: padding_for_comparisons(x, tokenizer),
        compute_metrics = compute_accuracy,
        args=train_args
    )

    #trainer.evaluate()
    train_result = trainer.train()
    trainer.save_model()

    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()


if __name__ == '__main__':

    import argparse
   
    parser = argparse.ArgumentParser(description="training_reward_models")
    parser.add_argument("--local_rank", type=int, default=0, help="local_rank for distributed training on gpus")
    parser.add_argument("--config", type=str, default="configs/reward_gpt2.yaml")    
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    
    cfg = config.load(args.config)#, config.RewardConfig)
    cfg.local_rank = args.local_rank
    
    main(cfg)