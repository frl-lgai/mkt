from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from transformers import AutoConfig, AutoTokenizer
from transformers import Trainer, TrainingArguments, default_data_collator

import deepspeed

#from transformers import AutoConfig, AutoTokenizer, DataCollatorWithPadding
#from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from mkt import data, config
from mkt.models.reward import AutoModelForRewardModel
from mkt.utils import *


def padding_for_comparisons(batch, tokenizer, padding='longest'):
    pad = lambda batch: tokenizer.pad(batch, padding=padding, return_tensors='pt')


    batch_neg = [{
        'input_ids':      e['input_ids_neg'],
        'attention_mask': e['attention_mask_neg'],
    } for e in batch]

    batch, batch_neg = pad(batch), pad(batch_neg)
    batch['input_ids_neg'] = batch_neg['input_ids']
    batch['attention_mask_neg'] = batch_neg['attention_mask']

    return batch


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

    dataset = data.load_feedback(cfg.data_dir, tokenizer=tokenizer, num_proc=cfg.num_proc)
    dataset = dataset.with_format('torch')

    #import ipdb; ipdb.set_trace()

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset['train'],
        eval_dataset=dataset['valid'],
        #data_collator=lambda x: padding_for_comparisons(x, tokenizer),
        args=train_args
    )

    # Train!
    train_result = trainer.train()
    trainer.save_model()

    metrics = train_result.metrics
    #metrics["train_samples"] = len(dataset['train'])

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