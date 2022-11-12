from typing import Optional, List, Dict, Any
from dataclasses import dataclass

import numpy as np

from transformers import AutoConfig, AutoTokenizer
from transformers import Trainer, TrainingArguments, default_data_collator
from transformers import EvalPrediction

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
    return {"accuracy": np.mean(logits_pos > logits_neg)}


def main(cfg):

    train_args = TrainingArguments(
        output_dir="/w/exp/mkt/eval_reward_1.7B_BI_MT_02",
        do_train=False,
        do_predict=True,
        #per_device_train_batch_size=64,
        #per_device_eval_batch_size=64,
        dataloader_drop_last=False,
        eval_accumulation_steps=32,
    )

    model_config = AutoConfig.from_pretrained(cfg.model_name_or_path)
    model_config.gradient_checkpointing = True
    model_config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForRewardModel.from_pretrained(cfg.model_name_or_path, config=model_config)
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(len(tokenizer))

    dataset = data.load_feedback(cfg.data_dir, split='valid', tokenizer=tokenizer, num_proc=8)
    dataset = dataset.with_format('torch')

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        #train_dataset=dataset['train'],
        #eval_dataset=dataset['valid'],
        data_collator=lambda x: padding_for_comparisons(x, tokenizer),
        compute_metrics = compute_accuracy,
        args=train_args
    )

    results = trainer.predict(dataset)
    print(results)

if __name__ == '__main__':

    import argparse
   
    parser = argparse.ArgumentParser(description="evaluate_reward_models")
    parser.add_argument("--model_name_or_path", type=str, default="/w/exp/mkt/reward_1.7B_BI_MT_02/checkpoint-450")
    parser.add_argument("--data_dir", type=str, default="/w/mkt/data/kobaco/comparisons")
    args = parser.parse_args()
    
    main(args)
