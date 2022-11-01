import os
import wandb

import torch.distributed as dist

#from transformers import AutoConfig, AutoModel, AutoTokenizer, GPTLingvoForCausalLM, AutoModelForCausalLM
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments, default_data_collator

import deepspeed

from mkt import data
from mkt.utils import *


def main(args):

    wandb.init(project=args.project, entity=args.entity, group=os.path.basename(__file__))

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=2,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=args.save_steps,
        load_best_model_at_end=True,
        logging_strategy="steps",
        logging_first_step=True,
        logging_steps=10,
        report_to="wandb",
        deepspeed=args.deepspeed_config,
        local_rank=args.local_rank,
    )

    config = AutoConfig.from_pretrained(args.model_dir)
    config.gradient_checkpointing = True
    config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_dir, config=config)
    model.resize_token_embeddings(len(tokenizer))

    train_dataset = data.load(args.data_dir, split='train', tokenizer=tokenizer, num_processes=1)
    train_dataset = data.prepare_for_language_modeling(train_dataset, block_size=1024, num_processes=1)
    train_dataset = train_dataset.with_format('torch')

    print(f'\n>>>> fin train_dataset {args.local_rank} <<<<\n')

    eval_dataset = data.load(args.data_dir, split='valid', tokenizer=tokenizer, num_processes=1)
    eval_dataset = data.prepare_for_language_modeling(eval_dataset, block_size=1024, num_processes=1)
    eval_dataset = eval_dataset.with_format('torch')

    print(f'\n>>>> fin eval_dataset {args.local_rank} <<<<\n')

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        args=train_args
    )

    train_results = trainer.train()

    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()


if __name__ == '__main__':
    seed_everything(42)

    import argparse
    parser = argparse.ArgumentParser(description="finetune_lingvo")

    parser.add_argument("--local_rank", type=int, default=0, help="local_rank for distributed training on gpus")

    parser.add_argument("--model_dir", type=str, default='/w/exaone_2022/model_8.8B_BI_MT_02')
    parser.add_argument("--data_dir", type=str, default='/w/data/mkt')
    parser.add_argument("--output_dir", type=str, default='/w/exp/mkt/model_8.8B_BI_MT_02')

    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=4e-5)
    parser.add_argument("--save_steps", type=int, default=100)

    parser.add_argument("--project", type=str, default='mkt')
    parser.add_argument("--entity", type=str, default='dhlee347')

    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()
    main(args)
